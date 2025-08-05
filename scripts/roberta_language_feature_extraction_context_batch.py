import os
from pathlib import Path
import numpy as np
import pandas as pd
import string
import h5py
from tqdm import tqdm
import json

# Add imports for torch and transformers
import torch
from transformers import RobertaModel, RobertaTokenizer


def load_expected_counts(path):
    """
    Load expected counts from a JSON file.
    The file should be a JSON dictionary with {filename: expected_row_count}.
    """
    with open(path, 'r') as f:
        expected_counts = json.load(f)
    return expected_counts


root_data_dir = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data"


class RobertaFeatureExtractor:
    def __init__(self, context_window=1, middle_layer=7, max_tokens=510, kept_tokens=10):
        """
        Initialize RoBERTa feature extractor with configurable parameters

        Args:
            context_window: Number of TRs to include in context window (default 1 = single TR)
            middle_layer: Which transformer layer to extract features from (default 7)
            max_tokens: Maximum number of tokens to process (default 510)
            kept_tokens: Number of tokens to keep from last hidden state (default 10)
        """
        self.context_window = context_window
        self.middle_layer_index = middle_layer
        self.max_tokens = max_tokens
        self.kept_tokens = kept_tokens

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = RobertaModel.from_pretrained("roberta-base").to(self.device).eval()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.hidden_size = self.model.config.hidden_size

    def pad_transcript_df(self, df, expected_rows):
        """Pad or truncate dataframe to expected length"""
        current_rows = df.shape[0]
        if current_rows < expected_rows:
            print(f"Padding from {current_rows} to {expected_rows} rows")
            n_pad = expected_rows - current_rows
            pad_df = pd.DataFrame({col: [''] * n_pad for col in df.columns})
            df = pd.concat([df, pad_df], ignore_index=True)
        elif current_rows > expected_rows:
            print(f"Warning: transcript longer ({current_rows}) than expected ({expected_rows}), truncating")
            df = df.iloc[:expected_rows].copy()
        else:
            print(f"No padding needed: {current_rows} rows")
        return df

    def get_context_text(self, df, current_idx):
        """Get concatenated text for context window"""
        start_idx = max(0, current_idx - self.context_window + 1)
        context_texts = []

        for i in range(start_idx, current_idx + 1):
            if i < len(df) and not df.iloc[i]["is_na"]:
                context_texts.append(df.iloc[i]["text_per_tr"])

        return " ".join(context_texts) if context_texts else ""

    def extract_features(self, tsv_path, expected_rows):
        """Extract features from a TSV file"""
        print(f"\nLoading TSV file: {tsv_path}")
        with open(tsv_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # minus header line
        print(f"Total lines in file (excluding header): {total_lines}")

        # Read TSV with no NA filtering
        df = pd.read_csv(
            tsv_path,
            sep='\t',
            keep_default_na=False,
            na_filter=False,
            dtype=str
        )
        print(f"Rows read by pandas: {df.shape[0]}")

        # Pad or truncate dataframe to expected_rows
        df = self.pad_transcript_df(df, expected_rows)

        # Mark missing or empty 'text_per_tr' explicitly
        df["is_na"] = df["text_per_tr"].apply(lambda x: (x is None) or (str(x).strip() == ""))
        print(f"Number of rows with empty/missing text_per_tr: {df['is_na'].sum()}")

        if df['is_na'].sum() > 0:
            print("Rows with empty or missing 'text_per_tr':")
            print(df.loc[df['is_na'], ['text_per_tr']])

        pooler_output, last_hidden_state = [], []

        for i in tqdm(range(df.shape[0]), desc=f"Extracting {Path(tsv_path).stem}"):
            context_text = self.get_context_text(df, i)

            if context_text:  # Only process if we have text in context window
                try:
                    # Process pooled output
                    inputs = self.tokenizer(context_text, return_tensors="pt", truncation=True, max_length=self.max_tokens + 2)
                    input_tensor = inputs['input_ids'].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_tensor, output_hidden_states=True)
                        pooled = outputs.hidden_states[self.middle_layer_index][0, 0].cpu().numpy()
                except Exception as e:
                    print(f"[ERROR] Pooler output extraction failed at row {i}: {e}")
                    pooled = np.full(self.hidden_size, np.nan, dtype='float32')

                try:
                    # Process last hidden state
                    np_text = context_text.translate(str.maketrans('', '', string.punctuation))
                    np_inputs = self.tokenizer(np_text, return_tensors="pt", truncation=True, max_length=self.max_tokens + 2)
                    np_tensor = np_inputs['input_ids'].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(np_tensor, output_hidden_states=True)
                        np_out = outputs.hidden_states[self.middle_layer_index][0][1:-1].cpu().numpy()
                    feat = np.full((self.kept_tokens, self.hidden_size), np.nan, dtype='float32')
                    tk_idx = min(self.kept_tokens, np_out.shape[0])
                    feat[-tk_idx:] = np_out[-tk_idx:]
                except Exception as e:
                    print(f"[ERROR] Last hidden state extraction failed at row {i}: {e}")
                    feat = np.full((self.kept_tokens, self.hidden_size), np.nan, dtype='float32')
            else:
                pooled = np.full(self.hidden_size, np.nan, dtype='float32')
                feat = np.full((self.kept_tokens, self.hidden_size), np.nan, dtype='float32')

            pooler_output.append(pooled)
            last_hidden_state.append(feat)

        print(f"Finished extraction: pooler_output length = {len(pooler_output)}, last_hidden_state length = {len(last_hidden_state)}")

        assert len(pooler_output) == df.shape[0], f"Length mismatch: pooler_output={len(pooler_output)}, TSV rows={df.shape[0]}"
        assert len(last_hidden_state) == df.shape[0], f"Length mismatch: last_hidden_state={len(last_hidden_state)}, TSV rows={df.shape[0]}"

        return np.array(pooler_output, dtype='float32'), np.array(last_hidden_state, dtype='float32')


def process_folder(folder_path, extractor, save_dir_features, expected_counts):
    """Process all TSV files in a folder with the given extractor"""
    folder_path = Path(folder_path).resolve()
    movies_root = Path("/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies").resolve()
    transcripts_root = Path("/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts").resolve()

    print(f"Processing folder: {folder_path}")
    print(f"Context window size: {extractor.context_window} TRs")

    try:
        relative_path = folder_path.relative_to(movies_root)
    except ValueError as e:
        print(f"Error: {folder_path} is not under {movies_root}")
        return

    transcript_folder = transcripts_root / relative_path
    if not transcript_folder.exists():
        print(f"Transcript folder not found: {transcript_folder}")
        return

    # Create output folder with context window info
    output_folder = Path(save_dir_features) / f"ctx{extractor.context_window}" / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Will save features to: {output_folder}")

    tsv_files = sorted(transcript_folder.glob("*.tsv"))
    print(f"Found {len(tsv_files)} .tsv files in {transcript_folder}")

    for tsv_file in tsv_files:
        save_path = output_folder / f"{tsv_file.stem}_features_language.h5"
        if save_path.exists():
            print(f"Skipping {tsv_file.name}, features already exist.")
            continue

        expected_rows = expected_counts.get(tsv_file.name)
        if expected_rows is None:
            print(f"Warning: No expected row count for {tsv_file.name}, skipping")
            continue

        pooler_output, last_hidden_state = extractor.extract_features(
            str(tsv_file), expected_rows
        )

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('pooler_output', data=pooler_output, dtype=np.float32)
            f.create_dataset('last_hidden_state', data=last_hidden_state, dtype=np.float32)
        print(f"Saved features to {save_path}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Path to movie folder to process")
    parser.add_argument("--context_window", type=int, default=1,
                        help="Number of TRs to include in context window")
    parser.add_argument("--middle_layer", type=int, default=7,
                        help="Which transformer layer to extract features from")
    parser.add_argument("--max_tokens", type=int, default=510,
                        help="Maximum number of tokens to process")
    parser.add_argument("--kept_tokens", type=int, default=10,
                        help="Number of tokens to keep from last hidden state")
    parser.add_argument("--output_dir", default="../results/roberta_language_features",
                        help="Base directory to save features")
    parser.add_argument("--expected_counts_file", default="../data/expected_counts.json",
                        help="Path to JSON file containing expected counts for transcripts")

    args = parser.parse_args()

    # Load expected counts from JSON file
    expected_counts = load_expected_counts(args.expected_counts_file)

    # Initialize extractor with specified parameters
    extractor = RobertaFeatureExtractor(
        context_window=args.context_window,
        middle_layer=args.middle_layer,
        max_tokens=args.max_tokens,
        kept_tokens=args.kept_tokens
    )

    # Create output directory with context window info
    save_dir = f"{args.output_dir}_ctx{args.context_window}_l{args.middle_layer}"
    os.makedirs(save_dir, exist_ok=True)

    process_folder(args.folder_path, extractor, save_dir, expected_counts)
