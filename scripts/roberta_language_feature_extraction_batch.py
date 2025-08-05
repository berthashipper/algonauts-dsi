import os
from pathlib import Path
import numpy as np
import pandas as pd
import string
import h5py
from tqdm import tqdm
from imports import import_torch, import_transformers
root_data_dir = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data"

torch = import_torch()
from transformers import RobertaTokenizer, RobertaModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = RobertaModel.from_pretrained("roberta-base").to(device).eval()
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

middle_layer_index = 7  # For roberta-base (12 layers), try 4–8 for best results
hidden_size = model.config.hidden_size


def pad_transcript_df(df, expected_rows):
    current_rows = df.shape[0]
    if current_rows < expected_rows:
        print(f"Padding from {current_rows} to {expected_rows} rows")
        n_pad = expected_rows - current_rows
        # Create empty rows with same columns, empty strings
        pad_df = pd.DataFrame({col: ['']*n_pad for col in df.columns})
        df = pd.concat([df, pad_df], ignore_index=True)
    elif current_rows > expected_rows:
        print(f"Warning: transcript longer ({current_rows}) than expected ({expected_rows}), truncating")
        df = df.iloc[:expected_rows].copy()
    else:
        print(f"No padding needed: {current_rows} rows")
    return df


def extract_language_features(tsv_path, model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device, expected_rows):
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
    df = pad_transcript_df(df, expected_rows)

    # Mark missing or empty 'text_per_tr' explicitly
    df["is_na"] = df["text_per_tr"].apply(lambda x: (x is None) or (str(x).strip() == ""))
    print(f"Number of rows with empty/missing text_per_tr: {df['is_na'].sum()}")

    if df['is_na'].sum() > 0:
        print("Rows with empty or missing 'text_per_tr':")
        print(df.loc[df['is_na'], ['text_per_tr']])

    pooler_output, last_hidden_state = [], []

    for i in tqdm(range(df.shape[0]), desc=f"Extracting {Path(tsv_path).stem}"):
        if not df.iloc[i]["is_na"]:
            tr_text = df.iloc[i]["text_per_tr"]
            try:
                inputs = tokenizer(tr_text, return_tensors="pt", truncation=True, max_length=num_used_tokens+2)
                input_tensor = inputs['input_ids'].to(device)
                with torch.no_grad():
                    outputs = model(input_tensor, output_hidden_states=True)
                    pooled = outputs.hidden_states[middle_layer_index][0, 0].cpu().numpy()
            except Exception as e:
                print(f"[ERROR] Pooler output extraction failed at row {i}: {e}")
                pooled = np.full(hidden_size, np.nan, dtype='float32')
        else:
            pooled = np.full(hidden_size, np.nan, dtype='float32')

        pooler_output.append(pooled)

        if not df.iloc[i]["is_na"]:
            try:
                np_text = df.iloc[i]["text_per_tr"].translate(str.maketrans('', '', string.punctuation))
                np_inputs = tokenizer(np_text, return_tensors="pt", truncation=True, max_length=num_used_tokens+2)
                np_tensor = np_inputs['input_ids'].to(device)
                with torch.no_grad():
                    outputs = model(np_tensor, output_hidden_states=True)
                    np_out = outputs.hidden_states[middle_layer_index][0][1:-1].cpu().numpy()
                feat = np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32')
                tk_idx = min(kept_tokens_last_hidden_state, np_out.shape[0])
                feat[-tk_idx:] = np_out[-tk_idx:]
            except Exception as e:
                print(f"[ERROR] Last hidden state extraction failed at row {i}: {e}")
                feat = np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32')
        else:
            feat = np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32')

        last_hidden_state.append(feat)

    print(f"Finished extraction: pooler_output length = {len(pooler_output)}, last_hidden_state length = {len(last_hidden_state)}")

    assert len(pooler_output) == df.shape[0], f"Length mismatch: pooler_output={len(pooler_output)}, TSV rows={df.shape[0]}"
    assert len(last_hidden_state) == df.shape[0], f"Length mismatch: last_hidden_state={len(last_hidden_state)}, TSV rows={df.shape[0]}"

    return np.array(pooler_output, dtype='float32'), np.array(last_hidden_state, dtype='float32')


def process_folder(folder_path, model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device, save_dir_features):
    # Convert to absolute paths
    folder_path = Path(folder_path).resolve()
    movies_root = Path("/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies").resolve()
    transcripts_root = Path("/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts").resolve()
    
    print(f"Processing folder: {folder_path}")
    print(f"Movies root: {movies_root}")
    print(f"Transcripts root: {transcripts_root}")

    # Get relative path from movies root (e.g. "ood/chaplin")
    try:
        relative_path = folder_path.relative_to(movies_root)
    except ValueError as e:
        print(f"Error: {folder_path} is not under {movies_root}")
        return

    # Find corresponding transcript folder
    transcript_folder = transcripts_root / relative_path
    if not transcript_folder.exists():
        print(f"Transcript folder not found: {transcript_folder}")
        return

    # Create output folder with same structure
    output_folder = Path(save_dir_features) / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Will save features to: {output_folder}")

    # Process TSV files
    tsv_files = sorted(transcript_folder.glob("*.tsv"))
    print(f"Found {len(tsv_files)} .tsv files in {transcript_folder}")

    # Expected row counts per TSV (from submission format)
    expected_counts = {
        "ood_chaplin1.tsv": 432,
        "ood_chaplin2.tsv": 405,
        "ood_mononoke1.tsv": 423,
        "ood_mononoke2.tsv": 426,
        "ood_passepartout1.tsv": 422,
        "ood_passepartout2.tsv": 436,
        "ood_planetearth1.tsv": 433,
        "ood_planetearth2.tsv": 418,
        "ood_pulpfiction1.tsv": 468,
        "ood_pulpfiction2.tsv": 378,
        "ood_wot1.tsv": 353,
        "ood_wot2.tsv": 324,
    }

    for tsv_file in tsv_files:
        save_path = output_folder / f"{tsv_file.stem}_features_language.h5"
        if save_path.exists():
            print(f"Skipping {tsv_file.name}, features already exist.")
            continue

        expected_rows = expected_counts.get(tsv_file.name)
        if expected_rows is None:
            print(f"Warning: No expected row count for {tsv_file.name}, skipping")
            continue

        pooler_output, last_hidden_state = extract_language_features(
            str(tsv_file), model, tokenizer, num_used_tokens,
            kept_tokens_last_hidden_state, device, expected_rows
        )

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('pooler_output', data=pooler_output, dtype=np.float32)
            f.create_dataset('last_hidden_state', data=last_hidden_state, dtype=np.float32)
        print(f"Saved features to {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python language_feature_extraction_batch.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    save_dir_features = "../results/roberta_language_features_l7"
    os.makedirs(save_dir_features, exist_ok=True)

    process_folder(folder_path, model, tokenizer, 510, 10, device, save_dir_features)


