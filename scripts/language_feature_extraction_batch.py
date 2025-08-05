import os
from pathlib import Path
import numpy as np
import pandas as pd
import string
import h5py
from tqdm import tqdm

from load_files import root_data_dir
from imports import import_torch, import_transformers

torch = import_torch()
BertTokenizer, BertModel = import_transformers()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, tokenizer = BertModel.from_pretrained("bert-base-uncased").to(device).eval(), \
                   BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


middle_layer_index = 6  # For BERT-base (12 layers), try 4–8; adjust as needed
hidden_size = model.config.hidden_size


def extract_language_features(tsv_path, model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device):
    df = pd.read_csv(tsv_path, sep='\t')
    df.insert(0, "is_na", df["text_per_tr"].isna())

    tokens, np_tokens = [], []
    pooler_output, last_hidden_state = [], []

    for i in tqdm(range(df.shape[0]), desc=f"Extracting {Path(tsv_path).stem}"):
        if not df.iloc[i]["is_na"]:
            tr_text = df.iloc[i]["text_per_tr"]
            tokens.extend(tokenizer.tokenize(tr_text))
            np_tokens.extend(tokenizer.tokenize(tr_text.translate(str.maketrans('', '', string.punctuation))))

        if len(tokens) > 0:
            ids = [101] + tokenizer.convert_tokens_to_ids(tokens[-num_used_tokens:]) + [102]
            input_tensor = torch.tensor(ids).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor, output_hidden_states=True)
                # Use the [CLS] (first) token from the middle layer
                pooler_output.append(outputs.hidden_states[middle_layer_index][0, 0].cpu().numpy())
        else:
            pooler_output.append(np.full(hidden_size, np.nan, dtype='float32'))

        if len(np_tokens) > 0:
            np_ids = [101] + tokenizer.convert_tokens_to_ids(np_tokens[-num_used_tokens:]) + [102]
            np_tensor = torch.tensor(np_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(np_tensor, output_hidden_states=True)
                # Use all tokens except special tokens from the middle layer
                np_out = outputs.hidden_states[middle_layer_index][0][1:-1].cpu().numpy()
            feat = np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32')
            tk_idx = min(kept_tokens_last_hidden_state, len(np_tokens))
            feat[-tk_idx:] = np_out[-tk_idx:]
            last_hidden_state.append(feat)
        else:
            last_hidden_state.append(np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32'))

    return np.array(pooler_output, dtype='float32'), np.array(last_hidden_state, dtype='float32')

def process_folder(folder_path, model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device, save_dir_features):
    folder_path = Path(folder_path)
    stimuli_root = Path(root_data_dir) / "stimuli" / "transcripts"
    relative_path = folder_path.relative_to(stimuli_root)
    output_folder = Path(save_dir_features) / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(folder_path.glob("*.tsv"))
    print(f"Found {len(tsv_files)} .tsv files in {folder_path}")

    for tsv_file in tsv_files:
        save_path = output_folder / f"{tsv_file.stem}_features_language.h5"
        if save_path.exists():
            print(f"Skipping {tsv_file.name}, features already exist.")
            continue

        pooler_output, last_hidden_state = extract_language_features(str(tsv_file),
            model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device)

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
    save_dir_features = "../results/language_features_l6"
    os.makedirs(save_dir_features, exist_ok=True)

    process_folder(folder_path, model, tokenizer, 510, 10, device, save_dir_features)

