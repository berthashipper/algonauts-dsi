import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from load_files import root_data_dir
import torch
import librosa
from moviepy.editor import VideoFileClip
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Load HuBERT model and feature extractor once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
hubert_model.eval()

def extract_audio_features(episode_path, tr, sr, device, save_dir_temp):
    # Load video clip
    clip = VideoFileClip(episode_path)

    # Generate all start times (INCLUDES last chunk)
    start_times = np.arange(0, clip.duration, tr)

    audio_features = []
    os.makedirs(save_dir_temp, exist_ok=True)
    n_success = 0 # Track successful chunks

    for start in tqdm(start_times, desc=f"Extracting: {os.path.basename(episode_path)}"):
        # Handle partial last chunk
        end = min(start + tr, clip.duration)

        # Unique filename with process ID to avoid collisions
        chunk_path = os.path.join(
            save_dir_temp,
            f'audio_chunk_{int(start*100)}_{os.getpid()}.wav'
        )

        try:
            # Extract audio segment
            clip.subclip(start, end).audio.write_audiofile(
                chunk_path, verbose=False, logger=None
            )

            # Load audio with correct sampling rate
            y, _ = librosa.load(chunk_path, sr=sr, mono=True)

            # Verify audio length (critical fix!)
            expected_samples = int(sr * tr)
            if len(y) < expected_samples:
                # Pad with zeros if too short (for last chunk)
                y = np.pad(y, (0, expected_samples - len(y)), mode='constant')

            # Extract features with HuBERT (MODIFIED SECTION)
            inputs = feature_extractor(
                y,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = hubert_model(input_values, output_hidden_states=True)
                # Extract layer __  (4th hidden state)
                layernum = outputs.hidden_states[19]  # [batch, time, features]
                features = layernum.mean(dim=1).cpu().numpy()

            audio_features.append(features[0]) # Remove batch dim
            n_success += 1

        except Exception as e:
            print(f"⚠️ Chunk {start:.2f}s failed: {str(e)}")
        finally:
            # Cleanup: remove temp file whether successful or not
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    print(f"✅ {n_success}/{len(start_times)} chunks succeeded")
    return np.array(audio_features, dtype='float32') if audio_features else np.array([])

def process_folder(folder_path, tr, sr, device, save_dir_temp, save_dir_features):
    folder_path = Path(folder_path)
    stimuli_root = Path(root_data_dir) / "stimuli" / "movies"
    relative_path = folder_path.relative_to(stimuli_root)
    output_folder = Path(save_dir_features) / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)

    mkv_files = sorted(folder_path.glob("*.mkv"))
    print(f"Found {len(mkv_files)} .mkv files in {folder_path}")

    for mkv_file in mkv_files:
        # Create unique temp directory per episode
        episode_temp_dir = os.path.join(save_dir_temp, f"temp_{mkv_file.stem}_{os.getpid()}")
        os.makedirs(episode_temp_dir, exist_ok=True)

        save_path = output_folder / f"{mkv_file.stem}_features_audio.npy"
        if save_path.exists():
            print(f"Audio features already exist for {mkv_file.name}, skipping.")
            continue

        print(f"Extracting audio features from {mkv_file.name}")
        features = extract_audio_features(str(mkv_file), tr, sr, device, episode_temp_dir)

        # Only save if features are non-empty
        if features.size > 0:
            np.save(save_path, features)
            print(f"Saved to {save_path}")
        else:
            print(f"❌ CRITICAL: Empty features for {mkv_file.name}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python hubert_audio_feature_extraction_batch.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    tr = 1.49 # time resolution chunk length in seconds
    sr = 16000 # sampling rate
    save_dir_temp = "../results/hubert_audio_features_l19/temp"
    save_dir_features = "../results/hubert_audio_features/layer19"

    os.makedirs(save_dir_features, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    process_folder(folder_path, tr, sr, device, save_dir_temp, save_dir_features)
