import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from load_files import root_data_dir
from imports import import_torch, import_librosa, import_moviepy

torch = import_torch()
librosa = import_librosa()
moviepy, VideoFileClip = import_moviepy()

def extract_audio_features(episode_path, tr, sr, device, save_dir_temp):
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    audio_features = []

    os.makedirs(save_dir_temp, exist_ok=True)

    for start in tqdm(start_times, desc=f"Extracting audio: {os.path.basename(episode_path)}"):
        clip_chunk = clip.subclip(start, start + tr)
        chunk_path = os.path.join(save_dir_temp, f'audio_chunk_{int(start * 100):06d}.wav')

        try:
            clip_chunk.audio.write_audiofile(chunk_path, verbose=False, logger=None)
            y, sr_ = librosa.load(chunk_path, sr=sr, mono=True)
            mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr_), axis=1)
            audio_features.append(mfcc_features)
        except Exception as e:
            print(f"Failed on chunk {chunk_path}: {e}")
            continue

    return np.array(audio_features, dtype='float32')


def process_folder(folder_path, tr, sr, device, save_dir_temp, save_dir_features):
    folder_path = Path(folder_path)
    stimuli_root = Path(root_data_dir) / "stimuli" / "movies"
    relative_path = folder_path.relative_to(stimuli_root)
    output_folder = Path(save_dir_features) / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)

    mkv_files = sorted(folder_path.glob("*.mkv"))
    print(f"Found {len(mkv_files)} .mkv files in {folder_path}")

    for mkv_file in mkv_files:
        save_path = output_folder / f"{mkv_file.stem}_features_audio.npy"
        if save_path.exists():
            print(f"Audio features already exist for {mkv_file.name}, skipping.")
            continue

        print(f"Extracting audio features from {mkv_file.name}")
        features = extract_audio_features(str(mkv_file), tr, sr, device, save_dir_temp)
        np.save(save_path, features)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python audio_feature_extraction_batch.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    tr = 1.49
    sr = 22050
    save_dir_temp = "../results/audio_features/temp"
    save_dir_features = "../results/audio_features"

    os.makedirs(save_dir_features, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    process_folder(folder_path, tr, sr, device, save_dir_temp, save_dir_features)
