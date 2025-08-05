import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from load_files import root_data_dir

from imports import import_torch, import_moviepy, import_torchvision_and_pytorchvideo
torch = import_torch()
moviepy, VideoFileClip = import_moviepy()
Compose, Lambda, CenterCrop, create_feature_extractor, Normalize, UniformTemporalSubsample, ShortSideScale = import_torchvision_and_pytorchvideo()

def define_frames_transform():
    transform = Compose([
        UniformTemporalSubsample(8),
        Lambda(lambda x: x / 255.0),
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=256),
        CenterCrop(256)
    ])
    return transform

transform = define_frames_transform()

def get_vision_model(device):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor, model_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
feature_extractor, model_layer = get_vision_model(device)


def extract_visual_features(episode_path, tr, feature_extractor, model_layer,
                            transform, device, save_dir_temp):
    import tempfile
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    temp_dir = tempfile.mkdtemp(dir=save_dir_temp)
    os.makedirs(temp_dir, exist_ok=True)
    visual_features = []

    for start in tqdm(start_times, desc=f"Extracting features from {os.path.basename(episode_path)}"):
        clip_chunk = clip.subclip(start, start + tr)
        chunk_path = os.path.join(temp_dir, f'visual_chunk_{int(start * 100):06d}.mp4')
        clip_chunk.write_videofile(chunk_path, verbose=False, audio=False, logger=None)
        if not os.path.exists(chunk_path):
            raise RuntimeError(f"Chunk file {chunk_path} not found.")

        try:
            video_clip = VideoFileClip(chunk_path)
            chunk_frames = [frame for frame in video_clip.iter_frames()]
        except OSError as e:
            print(f"Skipping corrupted chunk: {chunk_path} - {e}")
            with open("failed_chunks.log", "a") as log:
                log.write(f"{chunk_path}\n")
            continue

        frames_array = np.transpose(np.array(chunk_frames), (3, 0, 1, 2))
        inputs = torch.from_numpy(frames_array).float()
        inputs = transform(inputs).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = feature_extractor(inputs)

        visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))

        # Save partial result
        partial_name = Path(episode_path).stem + "_features_visual_partial.npy"
        np.save(os.path.join(save_dir_temp, partial_name), np.array(visual_features))

    visual_features = np.array(visual_features, dtype='float32')
    return visual_features


def process_folder(folder_path, tr, feature_extractor, model_layer, transform, device, save_dir_temp, save_dir_features):
    folder_path = Path(folder_path)
    # Preserve nested directory structure relative to stimuli/movies
    stimuli_root = Path(root_data_dir) / "stimuli" / "movies"
    relative_path = folder_path.relative_to(stimuli_root)
    output_folder = Path(save_dir_features) / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)


    mkv_files = sorted(folder_path.glob("*.mkv"))
    print(f"Found {len(mkv_files)} .mkv files in {folder_path}")

    for mkv_file in mkv_files:
        save_path = output_folder / f"{mkv_file.stem}_features_visual.npy"
        if save_path.exists():
            print(f"Features already exist for {mkv_file.name}, skipping.")
            continue
        print(f"Extracting visual features from {mkv_file.name}")
        visual_features = extract_visual_features(str(mkv_file), tr, feature_extractor,
                                                  model_layer, transform, device, save_dir_temp)
        np.save(save_path, visual_features)
        print(f"Saved features to {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python visual_feature_extraction_batch.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1]

    tr = 1.49
    save_dir_temp = "../results/visual_features/temp"
    save_dir_features = "../results/visual_features"
    os.makedirs(save_dir_features, exist_ok=True)

    process_folder(folder_path, tr, feature_extractor, model_layer, transform, device, save_dir_temp, save_dir_features)
