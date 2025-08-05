import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
from torchvision import transforms
from moviepy.editor import VideoFileClip
import tempfile
from load_files import root_data_dir

# This extracts multi-layer visual features from video files using a pretrained AlexNet model.
# The extracted features are saved as numpy arrays, with one feature vector per ~1.49 second chunk of video.

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TR = 1.49  # Temporal resolution in seconds

SAVE_DIR_TEMP = "../results/alexnet_visual_features/temp"
SAVE_DIR_FEATURES = "../results/alexnet_visual_features"
ALEXNET_LAYERS = [
    'features.3',    # Conv2: middle of early visual layers
    'features.6',    # Conv3: detects more complex patterns
    'features.8',    # Conv4: more abstract visual features
    'features.10',   # Conv5: high-level, dense spatial features
    'classifier.1'   # FC6: fully-connected, object-level semantics
]

def get_alexnet_model():
    """Load pretrained AlexNet and prepare for feature extraction"""
    model = models.alexnet(pretrained=True).to(DEVICE)
    model.eval()

    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    for layer_name in ALEXNET_LAYERS:
        module = model
        for attr in layer_name.split('.'):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        module.register_forward_hook(get_features(layer_name))

    return model, features

def preprocess_frames(frames):
    """Preprocess frames for AlexNet input"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    frames = [transform(frame) for frame in frames]
    return torch.stack(frames).to(DEVICE)

def extract_alexnet_features(video_path, model, features_dict):
    """Extract multi-level features from AlexNet"""
    clip = VideoFileClip(video_path)
    start_times = np.arange(0, clip.duration, TR)  # Include last partial chunk

    visual_features = []

    print(f"[INFO] Processing {os.path.basename(video_path)}")
    print(f"[INFO] Duration: {clip.duration:.2f}s | Chunks: {len(start_times)}")

    for start in tqdm(start_times, desc=f"Extracting features from {os.path.basename(video_path)}"):
        end = min(start + TR, clip.duration)
        try:
            chunk = clip.subclip(start, end)
            frames = [frame for frame in chunk.iter_frames()]
            if not frames:
                print(f"[WARNING] No frames in chunk {start:.2f}-{end:.2f}, padding with zeros.")
                # Pad with zeros or previous chunk's features to maintain consistent length
                if visual_features:
                    visual_features.append(visual_features[-1])
                else:
                    # Assume combined feature vector size by running a dummy tensor through the model once
                    dummy_input = torch.zeros((1, 3, 224, 224), device=DEVICE)
                    with torch.no_grad():
                        _ = model(dummy_input)
                    dummy_feats = []
                    for layer in ALEXNET_LAYERS:
                        feat = features_dict.get(layer)
                        if feat is None:
                            continue
                        if len(feat.shape) == 4:
                            feat = torch.mean(feat, dim=[2, 3])
                        feat_mean = feat.mean(dim=0)
                        dummy_feats.append(feat_mean.cpu().numpy().flatten())
                    combined_size = sum([f.size for f in dummy_feats])
                    visual_features.append(np.zeros(combined_size, dtype=np.float32))
                continue

            inputs = preprocess_frames(frames)
            with torch.no_grad():
                _ = model(inputs)

            layer_features = []
            for layer in ALEXNET_LAYERS:
                feat = features_dict.get(layer)
                if feat is None:
                    print(f"[ERROR] Missing layer: {layer}")
                    continue

                if len(feat.shape) == 4:
                    feat = torch.mean(feat, dim=[2, 3])
                feat_mean = feat.mean(dim=0)  # Temporal mean over frames
                flat_feat = feat_mean.cpu().numpy().flatten()
                layer_features.append(flat_feat)

            if layer_features:
                combined = np.concatenate(layer_features)
                visual_features.append(combined)
            else:
                # If no features extracted, pad with last or zeros
                if visual_features:
                    visual_features.append(visual_features[-1])
                else:
                    combined_size = 0
                    for layer in ALEXNET_LAYERS:
                        dummy_feat = np.zeros(1)  # fallback if really missing
                        combined_size += dummy_feat.size
                    visual_features.append(np.zeros(combined_size, dtype=np.float32))

        except Exception as e:
            print(f"[ERROR] Chunk {start:.2f}-{end:.2f}: {str(e)}")
            if visual_features:
                visual_features.append(visual_features[-1])
            else:
                combined_size = 0
                for layer in ALEXNET_LAYERS:
                    dummy_feat = np.zeros(1)
                    combined_size += dummy_feat.size
                visual_features.append(np.zeros(combined_size, dtype=np.float32))

    final_array = np.array(visual_features, dtype='float32')
    print(f"[INFO] Extracted {len(final_array)} chunks | Final shape: {final_array.shape}")
    return final_array

def process_video_file(video_path, model, features_dict, save_dir_features):
    video_path = Path(video_path)
    stimuli_root = Path(root_data_dir) / "stimuli" / "movies"
    relative_path = video_path.parent.relative_to(stimuli_root)
    output_folder = Path(save_dir_features) / relative_path
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{video_path.stem}_features_alexnet.npy"
    if output_path.exists():
        print(f"[INFO] Features already exist for {video_path.name}, skipping.")
        return

    print(f"[INFO] Processing {video_path.name}")
    features = extract_alexnet_features(str(video_path), model, features_dict)
    np.save(output_path, features)
    print(f"[INFO] Saved AlexNet features to {output_path}")

def process_folder(folder_path, model, features_dict):
    """Process all video files in a folder"""
    folder_path = Path(folder_path)
    video_files = sorted(folder_path.glob("*.mkv")) + sorted(folder_path.glob("*.mp4"))

    print(f"Found {len(video_files)} video files in {folder_path}")
    os.makedirs(SAVE_DIR_FEATURES, exist_ok=True)
    os.makedirs(SAVE_DIR_TEMP, exist_ok=True)

    for video_file in video_files:
        process_video_file(video_file, model, features_dict, SAVE_DIR_FEATURES)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python alex_net_visual_extraction.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    print("Loading AlexNet model...")
    alexnet, features_dict = get_alexnet_model()
    print(f"Using device: {DEVICE}")

    print(f"Processing folder: {folder_path}")
    process_folder(folder_path, alexnet, features_dict)
