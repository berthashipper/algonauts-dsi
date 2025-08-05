import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def collect_features_and_paths(input_dirs, raw_root):
    episode_paths = []
    all_features = []
    chunks_per_episode = []
    episode_names = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        for h5_file in input_dir.rglob("*.h5"):
            relative_path = h5_file.relative_to(raw_root).with_suffix(".npy")  # Save as .npy
            with h5py.File(h5_file, "r") as f:
                features = f["pooler_output"][:]  # shape: (num_chunks, feature_dim)
            all_features.append(features)
            chunks_per_episode.append(len(features))
            episode_paths.append(relative_path)
            episode_names.append(h5_file.stem)

    all_features = np.concatenate(all_features, axis=0)
    return all_features, episode_paths, chunks_per_episode, episode_names

def save_individual_pca_features(features, episode_paths, chunks, save_root):
    idx = 0
    for path, chunk_len in zip(episode_paths, chunks):
        out_path = Path(save_root) / path.with_suffix('.npy')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, features[idx:idx+chunk_len])
        idx += chunk_len

def main(args):
    raw_root = Path(args.raw_root)
    save_root = Path(args.save_root)
    input_dirs = [Path(d) for d in args.input_dirs]

    # Identify input files and skip already reduced ones
    valid_input_dirs = []
    all_h5_files = []
    for d in input_dirs:
        for h5_file in d.rglob("*.h5"):
            relative = h5_file.relative_to(raw_root)
            out_path = save_root / relative.with_suffix(".npy")
            if out_path.exists():
                print(f"Skipping already reduced: {relative}")
            else:
                all_h5_files.append(h5_file)
        if all_h5_files:
            valid_input_dirs.append(d)

    if not all_h5_files:
        print("Nothing to process.")
        return

    print(f"Loading features from {len(all_h5_files)} files...")
    features, episode_paths, chunks, _ = collect_features_and_paths(valid_input_dirs, raw_root)

    print("Standardizing...")
    features = np.nan_to_num(features)
    scaler = StandardScaler().fit(features)
    features_std = scaler.transform(features)

    save_root.mkdir(parents=True, exist_ok=True)
    np.save(save_root / 'scaler_param.npy', {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_,
        'var_': scaler.var_,
    })

    print("Running PCA...")
    n_components = args.n_components
    if n_components is None:
        n_components = min(250, features_std.shape[0], features_std.shape[1])

    pca = PCA(n_components=n_components, random_state=20200220).fit(features_std)

    np.save(save_root / 'pca_param.npy', {
        'components_': pca.components_,
        'explained_variance_': pca.explained_variance_,
        'explained_variance_ratio_': pca.explained_variance_ratio_,
        'mean_': pca.mean_,
        'singular_values_': pca.singular_values_,
    })

    features_pca = pca.transform(features_std).astype(np.float32)
    print(f"Saving PCA-reduced features to {save_root}")
    save_individual_pca_features(features_pca, episode_paths, chunks, save_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='List of directories with raw .h5 language features')
    parser.add_argument('--raw_root', required=True,
                        help='Root directory of original .h5 features')
    parser.add_argument('--save_root', required=True,
                        help='Root directory for saving PCA outputs')
    parser.add_argument('--full_components', action='store_true',
                        help='Use full number of PCA components instead of 250')
    parser.add_argument('--n_components', type=int, default=None,
                    help='Number of PCA components to keep')

    args = parser.parse_args()
    main(args)
