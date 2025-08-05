import argparse
import os
import numpy as np
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
        for npy_file in input_dir.rglob("*.npy"):
            relative_path = npy_file.relative_to(raw_root)
            features = np.load(npy_file)

            print(f"[DEBUG] {npy_file.name}: shape = {features.shape}")

            all_features.append(features)
            chunks_per_episode.append(len(features))
            episode_paths.append(relative_path)
            episode_names.append(npy_file.stem)

    all_shapes = [f.shape[1] for f in all_features]
    if len(set(all_shapes)) != 1:
        raise ValueError(f"Inconsistent feature dimensions: {set(all_shapes)}")

    all_features = np.concatenate(all_features, axis=0)
    return all_features, episode_paths, chunks_per_episode, episode_names

def save_individual_pca_features(features, episode_paths, chunks, save_root):
    idx = 0
    for path, chunk_len in zip(episode_paths, chunks):
        out_path = Path(save_root) / path
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, features[idx:idx+chunk_len])
        idx += chunk_len

def main(args):
    raw_root = Path(args.raw_root)
    save_root = Path(args.save_root)
    input_dirs = [Path(d) for d in args.input_dirs]

    # Identify input files and skip already-reduced ones
    valid_input_dirs = []
    all_npy_files = []
    for d in input_dirs:
        for npy_file in d.rglob("*.npy"):
            relative = npy_file.relative_to(raw_root)
            out_path = save_root / relative
            if out_path.exists():
                print(f"Skipping already reduced: {relative}")
            else:
                all_npy_files.append(npy_file)
        if all_npy_files:
            valid_input_dirs.append(d)

    if not all_npy_files:
        print("Nothing to process.")
        return

    print(f"Loading features from {len(all_npy_files)} files...")
    features, episode_paths, chunks, names = collect_features_and_paths(valid_input_dirs, raw_root)

    print("Standardizing...")
    features = np.nan_to_num(features)
    scaler = StandardScaler().fit(features)
    features_std = scaler.transform(features)

    # Save scaler parameters
    save_root.mkdir(parents=True, exist_ok=True)
    np.save(save_root / 'scaler_param.npy', {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_,
        'var_': scaler.var_,
    })

    print("Running PCA...")
    n_components = args.n_components
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
                        help='List of directories with raw .npy visual features')
    parser.add_argument('--raw_root', required=True,
                        help='Root directory of original .npy features (e.g., ../results/visual_features)')
    parser.add_argument('--save_root', required=True,
                        help='Root directory for saving PCA outputs (e.g., ../results/pca/visual)')
    parser.add_argument('--full_components', action='store_true',
                        help='Use full number of PCA components instead of 250')
    parser.add_argument('--n_components', type=int, default=None,
                    help='Number of PCA components to keep (default: 250)')
    args = parser.parse_args()
    main(args)
