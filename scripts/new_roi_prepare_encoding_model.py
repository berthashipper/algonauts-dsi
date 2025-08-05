import os
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
import pandas as pd

def load_pca_features(modality_path):
    """Load PCA features from specified directory, skipping param files."""
    features = {}
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"PCA modality directory not found: {modality_path}")
    for root, dirs, files in os.walk(modality_path):
        for fname in files:
            if not fname.endswith(".npy"):
                continue
            if fname in ["pca_param.npy", "scaler_param.npy"]:
                continue  # Skip PCA/scaler metadata files
            full_path = os.path.join(root, fname)
            key = os.path.relpath(full_path, modality_path)
            features[key] = np.load(full_path, allow_pickle=True)
    if not features:
        raise ValueError(f"No valid .npy feature files found in {modality_path}")
    print(f"Loaded {len(features)} feature files from {modality_path}")
    return features

def load_fmri(root_data_dir, subject):
    """Load fMRI responses for the given subject."""
    fmri = {}
    # Load Friends fMRI
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)
    # Load Movie10 fMRI
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)
    # Average repeats
    for movie_prefix in ['figures', 'life']:
        keys_all = list(fmri.keys())
        for s in range(12 if movie_prefix == 'figures' else 5):
            movie = f'{movie_prefix}{str(s+1).zfill(2)}'
            reps = [k for k in keys_all if movie in k]
            if len(reps) == 2:
                fmri[movie] = ((fmri[reps[0]] + fmri[reps[1]]) / 2).astype(np.float32)
                del fmri[reps[0]]
                del fmri[reps[1]]
    return fmri

def align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies):
    """Align features with fMRI samples."""
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)
    expected_length = None
    for movie in movies:
        if movie.startswith('friends'):
            season_num = movie.split('-')[1][1:]
            season_prefix = f"s{season_num}e"
            movie_splits = [key for key in fmri if key.startswith(season_prefix)]
        elif movie.startswith('movie10'):
            movie_folder = movie.split('-')[1]
            movie_splits = [key for key in fmri if key.startswith(movie_folder)]
        else:
            movie_splits = [key for key in fmri if key.startswith(movie)]
        if not movie_splits:
            print(f"Warning: no fmri keys found for movie '{movie}'")
            continue
        for split in movie_splits:
            fmri_split = fmri[split][excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, axis=0)
            for s in range(len(fmri_split)):
                f_all = []
                for mod in features.keys():
                    matching_keys = [k for k in features[mod].keys() if split in k]
                    if not matching_keys:
                        raise KeyError(f"Could not match split '{split}' to any key in {mod} features")
                    full_key = matching_keys[0]
                    if mod in ['visual', 'audio']:
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        feat_len = len(features[mod][full_key])
                        if idx_end > feat_len:
                            idx_end = feat_len
                            idx_start = idx_end - stimulus_window
                        f = features[mod][full_key][idx_start:idx_end].flatten()
                        f_all.append(f)
                    elif mod == 'language':
                        idx = excluded_samples_start if s < hrf_delay else s + excluded_samples_start - hrf_delay
                        lang_feat_len = len(features[mod][full_key])
                        f = features[mod][full_key][-1] if idx >= lang_feat_len - hrf_delay else features[mod][full_key][idx]
                        f_all.append(f.flatten())
                if f_all:
                    feature_vector = np.concatenate(f_all)
                    if expected_length is None:
                        expected_length = len(feature_vector)
                    if len(feature_vector) != expected_length:
                        print(f"\n[‼️ FEATURE LENGTH MISMATCH]")
                        print(f"  Split: {split} | Timepoint: {s}")
                        print(f"  Expected: {expected_length} | Got: {len(feature_vector)}")
                        print(f"  Modality component shapes: {[f.shape for f in f_all]}")
                        raise ValueError("Feature vector length mismatch detected.")
                    aligned_features.append(feature_vector)
                else:
                    aligned_features.append(np.array([]))
    return np.asarray(aligned_features, dtype=np.float32), aligned_fmri

def train_encoding(features_train, fmri_train):
    """Train ridge regression encoding model."""
    alphas = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5])
    model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    model.fit(features_train, fmri_train)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_root", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--train_movies", type=str, required=True)
    parser.add_argument("--val_movies", type=str, required=True)
    parser.add_argument("--modality_paths", type=str, required=True,
                        help="Comma-separated list; e.g., 'visual:/v,audio:/a,language:/l1,language:/l3,language:/l5,language:/l7'")
    parser.add_argument("--version_name", type=str, required=True)
    parser.add_argument("--output_model_dir", type=str, default="models")
    parser.add_argument("--group_plot", action="store_true")
    parser.add_argument("--context_windows", type=int, nargs='+', required=True)

    args = parser.parse_args()

    # Parse modality paths
    parsed_paths = []
    for item in args.modality_paths.split(","):
        item = item.strip()
        split_idx = item.find(":")
        if split_idx <= 0:
            raise ValueError(f"Malformed modality_paths item: {item}")
        mod = item[:split_idx]
        path = item[split_idx + 1 :]
        parsed_paths.append((mod, path))

    # Collect unique visual/audio paths and mapping of language contexts
    visual_path = None
    audio_path = None
    language_paths = dict()
    for mod, pth in parsed_paths:
        if mod == "visual":
            visual_path = pth
        elif mod == "audio":
            audio_path = pth
        elif mod == "language":
            if "_ctx" in pth:
                win = int(pth.split("_ctx")[-1])
                language_paths[win] = pth
            else:
                raise ValueError(f"Each language path must include _ctxN: {pth}")

    # Debug prints for paths
    print(f"Parsed visual_path: {visual_path}")
    print(f"Parsed audio_path: {audio_path}")
    print(f"Parsed language_paths: {language_paths}")

    # Sanity check
    context_windows = args.context_windows
    for win in context_windows:
        if win not in language_paths:
            raise ValueError(f"Missing a language path for context window {win}. Language paths: {language_paths}")

    movies_train = args.train_movies.split(",")
    movies_val = args.val_movies.split(",")

    if args.group_plot:
        subjects = [int(s) for s in args.subject.split(",")]
        context_window_to_acc = {}
        for win in context_windows:
            print(f"\n=== Processing context window {win} ===\n")
            accuracy_dict = {}
            for subject in subjects:
                print(f"Processing subject {subject} for window={win}")
                features = {}
                if visual_path is not None:
                    features["visual"] = load_pca_features(visual_path)
                if audio_path is not None:
                    features["audio"] = load_pca_features(audio_path)
                features["language"] = load_pca_features(language_paths[win])

                fmri = load_fmri(args.fmri_root, subject)
                features_val, fmri_val = align_features_and_fmri_samples(
                    features, fmri, 5, 5, 2, win, movies_val
                )
                if features_val.shape[0] == 0:
                    print(f"No validation features for subject {subject}, skipping.")
                    continue
                features_train, fmri_train = align_features_and_fmri_samples(
                    features, fmri, 5, 5, 2, win, movies_train
                )
                model = train_encoding(features_train, fmri_train)
                fmri_val_pred = model.predict(features_val)
                encoding_accuracy = np.array([
                    pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
                    for p in range(fmri_val.shape[1])
                ], dtype=np.float32)
                accuracy_dict[subject] = encoding_accuracy
            if len(accuracy_dict) > 0:
                all_acc = np.array(list(accuracy_dict.values()))
                mean_acc = np.mean(all_acc, axis=0)
                print(f"Window {win} mean accuracy shape: {mean_acc.shape}")
                context_window_to_acc[win] = mean_acc
            else:
                print(f"WARNING: No valid ROI accuracies for window {win}")

        df = pd.DataFrame(context_window_to_acc)
        print("\n--- Accuracy DataFrame head ---")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nNumber of NaNs per column:")
        print(df.isna().sum())
        print("\nColumns:")
        print(df.columns)
        df.columns = df.columns.astype(str)
        best_windows = df.idxmax(axis=1)
        max_accuracies = df.max(axis=1)
        print("\nBest Context Window value counts:")
        print(best_windows.value_counts())
        summary = pd.DataFrame({
            'Best Context Window': best_windows,
            'Max Accuracy': max_accuracies
        })
        df.to_csv('roi_accuracy_by_window.csv')
        print("\nSaved ROI × context window accuracy table as roi_accuracy_by_window.csv")
        summary.to_csv('roi_best_window_summary.csv')
        print("\nSaved summary table as roi_best_window_summary.csv")
    else:
        context_window_to_acc = {}
        features_visual = load_pca_features(visual_path) if visual_path is not None else None
        features_audio = load_pca_features(audio_path) if audio_path is not None else None
        subject = int(args.subject)
        fmri = load_fmri(args.fmri_root, subject)
        for win in context_windows:
            print(f"\n=== Processing context window {win} ===\n")
            features = {}
            if features_visual is not None:
                features["visual"] = features_visual
            if features_audio is not None:
                features["audio"] = features_audio
            features["language"] = load_pca_features(language_paths[win])
            features_train, fmri_train = align_features_and_fmri_samples(
                features, fmri, 5, 5, 2, win, movies_train
            )
            model = train_encoding(features_train, fmri_train)
            features_val, fmri_val = align_features_and_fmri_samples(
                features, fmri, 5, 5, 2, win, movies_val
            )
            fmri_val_pred = model.predict(features_val)
            encoding_accuracy = np.array([
                pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
                for p in range(fmri_val.shape[1])
            ], dtype=np.float32)
            context_window_to_acc[win] = encoding_accuracy

        df = pd.DataFrame(context_window_to_acc)
        print("\n--- Accuracy DataFrame head ---")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nNumber of NaNs per column:")
        print(df.isna().sum())
        print("\nColumns:")
        print(df.columns)
        df.columns = df.columns.astype(str)
        best_windows = df.idxmax(axis=1)
        max_accuracies = df.max(axis=1)
        print("\nBest Context Window value counts:")
        print(best_windows.value_counts())
        summary = pd.DataFrame({
            'Best Context Window': best_windows,
            'Max Accuracy': max_accuracies
        })
        filename_acc = f'roi_accuracy_by_window_sub{args.subject}.csv'
        filename_summary = f'roi_best_window_summary_sub{args.subject}.csv'
        df.to_csv(filename_acc)
        print(f"\nSaved ROI × context window accuracy table as {filename_acc}")
        summary.to_csv(filename_summary)
        print(f"\nSaved summary table as {filename_summary}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    main()
