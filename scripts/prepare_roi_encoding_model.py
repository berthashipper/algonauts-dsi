import os
import numpy as np
import h5py
import argparse
matplotlib_backend = "Agg"
import matplotlib
matplotlib.use(matplotlib_backend)
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Util: Load PCA features
def load_pca_features(modality_path):
    features = {}
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"PCA modality directory not found: {modality_path}")

    for root, dirs, files in os.walk(modality_path):
        for fname in files:
            if not fname.endswith(".npy"):
                continue
            if fname in ["pca_param.npy", "scaler_param.npy"]:
                continue
            full_path = os.path.join(root, fname)
            key = os.path.relpath(full_path, modality_path)
            features[key] = np.load(full_path, allow_pickle=True)

    if not features:
        raise ValueError(f"No valid .npy feature files found in {modality_path}")

    print(f"Loaded {len(features)} feature files from {modality_path}")
    return features

# Util: Load fMRI for subject
def load_fmri(root_data_dir, subject):
    fmri = {}
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)
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

# Util: Align features & fMRI
def align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end,
                                   hrf_delay, stimulus_window, movies):
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

# Util: Train encoding model
def train_encoding(features_train, fmri_train):
    alphas = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5])
    model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    model.fit(features_train, fmri_train)
    return model

# Main group/subject pipeline
def compute_roi_performance_across_trs(
    modalities,
    root_data_dir,
    subject,
    movies_train,
    movies_val,
    TR_list,
    excluded_samples_start=5,
    excluded_samples_end=5,
    hrf_delay=2
):
    n_rois = 1000
    n_trs = len(TR_list)
    roi_performance = np.zeros((n_rois, n_trs), dtype=np.float32)
    features = {mod: load_pca_features(modalities[mod]) for mod in modalities}
    fmri = load_fmri(root_data_dir, subject)
    for tr_idx, tr_setting in enumerate(TR_list):
        print(f"Processing TR/context {tr_setting} ({tr_idx+1}/{n_trs})")
        features_val, fmri_val = align_features_and_fmri_samples(
            features, fmri,
            excluded_samples_start=excluded_samples_start,
            excluded_samples_end=excluded_samples_end,
            hrf_delay=hrf_delay,
            stimulus_window=tr_setting,
            movies=movies_val
        )
        features_train, fmri_train = align_features_and_fmri_samples(
            features, fmri,
            excluded_samples_start=excluded_samples_start,
            excluded_samples_end=excluded_samples_end,
            hrf_delay=hrf_delay,
            stimulus_window=tr_setting,
            movies=movies_train
        )
        if features_val.shape[0] == 0 or fmri_val.shape[0] == 0:
            print(f"No validation data for TR/context {tr_setting}, skipping.")
            continue
        model = train_encoding(features_train, fmri_train)
        fmri_val_pred = model.predict(features_val)
        encoding_accuracy = np.array([
            pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
            for p in range(fmri_val.shape[1])
        ], dtype=np.float32)
        roi_performance[:, tr_idx] = encoding_accuracy
        print(f"Finished TR/context {tr_setting}")
    return roi_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_root", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True,
        help="Subject number or comma-separated subject list (e.g. 1 or '1,2,3,5')")
    parser.add_argument("--train_movies", type=str, required=True)
    parser.add_argument("--val_movies", type=str, required=True)
    parser.add_argument("--modality_paths", type=str, required=True,
        help="Comma-separated modality:path pairs, e.g. 'visual:/path1,audio:/path2'")
    parser.add_argument("--version_name", type=str, default="",
        help="Version tag to append to saved model filename")
    parser.add_argument("--output_model_dir", type=str, default="models",
        help="Directory to save trained model .pkl")
    parser.add_argument("--trs", type=str, default='1,2,3,5,8,10,12,14,16,20',
        help="Comma-separated list of temporal windows (e.g. TRs) for analysis")
    parser.add_argument("--output", type=str, default="roi_performance.npy",
        help="Output .npy file for the ROI x TR matrix")
    parser.add_argument("--group_plot", action="store_true",
        help="If set, computes mean accuracy plot over all subjects and saves group-level outputs")
    args = parser.parse_args()

    modalities = dict(item.split(":") for item in args.modality_paths.split(","))
    movies_train = args.train_movies.split(",")
    movies_val = args.val_movies.split(",")
    TR_list = [int(x) for x in args.trs.split(",")]

    if args.group_plot:
        subjects = [int(s) for s in args.subject.split(",")]
        roi_performance_all = []
        for subject in subjects:
            print(f"Processing subject {subject} for group mean plot")
            roi_performance = compute_roi_performance_across_trs(
                modalities,
                root_data_dir=args.fmri_root,
                subject=subject,
                movies_train=movies_train,
                movies_val=movies_val,
                TR_list=TR_list
            )
            roi_performance_all.append(roi_performance)
        roi_performance_all = np.array(roi_performance_all)
        group_mean = np.mean(roi_performance_all, axis=0)  # (n_rois, n_trs)
        np.save(args.output, group_mean)
        print(f"Saved group mean ROI × TR performance matrix to {args.output}")
    else:
        subject = int(args.subject)
        roi_performance = compute_roi_performance_across_trs(
            modalities,
            root_data_dir=args.fmri_root,
            subject=subject,
            movies_train=movies_train,
            movies_val=movies_val,
            TR_list=TR_list
        )
        np.save(args.output, roi_performance)
        print(f"Saved ROI × TR performance matrix to {args.output}")

