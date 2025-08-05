import os
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
import joblib

def load_pca_features(modality_path):
    """Load PCA features from specified directory, preserving relative paths as keys"""
    features = {}
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"PCA modality directory not found: {modality_path}")

    for root, dirs, files in os.walk(modality_path):
        for fname in files:
            if fname.endswith(".npy"):
                full_path = os.path.join(root, fname)
                key = os.path.relpath(full_path, modality_path)
                features[key] = np.load(full_path, allow_pickle=True)

    if not features:
        raise ValueError(f"No .npy files found in {modality_path}")
    
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

    # Average repeats for figures and life
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

def align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end,
                                   hrf_delay, stimulus_window, movies):
    """Align features with fMRI samples in an efficient way."""
    aligned_features_list = []
    aligned_fmri_list = []

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
            aligned_fmri_list.append(fmri_split)

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
                        if idx >= lang_feat_len - hrf_delay:
                            f = features[mod][full_key][-1]
                        else:
                            f = features[mod][full_key][idx]
                        f_all.append(f.flatten())

                aligned_features_list.append(np.concatenate(f_all) if f_all else np.array([]))

    aligned_features = np.asarray(aligned_features_list, dtype=np.float32)
    aligned_fmri = np.concatenate(aligned_fmri_list, axis=0)

    return aligned_features, aligned_fmri

def train_encoding(features_train, fmri_train):
    """Train ridge regression encoding model using GridSearchCV to avoid LAPACK overflow"""
    alphas = [1e6, 1e5, 1e4, 1e3, 1e2, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    ridge = Ridge()
    # Limit parallel jobs to avoid OOM
    grid = GridSearchCV(ridge, param_grid={'alpha': alphas}, cv=5, n_jobs=4, verbose=2)
    grid.fit(features_train, fmri_train)
    print(f"Best alpha found: {grid.best_params_['alpha']}")
    return grid.best_estimator_

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, root_data_dir):
    """Compute and plot encoding accuracy"""
    try:
        encoding_accuracy = np.array([pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0] 
                                    for p in range(fmri_val.shape[1])], dtype=np.float32)
        mean_acc = np.round(np.mean(encoding_accuracy), 3)

        atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
        atlas_path = os.path.join(root_data_dir, f'sub-0{subject}', 'atlas', atlas_file)
        
        masker = NiftiLabelsMasker(labels_img=atlas_path)
        masker.fit()
        acc_nii = masker.inverse_transform(encoding_accuracy)

        display = plotting.plot_glass_brain(
            acc_nii, display_mode="lyrz", cmap='hot_r', colorbar=True,
            plot_abs=False, symmetric_cbar=False,
            title=f"Encoding accuracy, sub-0{subject}, mean: {mean_acc}"
        )
        
        os.makedirs("output_plots", exist_ok=True)
        display.savefig(f"output_plots/encoding_accuracy_sub{subject}.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting accuracy for sub-{subject}: {e}")
    
    print(f"MEAN_ACCURACY sub-{subject}: {mean_acc}")

def load_and_align_all_subjects(fmri_root, subjects, features, movies,
                               excluded_samples_start, excluded_samples_end,
                               hrf_delay, stimulus_window):
    all_features = []
    all_fmri = []
    fmri_lengths = []  # Store lengths for per-subject val splitting

    for subj in subjects:
        fmri = load_fmri(fmri_root, subj)
        feats, fmri_aligned = align_features_and_fmri_samples(
            features, fmri,
            excluded_samples_start,
            excluded_samples_end,
            hrf_delay,
            stimulus_window,
            movies
        )
        all_features.append(feats)
        all_fmri.append(fmri_aligned)
        fmri_lengths.append(fmri_aligned.shape[0])

    combined_features = np.concatenate(all_features, axis=0)
    combined_fmri = np.concatenate(all_fmri, axis=0)

    return combined_features, combined_fmri, fmri_lengths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_root", type=str, required=True)
    parser.add_argument("--subjects", type=str, required=True,
                        help="Comma-separated list of subject IDs, e.g. '1,2,3,4'")
    parser.add_argument("--train_movies", type=str, required=True)
    parser.add_argument("--val_movies", type=str, required=True)
    parser.add_argument("--modality_paths", type=str, required=True,
                        help="Comma-separated modality:path pairs, e.g. 'visual:/path,audio:/path'")
    parser.add_argument("--version_name", type=str, required=True,
                        help="Version tag to append to saved model filename")
    parser.add_argument("--output_model_dir", type=str, default="models",
                        help="Directory to save trained model .pkl")

    args = parser.parse_args()

    subjects = [int(s) for s in args.subjects.split(",")]
    modality_paths = dict(item.split(":") for item in args.modality_paths.split(","))
    modalities = list(modality_paths.keys())
    movies_train = args.train_movies.split(",")
    movies_val = args.val_movies.split(",")

    features = {}
    for mod in modalities:
        features[mod] = load_pca_features(modality_paths[mod])
        print(f"Successfully loaded {mod} features")

    features_train, fmri_train, _ = load_and_align_all_subjects(
        args.fmri_root, subjects, features, movies_train,
        excluded_samples_start=5, excluded_samples_end=5,
        hrf_delay=2, stimulus_window=15
    )
    features_val, fmri_val, fmri_val_lengths = load_and_align_all_subjects(
        args.fmri_root, subjects, features, movies_val,
        excluded_samples_start=5, excluded_samples_end=5,
        hrf_delay=2, stimulus_window=15
    )

    if features_train.shape[1] == 0:
        raise ValueError("No features available for training - check your input paths")

    model = train_encoding(features_train, fmri_train)

    feature_keys = [f"{mod[:1]}-{os.path.basename(os.path.normpath(path))}" for mod, path in modality_paths.items()]
    feature_string = "_".join(feature_keys)
    subject_string = "_".join(str(s) for s in subjects)

    os.makedirs(args.output_model_dir, exist_ok=True)
    model_filename = f"model_subs-{subject_string}_{feature_string}_{args.version_name}.pkl"
    model_save_path = os.path.join(args.output_model_dir, model_filename)
    joblib.dump(model, model_save_path)
    print(f"Saved trained model to {model_save_path}")

    # Evaluate per subject
    start_idx = 0
    for i, subj in enumerate(subjects):
        length = fmri_val_lengths[i]
        fmri_subj = fmri_val[start_idx:start_idx + length]
        feats_subj = features_val[start_idx:start_idx + length]
        preds_subj = model.predict(feats_subj)
        compute_encoding_accuracy(fmri_subj, preds_subj, subj, "combined", args.fmri_root)
        start_idx += length

if __name__ == "__main__":
    main()
