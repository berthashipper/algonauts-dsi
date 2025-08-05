import os
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
import joblib

def load_pca_features(modality_path):
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

def align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end,
                                  hrf_delay, stimulus_window, movies):
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)

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

                aligned_features.append(np.concatenate(f_all) if f_all else np.array([]))

    return np.asarray(aligned_features, dtype=np.float32), aligned_fmri


def train_encoding(features_train, fmri_train):
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    param_grid = {
        'alpha': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
    }

    n_voxels = fmri_train.shape[1]
    models = []
    best_params_list = []

    for v in range(n_voxels):
        y = fmri_train[:, v]

        krr = KernelRidge(kernel='rbf')
        grid = GridSearchCV(krr, param_grid, cv=3, n_jobs=1, verbose=0, scoring='neg_mean_squared_error')

        grid.fit(features_train_scaled, y)

        models.append(grid.best_estimator_)
        best_params_list.append(grid.best_params_)

        if v % 100 == 0:
            print(f"Trained voxel {v+1}/{n_voxels}")

    print(f"Sample best params for voxel 0: {best_params_list[0]}")

    return models, scaler

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, root_data_dir):
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
        print(f"Error plotting accuracy: {e}")
    
    print(f"MEAN_ACCURACY: {mean_acc}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_root", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--train_movies", type=str, required=True)
    parser.add_argument("--val_movies", type=str, required=True)
    parser.add_argument("--modality_paths", type=str, required=True,
                  help="Comma-separated modality:path pairs, e.g. 'visual:/path1,audio:/path2'")
    parser.add_argument("--version_name", type=str, required=True,
                  help="Version tag to append to saved model filename")
    parser.add_argument("--output_model_dir", type=str, default="models",
                  help="Directory to save trained model and scaler .pkl")

    args = parser.parse_args()

    modality_paths = dict(item.split(":") for item in args.modality_paths.split(","))
    modalities = list(modality_paths.keys())
    movies_train = args.train_movies.split(",")
    movies_val = args.val_movies.split(",")

    features = {}
    for mod in modalities:
        try:
            features[mod] = load_pca_features(modality_paths[mod])
            print(f"Successfully loaded {mod} features")
        except Exception as e:
            print(f"Failed to load {mod} features: {e}")
            continue

    if not features:
        raise ValueError("No features were successfully loaded!")

    fmri = load_fmri(args.fmri_root, args.subject)

    features_train, fmri_train = align_features_and_fmri_samples(
        features, fmri, 
        excluded_samples_start=5, 
        excluded_samples_end=5,
        hrf_delay=2, 
        stimulus_window=15,
        movies=movies_train
    )

    if features_train.shape[1] == 0:
        raise ValueError("No features available for training - check your input paths")

    model, scaler = train_encoding(features_train, fmri_train)

    model_list, scaler = train_encoding(features_train, fmri_train)

    feature_keys = []
    for mod, path in modality_paths.items():
        base = os.path.basename(os.path.normpath(path))
        feature_keys.append(f"{mod[:1]}-{base}")
    feature_string = "_".join(feature_keys)

    os.makedirs(args.output_model_dir, exist_ok=True)
    model_list, scaler = train_encoding(features_train, fmri_train)

    # Save all models, e.g. using joblib dump with compression
    model_filename = f"models_sub-0{args.subject}_{feature_string}_{args.version_name}.pkl"
    scaler_filename = f"scaler_sub-0{args.subject}_{feature_string}_{args.version_name}.pkl"

    joblib.dump(model_list, os.path.join(args.output_model_dir, model_filename), compress=3)
    joblib.dump(scaler, os.path.join(args.output_model_dir, scaler_filename))

    print(f"Saved trained models to {os.path.join(args.output_model_dir, model_filename)}")
    print(f"Saved feature scaler to {os.path.join(args.output_model_dir, scaler_filename)}")

    features_val, fmri_val = align_features_and_fmri_samples(
        features, fmri,
        excluded_samples_start=5,
        excluded_samples_end=5,
        hrf_delay=2,
        stimulus_window=15,
        movies=movies_val
    )

    # Apply scaler transform on validation features before prediction
    features_val_scaled = scaler.transform(features_val)
    # Predict per voxel by calling predict for each model and stacking results
    fmri_val_pred = np.column_stack([model.predict(features_val_scaled) for model in model_list])

    compute_encoding_accuracy(fmri_val, fmri_val_pred, args.subject, "combined", args.fmri_root)

if __name__ == "__main__":
    main()
