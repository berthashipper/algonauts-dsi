import os
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting


def load_pca_features(pca_root_dir, modality):
    features = {}
    modality_dir = os.path.join(pca_root_dir, modality)
    if not os.path.exists(modality_dir):
        raise FileNotFoundError(f"PCA modality directory not found: {modality_dir}")

    for root, dirs, files in os.walk(modality_dir):
        for fname in files:
            if fname.endswith(f"features_{modality}.npy"):
                base_key = fname.replace(f"_features_{modality}.npy", "")
                # Extract relative path parts
                rel_path = os.path.relpath(root, modality_dir)  # e.g. 'friends/s4' or 'movie10/bourne'
                parts = rel_path.split(os.sep)

                # If Movie10 modality path (like 'movie10/bourne'), prepend second-level folder name to key
                if parts[0] == "friends":
                    # For friends: keep just filename key (which encodes season and episode)
                    key = base_key
                elif parts[0] == "movie10":
                    # For movie10: prepend movie folder (e.g. 'bourne') to filename key with underscore
                    if len(parts) > 1:
                        key = parts[1] + "_" + base_key
                    else:
                        key = base_key
                else:
                    key = base_key

                full_path = os.path.join(root, fname)
                features[key] = np.load(full_path)

    print(f"Loaded {len(features)} {modality} feature files from {modality_dir}")
    return features


def load_fmri(root_data_dir, subject):
    """
    Load fMRI responses for the given subject.
    """
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

    # Average repeats for 'figures' splits
    keys_all = list(fmri.keys())
    for s in range(12):
        movie = f'figures{str(s+1).zfill(2)}'
        reps = [k for k in keys_all if movie in k]
        if len(reps) == 2:
            fmri[movie] = ((fmri[reps[0]] + fmri[reps[1]]) / 2).astype(np.float32)
            del fmri[reps[0]]
            del fmri[reps[1]]

    # Average repeats for 'life' splits
    keys_all = list(fmri.keys())
    for s in range(5):
        movie = f'life{str(s+1).zfill(2)}'
        reps = [k for k in keys_all if movie in k]
        if len(reps) == 2:
            fmri[movie] = ((fmri[reps[0]] + fmri[reps[1]]) / 2).astype(np.float32)
            del fmri[reps[0]]
            del fmri[reps[1]]

    return fmri


def align_features_and_fmri_samples(features, fmri,
                                   excluded_samples_start, excluded_samples_end,
                                   hrf_delay, stimulus_window, movies):
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)

    for movie in movies:
        if movie.startswith('friends'):
            season_num = movie.split('-')[1][1:]  # 's07' -> '07'
            season_prefix = f"s{season_num}e"      # match fmri keys like 's07e01a'
            movie_splits = [key for key in fmri if key.startswith(season_prefix)]
        elif movie.startswith('movie10'):
            movie_folder = movie.split('-')[1]  # e.g. 'bourne'
            movie_splits = [key for key in fmri if key.startswith(movie_folder)]
        else:
            movie_splits = [key for key in fmri if key.startswith(movie)]

        if len(movie_splits) == 0:
            print(f"Warning: no fmri keys found for movie '{movie}'")

        for split in movie_splits:
            fmri_split = fmri[split]
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, axis=0)

            for s in range(len(fmri_split)):
                f_all = np.empty(0)
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

                        f = features[mod][full_key][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    elif mod == 'language':
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay

                        lang_feat_len = len(features[mod][full_key])
                        if idx >= lang_feat_len - hrf_delay:
                            f = features[mod][full_key][-1]
                        else:
                            f = features[mod][full_key][idx]

                        f_all = np.append(f_all, f.flatten())

                aligned_features.append(f_all)

    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    return aligned_features, aligned_fmri


def train_encoding(features_train, fmri_train):
    """
    Train a ridge-regression-based encoding model to predict fMRI responses
    using movie features. RidgeCV selects the best alpha per voxel.

    Parameters
    ----------
    features_train : np.ndarray
        Stimulus features (samples × features)
    fmri_train : np.ndarray
        fMRI responses (samples × voxels/parcels)

    Returns
    -------
    model : RidgeCV object
        Trained model with coefficients and intercepts.
    """
    alphas = np.array([
        1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1,
        0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5
    ])

    model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    model.fit(features_train, fmri_train)
    print(f"Selected alphas (per voxel): shape {model.alpha_.shape}")
    return model



def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, root_data_dir):
    try:
        encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
        for p in range(len(encoding_accuracy)):
            encoding_accuracy[p] = pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
        mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

        atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
        atlas_path = os.path.join(
            root_data_dir,
            f'sub-0{subject}',
            'atlas',
            atlas_file
        )

        atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
        atlas_masker.fit()
        encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

        title = f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: {mean_encoding_accuracy}"
        display = plotting.plot_glass_brain(
            encoding_accuracy_nii,
            display_mode="lyrz",
            cmap='hot_r',
            colorbar=True,
            plot_abs=False,
            symmetric_cbar=False,
            title=title
        )
        colorbar = display._cbar
        colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)

        # Save the figure to a folder
        output_dir = "output_plots"
        os.makedirs(output_dir, exist_ok=True)
        fig = display.frame_axes.figure
        fig.savefig(os.path.join(output_dir, f"encoding_accuracy_sub{subject}_{modality}.png"))

        plt.close(fig)
        print(f"Saved plot for modality {modality}", flush=True)

    except Exception as e:
        print(f"Failed to plot encoding accuracy for {modality}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_root", type=str, required=True, help="Root directory for PCA features")
    parser.add_argument("--fmri_root", type=str, required=True, help="Root directory for fMRI data")
    parser.add_argument("--subject", type=int, required=True, help="Subject number")
    args = parser.parse_args()

    pca_features_root = args.pca_root
    root_data_dir = args.fmri_root
    subject = args.subject

    # Updated parameters as you requested:
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 3
    stimulus_window = 5

    movies_train = [
        "friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05",
        "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"
    ]
    movies_val = ["friends-s06"]

    # Load features for all modalities
    features = {}
    for modality in ["visual", "audio", "language"]:
        try:
            features[modality] = load_pca_features(pca_features_root, modality)
        except Exception as e:
            print(f"Warning: Could not load {modality} features: {e}")

    fmri = load_fmri(root_data_dir, subject)

    # Align training data (concatenated features)
    features_train, fmri_train = align_features_and_fmri_samples(
        features, fmri,
        excluded_samples_start, excluded_samples_end,
        hrf_delay, stimulus_window,
        movies_train
    )

    print(f"Subject {subject} Training fMRI responses shape:")
    print(fmri_train.shape)
    print('(Train samples × Parcels)')
    print(f"\nSubject {subject} Training stimulus features shape:")
    print(features_train.shape)
    print('(Train samples × Features)')

    # Train encoding model on all combined features
    model = train_encoding(features_train, fmri_train)
    print("Encoding model trained successfully.", flush=True)

    del features_train, fmri_train  # free memory

    # Align validation data (combined features)
    features_val, fmri_val = align_features_and_fmri_samples(
        features, fmri,
        excluded_samples_start, excluded_samples_end,
        hrf_delay, stimulus_window,
        movies_val
    )

    del features, fmri  # free memory

    print("Validation fMRI responses shape:", fmri_val.shape, flush=True)
    print('(Validation samples × Parcels)', flush=True)
    print("Validation stimulus features shape:", features_val.shape, flush=True)
    print('(Validation samples × Features)', flush=True)

    # Predict validation fMRI responses
    fmri_val_pred = model.predict(features_val)

    print("Validation fMRI responses shape:", fmri_val.shape, flush=True)
    print('(Validation samples × Parcels)', flush=True)
    print("Validation predicted fMRI responses shape:", fmri_val_pred.shape, flush=True)
    print('(Validation samples × Parcels)', flush=True)

    # Compute and plot encoding accuracy for all combined modalities
    compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality="all", root_data_dir=root_data_dir)

if __name__ == "__main__":
    main()
