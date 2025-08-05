import os
import numpy as np
import h5py
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
import json

# -------- USER CONFIG --------
save_dir = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/"
os.makedirs(save_dir, exist_ok=True)

fmri_root = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
subject = 1  # or input argument
modality_paths = {
    "visual": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca9/alexnet_visual",
    "audio": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca10/hubert_audio_l19",
    "language": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx3"
}
movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s05", "friends-s06", "movie10-bourne", "movie10-figures", "movie10-wolf"]
movies_val = ["friends-s04", "movie10-life"]

excluded_samples_start = 5
excluded_samples_end = 5
hrf_delay = 2
stimulus_window = 10

# -------- FUNCTIONS --------

def load_pca_features(modality_path):
    features = {}
    for root, dirs, files in os.walk(modality_path):
        for fname in files:
            if not fname.endswith(".npy"):
                continue
            if fname in ["pca_param.npy", "scaler_param.npy"]:
                continue
            full_path = os.path.join(root, fname)
            key = os.path.relpath(full_path, modality_path)
            features[key] = np.load(full_path, allow_pickle=True)
    return features

def load_fmri(root_data_dir, subject):
    fmri = {}
    # Friends
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)
    # Movie10
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

def align_features_and_fmri_samples_track(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies):
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)
    expected_length = None
    split_boundaries = []
    current_start = 0
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
            n_timepoints = len(fmri_split)
            split_info = {'name': split, 'start': current_start, 'end': current_start + n_timepoints, 'n_timepoints': n_timepoints}
            split_boundaries.append(split_info)
            aligned_fmri = np.append(aligned_fmri, fmri_split, axis=0)
            current_start += n_timepoints
            for s in range(n_timepoints):
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
    return np.asarray(aligned_features, dtype=np.float32), aligned_fmri, split_boundaries

# -------- PIPELINE STEPS --------

# Load features and fMRI data
features = {mod: load_pca_features(modality_paths[mod]) for mod in modality_paths}
fmri = load_fmri(fmri_root, subject)

# Align training set
features_train, fmri_train, _ = align_features_and_fmri_samples_track(
    features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)

# Train encoding model
alphas = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5])
model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
model.fit(features_train, fmri_train)

# Align validation set and track splits
features_val, fmri_val, split_boundaries = align_features_and_fmri_samples_track(
    features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)

# Predict validation set
fmri_val_pred = model.predict(features_val)

# Save the predictions, observed, and split info
np.save(os.path.join(save_dir, f"fmri_val_pred_sub{subject}.npy"), fmri_val_pred)
np.save(os.path.join(save_dir, f"fmri_val_sub{subject}.npy"), fmri_val)
with open(os.path.join(save_dir, f"split_boundaries_sub{subject}.json"), 'w') as f:
    json.dump(split_boundaries, f)

print(f"Saved to {save_dir}.")
print("Validation split boundaries for post-hoc extraction:")
for s in split_boundaries:
    print(f"{s['name']}: timepoints {s['start']}–{s['end']} (n={s['n_timepoints']})")

# -------- OPTIONAL: Example of extracting a particular split
# Load again if desired, or immediately after above:
"""
import numpy as np, json
subject = 1
save_dir = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/"
with open(os.path.join(save_dir, f"split_boundaries_sub{subject}.json")) as f:
    split_boundaries = json.load(f)
target_split = [s for s in split_boundaries if s['name'] == "ses-027_task-s04e01a"][0]
start, end = target_split['start'], target_split['end']
fmri_val_pred = np.load(os.path.join(save_dir, f"fmri_val_pred_sub{subject}.npy"))
fmri_val = np.load(os.path.join(save_dir, f"fmri_val_sub{subject}.npy"))
split_pred = fmri_val_pred[start:end]
split_true = fmri_val[start:end]
# Now you can do ROI time course extraction and plotting, as in earlier answers.
"""
