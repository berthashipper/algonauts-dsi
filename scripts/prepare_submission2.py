import os
import numpy as np
import joblib
from tqdm import tqdm
import zipfile
import argparse

def load_pca_features_ood(features_base_dir, folder, pattern, movie_name):
    """Load PCA features for both parts for a modality."""
    features = {}
    for part in [1, 2]:
        fname = pattern.format(movie=movie_name, part=part)
        path = os.path.join(features_base_dir, folder, "ood", movie_name, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        features[f"{movie_name}{part}"] = np.load(path, allow_pickle=True)
    return features

def load_all_features_phase2(features_base_dir, expected_lengths):
    movies = [
        "chaplin", "mononoke", "passepartout",
        "planetearth", "pulpfiction", "wot"
    ]
    # PCA dims per modality
    pca_dims = {
        "visual": 10,
        "audio": 10,
        "language": 50
    }
    modalities = {
        "visual": {"folder": f"pca{pca_dims['visual']}/alexnet_visual", "pattern": "{movie}{part}_features_alexnet.npy"},
        "audio": {"folder": f"pca{pca_dims['audio']}/hubert_audio_l19", "pattern": "{movie}{part}_features_audio.npy"},
        "language": {"folder": f"pca{pca_dims['language']}/roberta_language_l7", "pattern": "ood_{movie}{part}_features_language.npy"},
    }

    features_all = {}
    stimulus_window = 10
    hrf_delay = 2

    for movie in movies:
        modality_features = {}
        print(f"\nLoading features for movie '{movie}':")
        for mod_key, mod_info in modalities.items():
            modality_features[mod_key] = load_pca_features_ood(
                features_base_dir, mod_info["folder"], mod_info["pattern"], movie
            )
            for part in [1, 2]:
                movie_part_key = f"{movie}{part}"
                feat_array = modality_features[mod_key][movie_part_key]
                print(f"  {mod_key} {movie_part_key} raw shape: {feat_array.shape}")

        for part in [1, 2]:
            movie_part_key = f"{movie}{part}"
            expected_len = expected_lengths[movie_part_key]

            vis_feats = modality_features["visual"][movie_part_key]
            aud_feats = modality_features["audio"][movie_part_key]
            lang_feats = modality_features["language"][movie_part_key]

            combined_feats = []
            for s in range(expected_len):
                f_all = []

                # Visual and audio use stimulus_window of 10 timepoints each, flatten PCs per window
                for mod_key, feats in [("visual", vis_feats), ("audio", aud_feats)]:
                    dim = pca_dims[mod_key]
                    if s < (stimulus_window + hrf_delay):
                        idx_start = 0
                        idx_end = idx_start + stimulus_window
                    else:
                        idx_start = s - hrf_delay - stimulus_window + 1
                        idx_end = idx_start + stimulus_window

                    # Clamp bounds
                    idx_end = min(idx_end, len(feats))
                    idx_start = max(0, idx_end - stimulus_window)
                    f = feats[idx_start:idx_end].flatten()
                    f_all.append(f)

                # Language: single timepoint, 50 PCs
                idx = 0 if s < hrf_delay else s - hrf_delay
                idx = min(idx, len(lang_feats) - 1)
                f_all.append(lang_feats[idx].flatten())

                combined_feats.append(np.concatenate(f_all))

            features_all[movie_part_key] = np.array(combined_feats)
            print(f"Combined features for {movie_part_key} shape: {features_all[movie_part_key].shape}")

    return features_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_base_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--pca_dim", type=int, default=250)
    parser.add_argument("--version_name", type=str, required=True)
    args = parser.parse_args()

    expected_lengths = {
        "chaplin1": 432, "chaplin2": 405,
        "mononoke1": 423, "mononoke2": 426,
        "passepartout1": 422, "passepartout2": 436,
        "planetearth1": 433, "planetearth2": 418,
        "pulpfiction1": 468, "pulpfiction2": 378,
        "wot1": 353, "wot2": 324
    }

    subjects = ['01', '02', '03', '05']
    subject_keys = [f"sub-{s}" for s in subjects]

    print("Loading OOD features (phase 2)...")
    features_all = load_all_features_phase2(args.features_base_dir, expected_lengths)

    print("Loading trained models...")
    models = {}
    for sub in subjects:
        model_name = f"model_sub-{sub}_v-alexnet_visual_a-hubert_audio_l19_l-roberta_language_l7_v10_10_50.pkl"
        model_path = os.path.join(args.model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Missing model for sub-{sub}: {model_path}")
        print(f"✅ Loading model for sub-{sub} from: {model_path}")
        models[f"sub-{sub}"] = joblib.load(model_path)

    print("Predicting fMRI responses...")
    submission_preds = {subj: {} for subj in subject_keys}
    for subj in subject_keys:
        model = models[subj]
        for movie_key, feat_array in tqdm(features_all.items(), desc=f"Predicting for {subj}"):
            preds = model.predict(feat_array).astype(np.float32)
            submission_preds[subj][movie_key] = preds

    print("Saving submission files...")
    os.makedirs(args.save_dir, exist_ok=True)
    submission_npy_path = os.path.join(args.save_dir, f"submission_phase2_pca{args.pca_dim}_{args.version_name}.npy")
    np.save(submission_npy_path, submission_preds, allow_pickle=True)

    submission_zip_path = submission_npy_path.replace(".npy", ".zip")
    with zipfile.ZipFile(submission_zip_path, "w") as zipf:
        zipf.write(submission_npy_path, os.path.basename(submission_npy_path))

    print(f"Submission saved to {submission_zip_path}")

if __name__ == "__main__":
    main()
