import os
import numpy as np
import joblib
from tqdm import tqdm
import zipfile
import argparse

def load_pca_features(modality_path):
    """
    Load PCA features from all .npy files under modality_path recursively,
    return dict with keys as relative file paths and values as loaded np arrays.
    """
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

    print(f"Loaded {len(features)} PCA feature files from {modality_path}")
    return features


def load_stimulus_features_friends_s7(visual_base, audio_base, language_base):
    """Load PCA features for friends season 7 from specified modality directories."""
    features_friends_s7 = {}

    visual_dir = os.path.join(visual_base, 'alexnet_visual', 'friends', 's7')
    audio_dir = os.path.join(audio_base, 'hubert_audio_l19', 'friends', 's7')
    language_dir = os.path.join(language_base, 'roberta_language_l7_ctx3', 'friends', 's7')

    features_friends_s7['visual'] = load_pca_features(visual_dir)
    features_friends_s7['audio'] = load_pca_features(audio_dir)
    features_friends_s7['language'] = load_pca_features(language_dir)

    return features_friends_s7

def align_friends_s7_for_submission(features_friends_s7, root_data_dir,
                                    hrf_delay=2, stimulus_window=15):
    aligned_features = {}
    subjects = [1, 2, 3, 5]

    for sub in subjects:
        subj_key = f'sub-0{sub}'
        aligned_features[subj_key] = {}

        sample_path = os.path.join(
            root_data_dir, 'fmri', subj_key, 'target_sample_number',
            f'{subj_key}_friends-s7_fmri_samples.npy'
        )
        fmri_samples_dict = np.load(sample_path, allow_pickle=True).item()

        for epi, n_samples in fmri_samples_dict.items():
            features_epi = []
            for s in range(n_samples):
                f_all = []

                for mod in ['visual', 'audio', 'language']:
                    # Find matching PCA feature key for episode
                    # Features keys are relative paths like 's7e01_visual.npy' or similar
                    # We need to find the key containing the episode string 'epi'
                    mod_features = features_friends_s7[mod]
                    matched_keys = [k for k in mod_features.keys() if epi in k]
                    if not matched_keys:
                        raise KeyError(f"No PCA features for episode '{epi}' in modality '{mod}'")
                    # Assuming only one match
                    mod_feats = mod_features[matched_keys[0]]

                    if mod in ['visual', 'audio']:
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window

                        if idx_end > len(mod_feats):
                            idx_end = len(mod_feats)
                            idx_start = idx_end - stimulus_window

                        f = mod_feats[idx_start:idx_end].flatten()
                        f_all.append(f)

                    elif mod == 'language':
                        idx = 0 if s < hrf_delay else s - hrf_delay
                        if idx >= len(mod_feats):
                            idx = len(mod_feats) - 1
                        f = mod_feats[idx].flatten()
                        f_all.append(f)

                features_epi.append(np.concatenate(f_all))
            aligned_features[subj_key][epi] = np.array(features_epi, dtype=np.float32)

    return aligned_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_base_dir", type=str, required=True,
                        help="Base directory where PCA features are stored")
    parser.add_argument("--root_data_dir", type=str, required=True,
                        help="Root data directory for fmri samples files")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model .pkl files")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save submission predictions and zip")
    parser.add_argument("--hrf_delay", type=int, default=2,
                        help="HRF delay for alignment")
    parser.add_argument("--stimulus_window", type=int, default=10,
                        help="Stimulus window for alignment")
    parser.add_argument("--version_name", type=str, required=True,
                    help="Version tag used in model filenames (e.g. 'v1')")

    args = parser.parse_args()

    print("Loading stimulus features...")
    visual_base = os.path.join(args.features_base_dir, 'pca10')
    audio_base = os.path.join(args.features_base_dir, 'pca10')
    language_base = os.path.join(args.features_base_dir, 'pca50')

    print(f"Visual features from: {visual_base}")
    print(f"Audio features from: {audio_base}")
    print(f"Language features from: {language_base}")

    features_friends_s7 = load_stimulus_features_friends_s7(
        visual_base=visual_base,
        audio_base=audio_base,
        language_base=language_base
    )

    print("Aligning features with sample counts...")
    aligned_features_friends_s7 = align_friends_s7_for_submission(
        features_friends_s7,
        args.root_data_dir,
        hrf_delay=args.hrf_delay,
        stimulus_window=args.stimulus_window
    )

    print("Loading trained models...")
    my_models = {}

    # Use this fixed pattern that matches your actual model files
    feature_string = "v-alexnet_visual_a-hubert_audio_l19_l-language"
    print(f"Feature string used for model filenames: {feature_string}")

    for sub in ['01', '02', '03', '05']:
        model_filename = f"model_sub-{sub}_v-alexnet_visual_a-hubert_audio_l19_l-roberta_language_l7_v10_10_50.pkl"
        model_path = os.path.join(args.model_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        my_models[f"sub-{sub}"] = joblib.load(model_path)

    print("Predicting fMRI responses for submission...")
    submission_predictions = {}
    for sub, features_dict in tqdm(aligned_features_friends_s7.items(), desc="Predicting"):
        submission_predictions[sub] = {}
        for epi, feat_epi in features_dict.items():
            fmri_pred = my_models[sub].predict(feat_epi).astype(np.float32)
            submission_predictions[sub][epi] = fmri_pred

    print("Saving predictions to file and zipping...")
    os.makedirs(args.save_dir, exist_ok=True)
    output_file = os.path.join(args.save_dir, "fmri_predictions_friends_s7.npy")
    np.save(output_file, submission_predictions)
    print(f"Saved predictions to {output_file}")

    zip_file = os.path.join(args.save_dir, "fmri_predictions_friends_s7.zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Zipped submission file to {zip_file}")


if __name__ == "__main__":
    main()
