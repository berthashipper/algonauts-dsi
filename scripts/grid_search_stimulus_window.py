import subprocess
import numpy as np
import json
import os

# Define parameters
stimulus_windows = [5, 10, 15, 20, 30, 40]
subject = 1
hrf_delay = 2
excluded_samples = 5

fmri_root = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
modality_paths = {
    "visual": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/alexnet_visual",
    "audio": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/audio",
    "language": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/language"
}

train_movies="friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures"
val_movies="friends-s04,movie10-life"

# Create a temporary version of prepare_encoding_model.py that lets us call it as a function
from prepare_encoding_model import load_pca_features, load_fmri, align_features_and_fmri_samples, train_encoding, compute_encoding_accuracy

features = {
    mod: load_pca_features(path)
    for mod, path in modality_paths.items()
}

fmri = load_fmri(fmri_root, subject)

results = []

for sw in stimulus_windows:
    print(f"\n=== Testing stimulus_window = {sw} ===")

    # Align and train
    features_train, fmri_train = align_features_and_fmri_samples(
        features, fmri,
        excluded_samples_start=excluded_samples,
        excluded_samples_end=excluded_samples,
        hrf_delay=hrf_delay,
        stimulus_window=sw,
        movies=train_movies.split(",")
    )

    model = train_encoding(features_train, fmri_train)

    # Predict on train
    fmri_train_pred = model.predict(features_train)
    train_corr = np.array([
        np.corrcoef(fmri_train[:, i], fmri_train_pred[:, i])[0, 1]
        for i in range(fmri_train.shape[1])
    ])
    mean_train = np.nanmean(train_corr)

    # Align and validate
    features_val, fmri_val = align_features_and_fmri_samples(
        features, fmri,
        excluded_samples_start=excluded_samples,
        excluded_samples_end=excluded_samples,
        hrf_delay=hrf_delay,
        stimulus_window=sw,
        movies=val_movies.split(",")
    )
    fmri_val_pred = model.predict(features_val)
    val_corr = np.array([
        np.corrcoef(fmri_val[:, i], fmri_val_pred[:, i])[0, 1]
        for i in range(fmri_val.shape[1])
    ])
    mean_val = np.nanmean(val_corr)

    results.append({
        "stimulus_window": sw,
        "train_accuracy": round(mean_train, 4),
        "val_accuracy": round(mean_val, 4),
        "overfit_gap": round(mean_train - mean_val, 4)
    })

# Sort by validation accuracy
results.sort(key=lambda x: x["val_accuracy"], reverse=True)

# Print results
print("\n=== Grid Search Results ===")
for r in results:
    print(f"SW={r['stimulus_window']:>2} | Train={r['train_accuracy']:.3f} | Val={r['val_accuracy']:.3f} | Gap={r['overfit_gap']:.3f}")

# Optionally: save results
with open("grid_search_stimulus_window_results.json", "w") as f:
    json.dump(results, f, indent=2)

if __name__ == "__main__":
    print("Starting grid search over stimulus windows...")
