import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import h5py
from nilearn.datasets import fetch_atlas_schaefer_2018
import os
import json

# -------------- CONFIGURATION: Set these before running ----------------
subject = 1
results_dir = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/"
h5_file = '/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
split_key = 's04e01a'
roi_name = '7Networks_LH_SomMot_4'
excluded_samples_start = 5
excluded_samples_end = 5
max_timepoints = 100  # Only plot first 100 samples

# -------------- SPLIT BOUNDARIES ---------------
with open(os.path.join(results_dir, f"split_boundaries_sub{subject}.json")) as f:
    split_boundaries = json.load(f)
target_split = [s for s in split_boundaries if s['name'] == split_key][0]
start, end = target_split['start'], target_split['end']
n_timepoints = end - start
print(f"Using split '{split_key}' timepoints {start}:{end} (length {n_timepoints})")

# -------------- ROI INDEX --------------
schaefer = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
roi_labels = schaefer['labels']
roi_idx = list(roi_labels).index(roi_name)
roi_label_full = roi_labels[roi_idx + 1] if roi_idx + 1 < len(roi_labels) else f"ROI {roi_idx}"
print(f"ROI index = {roi_idx}  Schaefer-1000 label: {roi_label_full}")

# -------------- OBSERVED DATA --------------
with h5py.File(h5_file, 'r') as f:
    h5_split_key = [k for k in f.keys() if split_key in k][0]
    observed_timeseries_full = f[h5_split_key][excluded_samples_start:-excluded_samples_end, roi_idx]
assert len(observed_timeseries_full) == n_timepoints, "Observed/data lengths don't match!"

# -------------- PREDICTED DATA --------------
fmri_val_pred = np.load(os.path.join(results_dir, f"fmri_val_pred_sub{subject}.npy"))
predicted_timeseries_full = fmri_val_pred[start:end, roi_idx]
assert len(predicted_timeseries_full) == n_timepoints, "Predicted/data lengths don't match!"

# Limit to first 100 timepoints
n_plot = min(max_timepoints, n_timepoints)
observed_timeseries = observed_timeseries_full[:n_plot]
predicted_timeseries = predicted_timeseries_full[:n_plot]

# -------------- Z-SCORE EACH TIMECOURSE --------------
def zscore(ts):
    return (ts - np.mean(ts)) / np.std(ts)
observed_z = zscore(observed_timeseries)
predicted_z = zscore(predicted_timeseries)

# -------------- CORRELATION --------------
r, _ = pearsonr(observed_z, predicted_z)
print(f"Pearson r for split {split_key}, ROI {roi_name} (first {n_plot} samples): {r:.3f}")

# -------------- PLOTTING --------------
output_path = f"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/predicted_vs_observed_{split_key}_{roi_idx}_first{n_plot}.png"
time = np.arange(n_plot)
plt.figure(figsize=(12, 5))
plt.plot(time, observed_z, label='Observed', color='#1b9e77', linewidth=2)
plt.plot(time, predicted_z, label='Predicted', color='#c25400', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('fMRI Signal (z-score)')
plt.title(f'{split_key}: Predicted vs. Observed fMRI\nROI: {roi_label_full}')
plt.legend()
plt.text(0.05, 0.94, f'Pearson r = {r:.2f}', transform=plt.gca().transAxes,
         fontsize=13, verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7))
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
