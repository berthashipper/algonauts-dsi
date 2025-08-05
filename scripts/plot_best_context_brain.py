import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--roi_csv", type=str, required=True, help="CSV file with best context window per ROI")
parser.add_argument("--atlas_path", type=str, required=True, help="Atlas NIfTI labels file")
parser.add_argument("--out_png", type=str, required=True, help="Output PNG file path")
args = parser.parse_args()

best = pd.read_csv(args.roi_csv, index_col=0)
windows = sorted(best['Best Context Window'].unique())
n_rois = best.index.max() + 1

# Reserve 0 for background, so mapping starts at 1
win2code = {w: i+1 for i, w in enumerate(windows)}
roi_codes = np.zeros(n_rois, dtype=np.int32)
for roi_idx, w in best['Best Context Window'].items():
    roi_codes[int(roi_idx)] = win2code[w]

atlas_img = nib.load(args.atlas_path)
masker = NiftiLabelsMasker(labels_img=atlas_img)
masker.fit()
ctx_img = masker.inverse_transform(roi_codes)

# --- NEW: use sequential colormap, lighter (min window) to darker (max window)
seq_cmap = plt.get_cmap('Blues', len(windows))
windows_arr = np.array(windows)
color_list = [(0, 0, 0, 0)]  # For background

for w in windows:
    # Map window value to [0, 1] for colormap
    norm_val = (w - windows_arr.min()) / (windows_arr.max() - windows_arr.min())
    color_list.append(seq_cmap(norm_val))

custom_cmap = mcolors.ListedColormap(color_list)

display = plotting.plot_glass_brain(
    ctx_img,
    title='Best Context Window Accuracy per ROI',
    cmap=custom_cmap,
    colorbar=False,
    display_mode='ortho',
    alpha=1,
    threshold=0.01
)

# Optionally add percent annotation to legend (recommended for clarity)
counts = best['Best Context Window'].value_counts()
percentages = counts / counts.sum() * 100
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
        label=f"{w} ({percentages[w]:.1f}%)",
        markerfacecolor=color_list[win2code[w]], markersize=10)
    for w in windows
]

fig = plt.gcf()
fig.legend(
    handles=handles,
    title="Best Context Window",
    loc='lower center',
    bbox_to_anchor=(0.5, -0.06),
    ncol=len(windows)
)

plt.savefig(args.out_png, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved glass brain projection: {args.out_png}")
