import numpy as np
import nibabel as nib
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. Specify your context windows (TRs) in order
tr_list = [1, 3, 5, 7, 10]

# 2. Load group ROI × TR matrix and the Schaefer-1000 atlas
roi_performance = np.load('/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roi_performance_group.npy')
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
atlas_img = nib.load(atlas['maps'])
atlas_data = atlas_img.get_fdata().astype(int)

# 3. For each ROI, select the TR index (0-based) that yielded maximum accuracy
best_tr_idx_per_roi = np.argmax(roi_performance, axis=1)

# 4. Build a volume for visualization: background=255, ROIs get TR indices (0=tr_list[0], 1=tr_list[1], ...)
best_tr_map = np.full_like(atlas_data, fill_value=255, dtype=np.uint8)
roi_labels = np.unique(atlas_data)[1:]
for roi_idx, roi_number in enumerate(roi_labels):
    best_tr_map[atlas_data == roi_number] = best_tr_idx_per_roi[roi_idx]
masked_tr_map = np.ma.masked_where(best_tr_map == 255, best_tr_map)
best_tr_img = nib.Nifti1Image(masked_tr_map.filled(0).astype(np.uint8), atlas_img.affine)

# 5. Discrete colormap for number of TRs; set masked and under colors as white
cmap = plt.get_cmap('tab10', len(tr_list))
cmap.set_bad('white')
cmap.set_under('white')

# 6. Create a figure of your chosen size for glass brain plotting
fig = plt.figure(figsize=(9, 5), facecolor='white')

display = plotting.plot_glass_brain(
    best_tr_img,
    display_mode='ortho',
    cmap=cmap,
    colorbar=False,          # We use a legend, not a default colorbar, for clarity
    vmin=0,
    vmax=len(tr_list)-1,
    plot_abs=False,
    figure=fig,
    title='TR with Maximal Accuracy per ROI (Group Average)'
)

# 7. Add a categorical legend inside the plot matching color to TR value
legend_patches = [Patch(color=cmap(i), label=f'TR {tr_list[i]}') for i in range(len(tr_list))]
plt.legend(
    handles=legend_patches,
    loc='upper right',
    bbox_to_anchor=(0.99, 0.98),
    frameon=True,
    fontsize=13,
    title="Context window (TR)"
)

plt.tight_layout()
plt.savefig('/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roi_tr_max_accuracy_glass_brain_with_legend.png',
            dpi=300, facecolor='white')
plt.close()
