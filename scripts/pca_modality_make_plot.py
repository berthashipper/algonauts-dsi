import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ====== Toggles ======
separate_plots = True       # Separate plots per modality or combined
poster_mode = True          # Increase font sizes & marker sizes for posters

# Style settings based on mode
if poster_mode:
    title_fs = 28
    label_fs = 24
    tick_fs = 20
    legend_fs = 20
    marker_size = 14
    line_width = 4
    fig_size_per_plot = 8
else:
    title_fs = 18
    label_fs = 18
    tick_fs = 12
    legend_fs = 12
    marker_size = 8
    line_width = 2.5
    fig_size_per_plot = 6

# Load CSV
csv_path = os.path.join('logs', 'accuracy_vs_pca_final.csv')
print(f"Loading CSV from: {csv_path}")

df = pd.read_csv(csv_path)
df['pca_components'] = pd.to_numeric(df['pca_components'], errors='coerce')
df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
df = df.dropna(subset=['pca_components', 'accuracy'])
df['pca_components'] = df['pca_components'].astype(int)

# Group and pivot
mean_df = df.groupby(["modality", "pca_components"])['accuracy'].mean().reset_index()
pivot_df = mean_df.pivot(index="pca_components", columns="modality", values="accuracy")
pivot_df = pivot_df.sort_index()

if pivot_df.empty:
    raise ValueError("Pivoted DataFrame is empty.")

sns.set(style="whitegrid", context="talk")

# Filter PCA components: exclude 6-9
filtered_pcs = [pc for pc in pivot_df.index if pc not in {6,7,8,9}]
filtered_pivot_df = pivot_df.loc[filtered_pcs]
pc_labels = filtered_pcs
x_vals = np.array(pc_labels)

# Colors & styles
colors = {
    'visual': '#F7D76D',
    'audio':  '#7A9AD9',
    'language': '#B07592'
}

# ====== Plotting ======
if not separate_plots:
    plt.figure(figsize=(12, 7))

    for modality in filtered_pivot_df.columns:
        values = filtered_pivot_df[modality].values
        if pd.isna(values).all():
            continue
        plt.plot(
            x_vals,
            values,
            label=modality.capitalize(),
            color=colors[modality],
            linestyle='-',
            marker='o',
            linewidth=line_width,
            markersize=marker_size,
            alpha=0.9
        )

    plt.xticks(x_vals, [str(pc) for pc in pc_labels], fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xlabel("Principal Components", fontsize=label_fs)
    plt.ylabel("Mean Encoding Accuracy", fontsize=label_fs)
    plt.title("Mean Accuracy vs. PCA Components by Modality", fontsize=title_fs)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.legend(title="Modality", fontsize=legend_fs, title_fontsize=legend_fs, 
               loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("accuracy_vs_pca_by_modality.png", dpi=300, bbox_inches="tight")

else:
    modalities = ['visual', 'audio', 'language']
    n = len(modalities)

    fig, axes = plt.subplots(1, n, figsize=(fig_size_per_plot*n, fig_size_per_plot),
                             sharex=False, sharey=False)

    if n == 1:
        axes = [axes]

    # Custom y-limits for each modality
    y_limits = {
        'visual': (0.10, 0.12),
        'audio': (0.14, 0.16),
        'language': (0.06, 0.08)
    }

    for ax, modality in zip(axes, modalities):
        values = filtered_pivot_df[modality].values
        if pd.isna(values).all():
            continue

        ax.plot(
            x_vals,
            values,
            label=modality.capitalize(),
            color=colors[modality],
            linestyle='-',
            marker='o',
            linewidth=line_width,
            markersize=marker_size,
            alpha=0.9
        )

        ax.set_title(f"{modality.capitalize()} Accuracy", fontsize=title_fs, pad=5)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(pc) for pc in pc_labels], fontsize=tick_fs)

        ymin, ymax = y_limits[modality]
        ax.set_ylim([ymin, ymax])

        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', which='major', labelsize=tick_fs)
        ax.tick_params(axis='y', which='minor', length=4)

        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend(frameon=False, fontsize=legend_fs)

    # Centered figure labels
    fig.suptitle("Mean Accuracy vs. PCA Components by Modality", fontsize=title_fs+2, y=0.95)
    fig.supxlabel("Principal Components", fontsize=label_fs)
    fig.supylabel("Mean Encoding Accuracy", fontsize=label_fs)

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.98, wspace=0.25)

    plt.savefig("accuracy_vs_pca_separate.png", dpi=300, bbox_inches="tight")
    print("✅ Separate plots saved as 'accuracy_vs_pca_separate.png'")
