import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

def load_weights_by_modality(model_path, pca_dim=10):
    model = joblib.load(model_path)
    W = model.coef_  # shape: (n_voxels, 210)

    # Fixed slicing based on stimulus window assumptions
    visual_weights = W[:, :pca_dim * 10].flatten()  # first 100
    audio_weights = W[:, pca_dim * 10:pca_dim * 20].flatten()  # next 100
    language_weights = W[:, pca_dim * 20:].flatten()  # last 10

    return visual_weights, audio_weights, language_weights

def plot_weight_histograms(weight_dict, pca_dim=10, save_path=None):
    plt.figure(figsize=(18, 6))
    
    modality_colors = {
        "visual": "blue",
        "audio": "green",
        "language": "red"
    }

    for i, (modality, all_weights) in enumerate(weight_dict.items()):
        plt.subplot(1, 3, i + 1)
        plt.hist(
            all_weights, 
            bins=200, 
            color=modality_colors[modality], 
            alpha=0.7,
            range=(-0.002, 0.002)
        )
        plt.title(f"{modality.capitalize()} Weights\n(All Subjects, {pca_dim} PCs)", fontsize=14)
        plt.xlabel("Ridge Regression Weight Value", fontsize=12)
        plt.ylabel("Number of Weights", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.suptitle(
        f"Distribution of Ridge Regression Weights by Modality (PCA={pca_dim})",
        fontsize=16
    )
    plt.subplots_adjust(top=0.85)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    model_dir = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models/pca300"
    version = "v1"
    pca_dim = 300
    subjects = ['01', '02', '03', '05']

    modality_weights = {
        "visual": [],
        "audio": [],
        "language": [],
    }

    for subj in subjects:
        model_path = os.path.join(
            model_dir,
            f"model_sub-{subj}_v-alexnet_visual_a-hubert_audio_l19_l-roberta_language_l7_{version}.pkl"
        )
        print(f"Loading {model_path}")
        vis, aud, lang = load_weights_by_modality(model_path, pca_dim)
        modality_weights["visual"].extend(vis)
        modality_weights["audio"].extend(aud)
        modality_weights["language"].extend(lang)

    # Convert to numpy arrays
    for key in modality_weights:
        modality_weights[key] = np.array(modality_weights[key])

    # Save figure to this path
    save_path = f"weight_distribution_pca{pca_dim}_{version}.png"
    plot_weight_histograms(modality_weights, pca_dim=pca_dim, save_path=save_path)
