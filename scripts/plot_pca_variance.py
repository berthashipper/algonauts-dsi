import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pca_path", type=str, required=True, help="Path to pca_param.npy")
parser.add_argument("--output", type=str, default="pca_variance.png", help="Output image file")
args = parser.parse_args()

pca_info = np.load(args.pca_path, allow_pickle=True).item()
expl_var_ratio = pca_info['explained_variance_ratio_']
cum_var = np.cumsum(expl_var_ratio)

plt.figure(figsize=(8, 4))
plt.plot(cum_var, marker='o')
plt.axhline(0.90, color='r', linestyle='--', label='90% variance')
plt.axhline(0.85, color='g', linestyle='--', label='85% variance')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(args.output)
print(f"Plot saved to {args.output}")
