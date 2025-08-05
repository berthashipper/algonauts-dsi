import matplotlib.pyplot as plt

# Original PCA≥10 results
results = {
    1: {10: 0.193, 30: 0.188, 50: 0.185, 100: 0.182, 150: 0.179, 200: 0.177, 250: 0.177, 300: 0.175},
    2: {10: 0.181, 30: 0.176, 50: 0.172, 100: 0.168, 150: 0.165, 200: 0.163, 250: 0.162, 300: 0.162},
    3: {10: 0.195, 30: 0.191, 50: 0.187, 100: 0.181, 150: 0.179, 200: 0.178, 250: 0.178, 300: 0.176},
    5: {10: 0.160, 30: 0.154, 50: 0.151, 100: 0.146, 150: 0.144, 200: 0.143, 250: 0.142, 300: 0.141},
}

# New PCA <10 points to add
new_points = {
    1: {5: 0.192, 6: 0.192, 7: 0.192, 8: 0.192, 9: 0.193},
    2: {5: 0.179, 6: 0.179, 7: 0.180, 8: 0.180, 9: 0.181},
    3: {5: 0.195, 6: 0.195, 7: 0.195, 8: 0.195, 9: 0.196},
    5: {5: 0.159, 6: 0.159, 7: 0.160, 8: 0.160, 9: 0.160},
}

# Merge new points into original results
for subj in results:
    results[subj].update(new_points.get(subj, {}))

plt.figure(figsize=(12, 7))

for subj, pcs_acc in results.items():
    pcs = sorted(pcs_acc.keys())
    accs = [pcs_acc[pc] for pc in pcs]
    plt.plot(pcs, accs, marker='o', linewidth=2, markersize=6, label=f"Subject {subj}")

plt.xlabel("Number of PCA Components")
plt.ylabel("Mean Accuracy")
plt.title("Mean Accuracy vs. PCA Components by Subject")
plt.legend(title="Subject")
plt.grid(True)

# Extend x-axis a bit below minimum and above maximum PCA values for clarity
all_pca_values = [pc for subj in results for pc in results[subj].keys()]
min_pca, max_pca = min(all_pca_values), max(all_pca_values)
plt.xlim(min_pca - 2, max_pca + 10)

plt.tight_layout()
plt.savefig("mean_accuracy_vs_pca_extended_clear.png", dpi=300)
print("Plot saved as 'mean_accuracy_vs_pca_extended_clear.png'")
