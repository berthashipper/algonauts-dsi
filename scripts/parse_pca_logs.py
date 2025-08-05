import re
import glob
import os

LOG_DIR = "./logs"
SUBJECTS = {1, 2, 3, 5}

log_files = glob.glob(os.path.join(LOG_DIR, "prepare_encoding_468418_*"))
accuracy_pattern = re.compile(r"MEAN_ACCURACY:\s*([\d\.]+)")

results = {}

for path in log_files:
    with open(path, "r") as f:
        content = f.read()

    # Extract (pca, subject) pairs from the model path in logs
    model_matches = re.findall(r"/models/pca(\d+)/model_sub-(\d+)_", content)
    accuracy_matches = accuracy_pattern.findall(content)

    if not model_matches or not accuracy_matches:
        continue

    for (pca, subj), acc in zip(model_matches, accuracy_matches):
        pca = int(pca)
        subj = int(subj)
        acc = float(acc)
        if subj in SUBJECTS:
            results.setdefault(subj, {})[pca] = acc

for subj in sorted(results.keys()):
    print(f"\nSubject {subj}:")
    for pca in sorted(results[subj].keys()):
        acc = results[subj][pca]
        print(f"  PCA: {pca:>3} → Accuracy: {acc:.6f}")
