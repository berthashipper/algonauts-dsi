import glob
import os
import numpy as np

def main():
    files = sorted(glob.glob("output_metrics/mean_accuracy_sub*.txt"))
    mean_accs = []

    for file in files:
        with open(file, "r") as f:
            val = float(f.read().strip())
            mean_accs.append(val)
            print(f"{os.path.basename(file)}: {val}")

    group_mean = np.round(np.mean(mean_accs), 3)
    print(f"\nMean accuracy across all {len(mean_accs)} subjects: {group_mean}")

if __name__ == "__main__":
    main()
