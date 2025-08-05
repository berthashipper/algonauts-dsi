#!/usr/bin/env python3
import subprocess
import sys
import os
import csv
from datetime import datetime

FRIENDS_MOVIES = [
    "friends-s01", "friends-s02", "friends-s03",
    "friends-s04",  # fixed validation movie
    "friends-s05", "friends-s06",
]

MOVIE10_MOVIES = [
    "movie10-bourne",
    "movie10-figures",
    "movie10-life",
    "movie10-wolf"
]

ALL_MOVIES = FRIENDS_MOVIES + MOVIE10_MOVIES

def run_encoding_for_split(pca_root, fmri_root, subject, train_movies, val_movies):
    train_str = ",".join(train_movies)
    val_str = ",".join(val_movies)

    cmd = [
        sys.executable, "prepare_encoding_model.py",
        "--pca_root", pca_root,
        "--fmri_root", fmri_root,
        "--subject", str(subject),
        "--train_movies", train_str,
        "--val_movies", val_str
    ]

    print(f"Running command:\n{' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fname = f"logs/encoding_sub{subject}_{'_'.join(val_movies)}_{timestamp}.log"
    with open(log_fname, "w") as f:
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)

    print(f"Logs saved to {log_fname}", flush=True)

    mean_accuracy = None
    for line in proc.stdout.splitlines():
        if line.startswith("MEAN_ACCURACY:"):
            try:
                mean_accuracy = float(line.split(":")[1].strip())
            except Exception as e:
                print(f"Failed to parse mean accuracy: {e}", flush=True)

    if mean_accuracy is None:
        print("WARNING: Mean accuracy not found in output!", flush=True)

    return mean_accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_root", type=str, required=True)
    parser.add_argument("--fmri_root", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    args = parser.parse_args()

    pca_root = args.pca_root
    fmri_root = args.fmri_root
    subject = args.subject

    if not os.path.exists("logs"):
        os.makedirs("logs")

    results = []

    for movie10 in MOVIE10_MOVIES:
        val_movies = ["friends-s04", movie10]
        train_movies = [m for m in ALL_MOVIES if m not in val_movies]
        print(f"\n===== Training on {train_movies} | Validating on {val_movies} =====\n", flush=True)

        acc = run_encoding_for_split(pca_root, fmri_root, subject, train_movies, val_movies)
        results.append((subject, ",".join(train_movies), ",".join(val_movies), acc))
        print(f"Result: train on {train_movies}, validate on {val_movies}, mean accuracy = {acc}", flush=True)

    # Save results CSV
    out_csv = f"logs/encoding_s04+movie10_gridsearch_sub{subject}.csv"
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "train_movies", "val_movies", "mean_accuracy"])
        writer.writerows(results)

    print(f"\nSaved grid search results to {out_csv}", flush=True)

    # Print best result
    best = max(results, key=lambda x: (x[3] if x[3] is not None else -1))
    print(f"\nBEST COMBINATION for subject {subject}:")
    print(f"Train on: {best[1]}")
    print(f"Validate on: {best[2]}")
    print(f"Mean accuracy: {best[3]}")

if __name__ == "__main__":
    main()
