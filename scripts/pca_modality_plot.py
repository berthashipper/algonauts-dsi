import os
import numpy as np
import argparse
import csv
import subprocess
from collections import defaultdict

def run_encoding_for_modality(fmri_root, subject, train_movies, val_movies, modality, pca_path_template, pca_values):
    """Run encoding model for a single modality and return accuracy results."""
    results = {}
    
    for pca in pca_values:
        modality_path = pca_path_template.format(pca=pca)
        cmd = [
            "python", "prepare_encoding_model.py",
            "--fmri_root", fmri_root,
            "--subject", str(subject),
            "--train_movies", train_movies,
            "--val_movies", val_movies,
            "--modality_paths", f"{modality}:{modality_path}",
            "--version_name", f"pca{pca}",
            "--output_model_dir", "models",
            "--group_plot"
        ]
        
        print(f"\n=== Running for subject {subject}, {modality}, PCA={pca} ===")
        print(f"Path being used: {modality_path}")
        print("Command:", " ".join(cmd))
        
        try:
            # Check if path exists before running
            if not os.path.exists(modality_path):
                print(f"WARNING: Path does not exist: {modality_path}")
                results[pca] = None
                continue
                
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            
            # Debug output
            print("Command output:")
            print(output)
            
            # Extract mean accuracy from output
            accuracy = None
            for line in output.split('\n'):
                if "MEAN_ACCURACY:" in line:
                    accuracy = float(line.split()[-1])
                    break
            
            if accuracy is not None:
                print(f"Successfully completed - Accuracy: {accuracy}")
                results[pca] = accuracy
            else:
                print("WARNING: Could not find accuracy in output")
                results[pca] = None
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR running PCA {pca} for subject {subject}:")
            print(e.output)
            results[pca] = None
    
    return results

def main():
    # Configuration
    fmri_root = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
    subjects = [1, 2, 3, 5]  # All subjects we have data for
    train_movies = "friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures,movie10-wolf"
    val_movies = "friends-s04,movie10-life"
    
    # PCA values to test (from your plot_pca.py)
    pca_values = [150, 200, 250, 300]
    
    # Updated path templates - removed ctx3 from language path
    modality_paths = {
        "visual": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca{pca}/alexnet_visual",
        "audio": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca{pca}/hubert_audio_l19",
        "language": "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca{pca}/roberta_language_l7"  # Removed ctx3
    }
    
    # Dictionary to store all results: modality -> subject -> pca -> accuracy
    all_results = defaultdict(lambda: defaultdict(dict))
    
    print("\n=== Starting PCA Sweep Experiment ===")
    print(f"Testing modalities: {list(modality_paths.keys())}")
    print(f"Testing PCA values: {pca_values}")
    print(f"Testing subjects: {subjects}\n")
    
    # Run experiments for each modality and subject
    for modality, path_template in modality_paths.items():
        print(f"\n===== Processing modality: {modality} =====")
        print(f"Path template: {path_template}")
        
        for subject in subjects:
            print(f"\n----- Subject {subject} -----")
            results = run_encoding_for_modality(
                fmri_root, subject, train_movies, val_movies,
                modality, path_template, pca_values
            )
            all_results[modality][subject] = results
            
            # Print current results for this subject
            print(f"\nCurrent results for {modality}, subject {subject}:")
            for pca, acc in results.items():
                print(f"  PCA {pca}: {acc}")
            
            # Save intermediate results after each subject
            save_results_to_csv(all_results, "accuracy_vs_pca_intermediate.csv")
    
    # Save final results
    save_results_to_csv(all_results, "accuracy_vs_pca_final_more.csv")
    
    print("\n=== All experiments completed ===")
    print("Final results saved to accuracy_vs_pca_final.csv")

def save_results_to_csv(results_dict, filename):
    """Save the results dictionary to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['modality', 'subject', 'pca_components', 'accuracy'])
        
        # Write data rows
        for modality, subject_data in results_dict.items():
            for subject, pca_data in subject_data.items():
                for pca, accuracy in pca_data.items():
                    writer.writerow([modality, subject, pca, accuracy])
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()
