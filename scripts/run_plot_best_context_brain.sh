#!/bin/bash
#SBATCH --job-name=plot_best_context_brain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plot_best_context_%j.out
#SBATCH --error=logs/plot_best_context_%j.err
#SBATCH --partition=general

# Activate your Python environment
source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Define paths
ROI_CSV="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/roi_best_window_summary.csv"
ATLAS_PATH="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri/sub-01/atlas/sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
SCRIPT_PATH="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/plot_best_context_brain.py"
OUTPUT_DIR="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_PATH" \
    --roi_csv "$ROI_CSV" \
    --atlas_path "$ATLAS_PATH" \
    --out_png "$OUTPUT_DIR/best_context_brain_map.png"
