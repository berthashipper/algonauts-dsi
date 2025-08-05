#!/bin/bash
#SBATCH --job-name=roi_tr_group
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/roi_tr_group_%j.out
#SBATCH --error=logs/roi_tr_group_%j.err
#SBATCH --partition=general

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FMRI_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"

# Specify subject IDs for group analysis
SUBJECTS="1,2,3,5"

TRAIN_MOVIES="friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures,movie10-wolf"
VAL_MOVIES="friends-s04,movie10-life"

# Set your desired TR/context windows
TRS="1,3,5,7,10"

# Output matrix file for ROI × TR results
OUTPUT_FILE="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roi_performance_group.npy"

python prepare_roi_encoding_model.py \
    --fmri_root "$FMRI_ROOT" \
    --subject "$SUBJECTS" \
    --train_movies "$TRAIN_MOVIES" \
    --val_movies "$VAL_MOVIES" \
    --modality_paths "language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx5" \
    --trs "$TRS" \
    --output "$OUTPUT_FILE" \
    --group_plot
