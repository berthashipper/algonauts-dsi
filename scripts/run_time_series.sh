#!/bin/bash
#SBATCH --job-name=time_series
#SBATCH --output=logs/time_series_%j.out
#SBATCH --error=logs/time_series_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

PYTHON=python3
SCRIPT=/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/time_series.py

OBS_H5=/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5
FMRI_VAL_PRED=/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/fmri_val_pred_sub1.npy
ROI_NAME="7Networks_LH_SomMot_4"
SPLIT_KEY="s04e01a"

$PYTHON $SCRIPT \
    --h5_file "$OBS_H5" \
    --fmri_val_pred "$FMRI_VAL_PRED" \
    --roi_name "$ROI_NAME" \
    --split_key "$SPLIT_KEY"
