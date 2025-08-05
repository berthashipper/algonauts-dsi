#!/bin/bash
#SBATCH --job-name=prepare_encoding_allsubs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=012:00:00
#SBATCH --output=logs/prepare_encoding_allsubs_%j.out
#SBATCH --error=logs/prepare_encoding_allsubs_%j.err
#SBATCH --partition=general

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FMRI_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"

TRAIN_MOVIES="friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures"
VAL_MOVIES="friends-s04,movie10-life"

SUBJECTS="1,2,3,5"

python prepare_group_encoding_model.py \
    --fmri_root "$FMRI_ROOT" \
    --subjects "$SUBJECTS" \
    --train_movies "$TRAIN_MOVIES" \
    --val_movies "$VAL_MOVIES" \
    --modality_paths "visual:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/alexnet_visual,audio:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/hubert_audio,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/language" \
    --version_name "v1_allsubs" \
    --output_model_dir "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models"
