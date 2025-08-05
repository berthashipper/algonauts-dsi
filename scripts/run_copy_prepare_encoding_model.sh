#!/bin/bash
#SBATCH --job-name=prepare_encoding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_encoding_%A_%a.out
#SBATCH --error=logs/prepare_encoding_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=1-3,5

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FMRI_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
SUBJECT=${SLURM_ARRAY_TASK_ID}

TRAIN_MOVIES="friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures"
VAL_MOVIES="friends-s04,movie10-life"

python prepare_encoding_model.py \
    --fmri_root "$FMRI_ROOT" \
    --subject "$SUBJECT" \
    --train_movies "$TRAIN_MOVIES" \
    --val_movies "$VAL_MOVIES" \
    --modality_paths "visual:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca10/alexnet_visual,audio:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca10/hubert_audio_l19,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx3" \
    --version_name "v_ctx" \
    --output_model_dir "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models"
