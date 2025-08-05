#!/bin/bash
#SBATCH --job-name=prepare_encoding_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_encoding_%A_%a.out
#SBATCH --error=logs/prepare_encoding_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-3  # 3 PCA values × 4 subjects = 12 jobs

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Grid values
PCA_VALUES=(200)
SUBJECT_LIST=(1 2 3 5)

# Decode array index
PCA_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SUBJECT_LIST[@]}))
SUBJ_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SUBJECT_LIST[@]}))

PCA=${PCA_VALUES[$PCA_INDEX]}
SUBJECT=${SUBJECT_LIST[$SUBJ_INDEX]}

# Set paths
FMRI_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
TRAIN_MOVIES="friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures"
VAL_MOVIES="friends-s04,movie10-life"

VISUAL_PATH="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${PCA}/alexnet_visual"
AUDIO_PATH="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/hubert_audio"
LANG_PATH="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca/language"

MODALITY_PATHS="visual:$VISUAL_PATH,audio:$AUDIO_PATH,language:$LANG_PATH"

# Optional version name includes PCA
VERSION="pca${PCA}_v1"

# Output directory
OUTPUT_DIR="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models"

# Run
python prepare_encoding_model.py \
    --fmri_root "$FMRI_ROOT" \
    --subject "$SUBJECT" \
    --train_movies "$TRAIN_MOVIES" \
    --val_movies "$VAL_MOVIES" \
    --modality_paths "$MODALITY_PATHS" \
    --version_name "$VERSION" \
    --output_model_dir "$OUTPUT_DIR"
