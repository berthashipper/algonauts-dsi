#!/bin/bash
#SBATCH --job-name=pca_rob_lang_features_l7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pca_r_l_features_l7_%A_%a.out
#SBATCH --error=logs/pca_r_l_features_l7_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-10

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

CTX_VALUES=(15)

SUBFOLDERS=(
  "friends/s1"
  "friends/s2"
  "friends/s3"
  "friends/s4"
  "friends/s5"
  "friends/s6"
  "friends/s7"
  "movie10/bourne"
  "movie10/wolf"
  "movie10/life"
  "movie10/figures"
)

# Build all input directories
ALL_INPUT_DIRS=()
for CTX in "${CTX_VALUES[@]}"; do
  for SUB in "${SUBFOLDERS[@]}"; do
    ALL_INPUT_DIRS+=("/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_ctx${CTX}_l7/ctx${CTX}/${SUB}")
  done
done

# Assign task input directory
INPUT_DIR="${ALL_INPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"

# Extract ctx from INPUT_DIR path
CTX=$(basename $(dirname $(dirname "$INPUT_DIR")) | tr -dc '0-9')
N_COMPONENTS=50

RAW_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_ctx${CTX}_l7/ctx${CTX}"
SAVE_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${N_COMPONENTS}/roberta_language_l7_ctx${CTX}"

echo "Running PCA for context $CTX on: $INPUT_DIR"
cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

python pca_language_batch.py \
  --input_dirs "$INPUT_DIR" \
  --raw_root "$RAW_ROOT" \
  --save_root "$SAVE_ROOT" \
  --n_components $N_COMPONENTS
