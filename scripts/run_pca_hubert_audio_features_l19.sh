#!/bin/bash
#SBATCH --job-name=pca_hubert_l19_a_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pca_hubert_l19_a_features_%A_%a.out
#SBATCH --error=logs/pca_hubert_l19_a_features_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-89

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/new_env/bin/activate

# List of input directories corresponding to each array task
INPUT_DIRS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s1"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s2"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s3"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s4"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s5"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s6"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/friends/s7"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/movie10/bourne"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/movie10/wolf"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/movie10/life"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/movie10/figures"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/chaplin"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/mononoke"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/passepartout"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/planetearth"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/pulpfiction"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/wot"
)

# List of N_COMPONENTS values for the grid search
N_COMPONENTS_LIST=(5 6 7 8 9)

# Calculate the index for INPUT_DIR and N_COMPONENTS based on SLURM_ARRAY_TASK_ID
INPUT_DIR_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
N_COMPONENTS_INDEX=$((SLURM_ARRAY_TASK_ID % 5))

INPUT_DIR=${INPUT_DIRS[$INPUT_DIR_INDEX]}
N_COMPONENTS=${N_COMPONENTS_LIST[$N_COMPONENTS_INDEX]}

# Define root directories
RAW_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19"
SAVE_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${N_COMPONENTS}/hubert_audio_l19"

echo "Running PCA on HuBERT audio features in $INPUT_DIR with $N_COMPONENTS components"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

python pca_audio_batch.py \
  --input_dirs "$INPUT_DIR" \
  --raw_root "$RAW_ROOT" \
  --save_root "$SAVE_ROOT" \
  --n_components $N_COMPONENTS
