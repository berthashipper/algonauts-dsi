#!/bin/bash
#SBATCH --job-name=pca_alexnet_v_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pca_alexnet_v_features_%A_%a.out
#SBATCH --error=logs/pca_alexnet_v_features_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-89

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# List of input directories corresponding to each array task
INPUT_DIRS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s1"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s2"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s3"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s4"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s5"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s6"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/friends/s7"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/movie10/bourne"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/movie10/wolf"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/movie10/life"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/movie10/figures"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/ood/chaplin"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/ood/mononoke"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/ood/passepartout"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/ood/planetearth"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/ood/pulpfiction"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features/ood/wot"
)

# List of N_COMPONENTS values for the grid search (new PCA components)
N_COMPONENTS_LIST=(10)

# Calculate the index for INPUT_DIR and N_COMPONENTS based on SLURM_ARRAY_TASK_ID
INPUT_DIR_INDEX=$((SLURM_ARRAY_TASK_ID / 5))  # Get the input dir index (7 dirs)
N_COMPONENTS_INDEX=$((SLURM_ARRAY_TASK_ID % 5))  # Get the N_COMPONENTS index (5 components)

INPUT_DIR=${INPUT_DIRS[$INPUT_DIR_INDEX]}
N_COMPONENTS=${N_COMPONENTS_LIST[$N_COMPONENTS_INDEX]}

# Define root directories
RAW_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features"
SAVE_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${N_COMPONENTS}/alexnet_visual"

echo "Running PCA on AlexNet visual features in $INPUT_DIR with $N_COMPONENTS components"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

python pca_visual_batch.py \
    --input_dirs "$INPUT_DIR" \
    --raw_root "$RAW_ROOT" \
    --save_root "$SAVE_ROOT" \
    --n_components $N_COMPONENTS
