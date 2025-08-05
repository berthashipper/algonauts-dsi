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
#SBATCH --array=0-10

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# List of PCA dimensions to test
PCA_COMPONENTS=(200)

# List of input directories (length = 11)
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
)

# Calculate indices for PCA and input dir
PCA_INDEX=$((SLURM_ARRAY_TASK_ID / ${#INPUT_DIRS[@]}))
DIR_INDEX=$((SLURM_ARRAY_TASK_ID % ${#INPUT_DIRS[@]}))

N_COMPONENTS=${PCA_COMPONENTS[$PCA_INDEX]}
INPUT_DIR=${INPUT_DIRS[$DIR_INDEX]}

RAW_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/alexnet_visual_features"
SAVE_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${N_COMPONENTS}/alexnet_visual"

echo "Running PCA with ${N_COMPONENTS} components on $INPUT_DIR"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

python pca_visual_batch.py \
    --input_dirs "$INPUT_DIR" \
    --raw_root "$RAW_ROOT" \
    --save_root "$SAVE_ROOT" \
    --n_components $N_COMPONENTS
