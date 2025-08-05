#!/bin/bash
#SBATCH --job-name=encoding_gridsearch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=logs/encoding_gridsearch_%A_%a.out
#SBATCH --error=logs/encoding_gridsearch_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=1-3,5

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

PCA_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca"
FMRI_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
SUBJECT=${SLURM_ARRAY_TASK_ID}

python movie_sets_grid_search.py --pca_root "$PCA_ROOT" --fmri_root "$FMRI_ROOT" --subject "$SUBJECT"
