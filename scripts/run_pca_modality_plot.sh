#!/bin/bash
#SBATCH --job-name=pca_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pca_sweep_%j.out
#SBATCH --error=logs/pca_sweep_%j.err
#SBATCH --partition=general

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

python pca_modality_plot.py
