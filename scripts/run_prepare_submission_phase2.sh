#!/bin/bash
#SBATCH --job-name=submit_phase2
#SBATCH --output=logs/submit_phase2_%A.out
#SBATCH --error=logs/submit_phase2_%A.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate the virtual environment
source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Run Phase 2 submission preparation
python prepare_submission2.py \
  --features_base_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results \
  --model_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models \
  --save_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/submission_phase2_pca10_v10_10_50 \
  --pca_dim 10 \
  --version_name v_final
