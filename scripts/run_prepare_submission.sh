#!/bin/bash
#SBATCH --job-name=prepare_submission
#SBATCH --output=logs/submission_%A.out
#SBATCH --error=logs/submission_%A.err
#SBATCH --time=01:00:00            # max runtime hh:mm:ss
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Run the prepare_submission script
python prepare_submission.py \
  --features_base_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results \
  --root_data_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data \
  --model_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models \
  --save_dir /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/submission_alexnet_hubertl19_robertal7ctx3 \
  --version_name v_final
