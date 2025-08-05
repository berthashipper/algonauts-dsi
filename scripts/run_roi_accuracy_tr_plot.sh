#!/bin/bash
#SBATCH --job-name=roi_tr_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/roi_tr_plot_%j.out
#SBATCH --error=logs/roi_tr_plot_%j.err
#SBATCH --partition=general


# Activate your virtual environment, if used
source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Optional: change into the results directory (if needed)
cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

# Run your plotting script
python roi_accuracy_tr_plot.py

echo "Job completed."
