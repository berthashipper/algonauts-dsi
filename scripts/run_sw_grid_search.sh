#!/bin/bash
#SBATCH --job-name=grid_search_sw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/grid_search_sw_%A_%a.out
#SBATCH --error=logs/grid_search_sw_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=1-3,5

# Activate environment
source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Set subject for each job in array
export SUBJECT=${SLURM_ARRAY_TASK_ID}

# Export SUBJECT so the Python script can read it via os.environ
python grid_search_stimulus_window.py
