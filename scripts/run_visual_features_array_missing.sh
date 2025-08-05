#!/bin/bash
#SBATCH --job-name=visual_feats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/visual_feats_life_%j.out
#SBATCH --error=logs/visual_feats_life_%j.err
#SBATCH --partition=general


source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate


# List of folders to process (each SLURM array task gets one)
FOLDER="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/life"

echo "Re-processing Life  folder: $FOLDER"
cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts
python visual_feature_extraction_batch.py "$FOLDER"
