#!/bin/bash
#SBATCH --job-name=audio_feats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/audio_feats_%A_%a.out
#SBATCH --error=logs/audio_feats_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-10

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FOLDERS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s1"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s2"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s3"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s4"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s5"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s6"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s7"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/bourne"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/wolf"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/life"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/figures"
)

FOLDER=${FOLDERS[$SLURM_ARRAY_TASK_ID]}

echo "Processing folder: $FOLDER"
cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts
python audio_feature_extraction_batch.py "$FOLDER"
