#!/bin/bash
#SBATCH --job-name=mod_lang_feats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mod_lang_feats_%A_%a.out
#SBATCH --error=logs/mod_lang_feats_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-10

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FOLDERS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s1"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s2"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s3"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s4"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s5"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s6"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/friends/s7"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/movie10/bourne"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/movie10/wolf"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/movie10/life"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/movie10/figures"
)

FOLDER=${FOLDERS[$SLURM_ARRAY_TASK_ID]}
echo "Processing folder: $FOLDER"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts
python modern_language_feature_extraction_batch.py "$FOLDER"
