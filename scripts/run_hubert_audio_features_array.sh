#!/bin/bash
#SBATCH --job-name=hubert_audio_feats_l19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hubert_audio_feats_l19_%A_%a.out
#SBATCH --error=logs/hubert_audio_feats_l19_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-5

# Control threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TEMP_DIR="../results/hubert_audio_features/temp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FOLDERS=(
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s1"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s2"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s3"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s4"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s5"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s6"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s7"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/bourne"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/wolf"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/life"
#/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/figures"

# Add OOD folders here:
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/ood/chaplin"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/ood/mononoke"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/ood/passepartout"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/ood/planetearth"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/ood/pulpfiction"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/ood/wot"
)

FOLDER=${FOLDERS[$SLURM_ARRAY_TASK_ID]}

echo "Processing folder: $FOLDER"
cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts
python hubert_audio_feature_extraction_batch.py "$FOLDER"
