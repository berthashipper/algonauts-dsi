#!/bin/bash
#SBATCH --job-name=rob_lang_feats_ctx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rob_lang_feats_ctx_%A_%a.out
#SBATCH --error=logs/rob_lang_feats_ctx_%A_%a.err
#SBATCH --partition=general
#SBATCH --array=0-10

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

# Define context windows and folder list
CTX_WINDOWS=(15)
FOLDERS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s1"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s2"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s3"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s4"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s5"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s6"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/friends/s7"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/bourne"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/life"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/wolf"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/movies/movie10/figures"
)

# Map SLURM_ARRAY_TASK_ID to folder and context window
TASK_ID=$SLURM_ARRAY_TASK_ID
FOLDER_IDX=$TASK_ID   # integer division
CTX_IDX=0      # remainder gives context index

FOLDER=${FOLDERS[$FOLDER_IDX]}
CTX=${CTX_WINDOWS[$CTX_IDX]}

echo "Processing folder: $FOLDER with context window: ${CTX}TR"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts
python roberta_language_feature_extraction_context_batch.py \
    "$FOLDER" \
    --context_window "$CTX" \
    --middle_layer 7 \
    --max_tokens 510 \
    --kept_tokens 10 \
    --output_dir "../results/roberta_language_features"
