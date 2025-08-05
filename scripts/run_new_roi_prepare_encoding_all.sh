#!/bin/bash
#SBATCH --job-name=prepare_encoding_group
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_encoding_group_%j.out
#SBATCH --error=logs/prepare_encoding_group_%j.err
#SBATCH --partition=general

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

FMRI_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"

# List all subjects here (adjust to your actual subject IDs)
SUBJECTS="1,2,3,5"

TRAIN_MOVIES="friends-s01,friends-s02,friends-s03,friends-s05,friends-s06,movie10-bourne,movie10-figures,movie10-wolf"
VAL_MOVIES="friends-s04,movie10-life"

python new_roi_prepare_encoding_model.py \
    --fmri_root "$FMRI_ROOT" \
    --subject "$SUBJECTS" \
    --train_movies "$TRAIN_MOVIES" \
    --val_movies "$VAL_MOVIES" \
    --modality_paths "language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx1,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx2,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx3,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx4,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx5,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx6,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx7,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx8,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx9,language:/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca50/roberta_language_l7_ctx10" \
    --version_name "v_ctx1to10" \
    --output_model_dir "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts/models" \
    --group_plot \
    --context_windows 1 2 3 4 5 6 7 8 9 10
