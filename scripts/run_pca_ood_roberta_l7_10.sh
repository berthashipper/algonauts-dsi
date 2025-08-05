#!/bin/bash
#SBATCH --job-name=pca_ood_roberta_l7_250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pca_ood_roberta_l7_250_%j.out
#SBATCH --error=logs/pca_ood_roberta_l7_250_%j.err
#SBATCH --partition=general

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/env/bin/activate

OOD_DIRS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7/ood/chaplin"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7/ood/mononoke"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7/ood/passepartout"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7/ood/planetearth"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7/ood/pulpfiction"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7/ood/wot"
)

N_COMPONENTS=250
RAW_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/roberta_language_features_l7"
SAVE_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${N_COMPONENTS}/roberta_language_l7"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

for INPUT_DIR in "${OOD_DIRS[@]}"; do
  echo "Running PCA on $INPUT_DIR with $N_COMPONENTS components"

  python pca_language_batch.py \
    --input_dirs "$INPUT_DIR" \
    --raw_root "$RAW_ROOT" \
    --save_root "$SAVE_ROOT" \
    --n_components $N_COMPONENTS
done
