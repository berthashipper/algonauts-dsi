#!/bin/bash
#SBATCH --job-name=pca_ood_hubert_l19_10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pca_ood_hubert_l19_10_%j.out
#SBATCH --error=logs/pca_ood_hubert_l19_10_%j.err
#SBATCH --partition=general

source /net/projects/ycleong/users/dsi_sl_2025/algonauts_2025/new_env/bin/activate

OOD_DIRS=(
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/chaplin"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/mononoke"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/passepartout"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/planetearth"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/pulpfiction"
"/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19/ood/wot"
)

N_COMPONENTS=10
RAW_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/hubert_audio_features/layer19"
SAVE_ROOT="/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/pca${N_COMPONENTS}/hubert_audio_l19"

cd /net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/scripts

for INPUT_DIR in "${OOD_DIRS[@]}"; do
  echo "Running PCA on $INPUT_DIR with $N_COMPONENTS components"

  python pca_audio_batch.py \
    --input_dirs "$INPUT_DIR" \
    --raw_root "$RAW_ROOT" \
    --save_root "$SAVE_ROOT" \
    --n_components $N_COMPONENTS
done
