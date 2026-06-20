#!/usr/bin/env bash
#SBATCH --job-name=pc_knn_viz
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_pointcloud_knn_viz_%j.out
#SBATCH --error=slurm_pointcloud_knn_viz_%j.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312

: "${CHECKPOINT:?Set CHECKPOINT to models.pth.tar}"
: "${PRETRAIN_CLASSES:?Set PRETRAIN_CLASSES to 10, 20, or 30}"
ROTATION="${ROTATION:-none}"
ENCODER_VARIANT="${ENCODER_VARIANT:-pretrained}"
NEIGHBORS="${NEIGHBORS:-5}"
PROJECTION="${PROJECTION:-pca}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$EBJEPA_CKPTS/pointcloud/knn_visualizations/pretrain_${PRETRAIN_CLASSES}_${ROTATION}_${ENCODER_VARIANT}}"

uv run --no-sync --project "$REPO" python -m examples.pointcloud.visualize_encoder_knn \
    --checkpoint "$CHECKPOINT" \
    --pretrain-classes "$PRETRAIN_CLASSES" \
    --rotation "$ROTATION" \
    --encoder-variant "$ENCODER_VARIANT" \
    --neighbors "$NEIGHBORS" \
    --projection "$PROJECTION" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR"
