#!/usr/bin/env bash
#SBATCH --job-name=pc_ratio_knn
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-11%3
#SBATCH --output=slurm_pointcloud_ratio_knn_%A_%a.out
#SBATCH --error=slurm_pointcloud_ratio_knn_%A_%a.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${SLURM_JOB_ID}}"
CHECKPOINT_ROOT="${POINTCLOUD_VICREG_TUNED_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_vicreg_tuned_ratio_matched}"
OUTPUT_DIR="${POINTCLOUD_RATIO_KNN_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_ratio_encoder_knn}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.ratio_encoder_knn run \
    --config examples/pointcloud/cfgs/ratio_encoder_knn.yaml \
    --task-id "$SLURM_ARRAY_TASK_ID" --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$OUTPUT_DIR"
