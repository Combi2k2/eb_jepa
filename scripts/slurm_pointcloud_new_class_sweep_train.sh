#!/usr/bin/env bash
#SBATCH --job-name=pc_class_sweep
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:50
#SBATCH --array=0-1%2
#SBATCH --output=slurm_pointcloud_new_class_sweep_%A_%a.out
#SBATCH --error=slurm_pointcloud_new_class_sweep_%A_%a.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_NEW_CLASS_SWEEP_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_new_class_10_30}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.new_class_sweep train \
    --config examples/pointcloud/cfgs/new_class_sweep.yaml \
    --task-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT_DIR"
