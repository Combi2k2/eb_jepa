#!/usr/bin/env bash
#SBATCH --job-name=pc_tune_param
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:50
#SBATCH --array=0-3%3
#SBATCH --output=slurm_pointcloud_tune_param_%A_%a.out
#SBATCH --error=slurm_pointcloud_tune_param_%A_%a.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_TUNE_PARAM_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_tune_param}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.ratio_sweep run \
    --config examples/pointcloud/cfgs/tune_param.yaml \
    --task-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT_DIR"
