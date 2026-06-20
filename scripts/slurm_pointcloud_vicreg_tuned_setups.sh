#!/usr/bin/env bash
#SBATCH --job-name=pc_vicreg_tuned
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:50
#SBATCH --array=0-11%3
#SBATCH --output=slurm_pointcloud_vicreg_tuned_%A_%a.out
#SBATCH --error=slurm_pointcloud_vicreg_tuned_%A_%a.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_VICREG_TUNED_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_vicreg_tuned_ratio_matched}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.ratio_sweep run \
    --config examples/pointcloud/cfgs/vicreg_tuned_ratio_matched.yaml \
    --task-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT_DIR"
