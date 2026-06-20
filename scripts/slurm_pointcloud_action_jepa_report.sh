#!/usr/bin/env bash
#SBATCH --job-name=pc_action_report
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_pointcloud_action_jepa_report_%j.out
#SBATCH --error=slurm_pointcloud_action_jepa_report_%j.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_ACTION_JEPA_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_action_predictor}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.action_jepa collect \
    --config examples/pointcloud/cfgs/action_jepa.yaml --output-dir "$OUTPUT_DIR"
