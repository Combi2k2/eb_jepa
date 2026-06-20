#!/usr/bin/env bash
#SBATCH --job-name=pc_class_match_report
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_pointcloud_new_class_matched_rotation_report_%j.out
#SBATCH --error=slurm_pointcloud_new_class_matched_rotation_report_%j.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_NEW_CLASS_SWEEP_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_new_class_10_30}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.new_class_matched_rotation collect \
    --config examples/pointcloud/cfgs/new_class_sweep.yaml --output-dir "$OUTPUT_DIR"
