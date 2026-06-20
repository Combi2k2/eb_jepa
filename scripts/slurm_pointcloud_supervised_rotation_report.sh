#!/usr/bin/env bash
#SBATCH --job-name=pc_sup_rot_report
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_pointcloud_supervised_rotation_report_%j.out
#SBATCH --error=slurm_pointcloud_supervised_rotation_report_%j.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_SUPERVISED_ROTATION_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_supervised_rotation}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.ratio_sweep collect \
    --config examples/pointcloud/cfgs/supervised_rotation.yaml --output-dir "$OUTPUT_DIR"
