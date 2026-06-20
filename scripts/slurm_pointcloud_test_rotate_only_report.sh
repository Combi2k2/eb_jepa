#!/usr/bin/env bash
#SBATCH --job-name=pc_rot_only_report
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_test_rotate_only_report_%j.out
#SBATCH --error=slurm_test_rotate_only_report_%j.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_ROTATE_ONLY_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_new_pointnet_rotation_protocol}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.test_rotate_only \
    collect-test --config examples/pointcloud/cfgs/test_rotate_only.yaml \
    --output-dir "$OUTPUT_DIR"
