#!/usr/bin/env bash
#SBATCH --job-name=pc_rot_only_train
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:50
#SBATCH --array=0-3%3
#SBATCH --output=slurm_test_rotate_only_train_%A_%a.out
#SBATCH --error=slurm_test_rotate_only_train_%A_%a.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
exec 9>"${UV_PROJECT_ENVIRONMENT}.sync.lock"
flock 9
uv sync --project "$REPO"
flock -u 9
OUTPUT_DIR="${POINTCLOUD_ROTATE_ONLY_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_new_pointnet_rotation_protocol}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.test_rotate_only train \
    --config examples/pointcloud/cfgs/test_rotate_only.yaml \
    --task-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT_DIR"
