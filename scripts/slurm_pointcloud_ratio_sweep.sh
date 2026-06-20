#!/usr/bin/env bash
#SBATCH --job-name=pc_ratio_sweep
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:50
#SBATCH --array=0-11%3
#SBATCH --output=slurm_pointcloud_ratio_%A_%a.out
#SBATCH --error=slurm_pointcloud_ratio_%A_%a.err

set -euo pipefail

REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312

echo "=== host=$(hostname) task=$SLURM_ARRAY_TASK_ID date=$(date) ==="
echo "=== gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a) ==="

if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi

# Array tasks share one architecture-specific environment. Serialize sync to
# avoid concurrent writes, then run each experiment independently.
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
exec 9>"${UV_PROJECT_ENVIRONMENT}.sync.lock"
flock 9
uv sync --project "$REPO"
flock -u 9

OUTPUT_DIR="${POINTCLOUD_SWEEP_DIR:-$EBJEPA_CKPTS/pointcloud/ratio_sweep}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.ratio_sweep run \
    --config examples/pointcloud/cfgs/ratio_sweep.yaml \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUTPUT_DIR"
