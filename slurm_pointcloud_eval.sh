#!/bin/bash
#SBATCH --job-name=pointcloud_eval
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=output_terminal/pointcloud/pointcloud_eval_%j.out
#SBATCH --error=output_terminal/pointcloud/pointcloud_eval_%j.err

set -euo pipefail

REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"

CHECKPOINT="${POINTCLOUD_CHECKPOINT:-$REPO/checkpoints/pointcloud/dev/latest.pth.tar}"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

echo "=== Host: $(hostname) | Arch: $ARCH | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== Checkpoint: $CHECKPOINT ==="

module load python312

if ! uv --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi

echo ">>> Syncing dependencies..."
uv sync --dev --project "$REPO"

echo ">>> Starting PointCloud evaluation..."
uv run --project "$REPO" python -m examples.pointcloud.eval --ckpt "$CHECKPOINT"

echo "=== Done ==="
