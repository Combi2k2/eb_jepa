#!/bin/bash
#SBATCH --job-name=pointcloud_ssl
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:50
#SBATCH --output=output_terminal/pointcloud/pointcloud_ssl_%j.out
#SBATCH --error=output_terminal/pointcloud/pointcloud_ssl_%j.err

set -euo pipefail

REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
export POINTCLOUD_CKPT_DIR="${POINTCLOUD_CKPT_DIR:-$REPO/checkpoints/pointcloud/dev}"

echo "=== Host: $(hostname) | Arch: $ARCH | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== Checkpoints: $POINTCLOUD_CKPT_DIR ==="

module load python312

if ! uv --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi

echo ">>> Syncing dependencies..."
uv sync --dev --project "$REPO"

echo ">>> Starting PointCloud SSL training..."
bash "$REPO/scripts/train_pointcloud.sh"

echo "=== Done ==="
