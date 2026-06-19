#!/bin/bash
#SBATCH --job-name=pointnet_ssl_probe
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=pointnet_ssl_probe_%j.out
#SBATCH --error=pointnet_ssl_probe_%j.err

set -e

REPO="$SLURM_SUBMIT_DIR"
source "$REPO/env.sh"
module load python312
cd "$REPO"

uv sync --dev --project "$REPO"

uv run --project "$REPO" python -m examples.pointcloud.main \
    --fname "$REPO/examples/pointcloud/cfgs/train.yaml"

uv run --project "$REPO" python -m examples.pointcloud.eval \
    --ckpt "$REPO/checkpoints/pointcloud/ssl_pointnet/latest.pth.tar"
