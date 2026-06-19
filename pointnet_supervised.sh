#!/bin/bash
#SBATCH --job-name=pointnet_supervised
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=pointnet_supervised_%j.out
#SBATCH --error=pointnet_supervised_%j.err

set -e

REPO="$SLURM_SUBMIT_DIR"
source "$REPO/env.sh"
module load python312

uv sync --dev --project "$REPO"
uv run --project "$REPO" python -m examples.pointcloud.supervised \
    --fname "$REPO/examples/pointcloud/cfgs/supervised.yaml"
