#!/bin/bash
#SBATCH --job-name=pointnet_rotated_eval
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=pointnet_rotated_eval_%j.out
#SBATCH --error=pointnet_rotated_eval_%j.err

set -e

REPO="$SLURM_SUBMIT_DIR"
source "$REPO/env.sh"
module load python312
cd "$REPO"

: "${POINTNET_CKPT:?POINTNET_CKPT must name a supervised checkpoint}"
TEST_ROTATE="${TEST_ROTATE:-so3}"
TEST_SEED="${TEST_SEED:-0}"

uv sync --dev --project "$REPO"
uv run --project "$REPO" python -m examples.pointcloud.supervised \
    --eval-only --ckpt "$POINTNET_CKPT" \
    --test-rotate "$TEST_ROTATE" --test-seed "$TEST_SEED"
