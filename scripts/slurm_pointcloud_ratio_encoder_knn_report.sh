#!/usr/bin/env bash
#SBATCH --job-name=pc_ratio_knn_report
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm_pointcloud_ratio_knn_report_%j.out
#SBATCH --error=slurm_pointcloud_ratio_knn_report_%j.err

set -euo pipefail
REPO="${EBJEPA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO"
source "$REPO/env.sh"
module load python312
OUTPUT_DIR="${POINTCLOUD_RATIO_KNN_DIR:-$EBJEPA_CKPTS/pointcloud/eb_jepa_ratio_encoder_knn}"
uv run --no-sync --project "$REPO" python -m examples.pointcloud.ratio_encoder_knn collect \
    --config examples/pointcloud/cfgs/ratio_encoder_knn.yaml --output-dir "$OUTPUT_DIR"
