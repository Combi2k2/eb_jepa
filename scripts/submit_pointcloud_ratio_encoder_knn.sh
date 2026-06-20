#!/usr/bin/env bash
set -euo pipefail
REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
knn_job=$(sbatch --parsable scripts/slurm_pointcloud_ratio_encoder_knn.sh | cut -d';' -f1)
report_job=$(sbatch --parsable --dependency=afterok:"$knn_job" \
    scripts/slurm_pointcloud_ratio_encoder_knn_report.sh | cut -d';' -f1)
echo "ratio encoder k-NN array: $knn_job"
echo "summary report:           $report_job"
