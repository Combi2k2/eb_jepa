#!/usr/bin/env bash
set -euo pipefail
REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
train_job=$(sbatch --parsable scripts/slurm_pointcloud_new_loss.sh | cut -d';' -f1)
report_job=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_new_loss_report.sh | cut -d';' -f1)
echo "new-loss array: $train_job"
echo "summary report: $report_job"
