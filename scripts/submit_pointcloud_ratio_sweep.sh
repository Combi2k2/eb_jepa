#!/usr/bin/env bash

set -euo pipefail

REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

train_job=$(sbatch --parsable scripts/slurm_pointcloud_ratio_sweep.sh | cut -d';' -f1)
report_job=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_ratio_report.sh | cut -d';' -f1)

echo "training array job: $train_job"
echo "report job:         $report_job (runs after all array tasks succeed)"
