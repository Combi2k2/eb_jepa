#!/usr/bin/env bash
set -euo pipefail
REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
train_job=$(sbatch --parsable scripts/slurm_pointcloud_action_jepa.sh | cut -d';' -f1)
report_job=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_action_jepa_report.sh | cut -d';' -f1)
echo "action-JEPA array: $train_job"
echo "summary report:    $report_job"
