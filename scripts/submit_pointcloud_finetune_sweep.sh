#!/usr/bin/env bash

set -euo pipefail

REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

finetune_job=$(sbatch --parsable scripts/slurm_pointcloud_finetune_sweep.sh | cut -d';' -f1)
report_job=$(sbatch --parsable --dependency=afterok:"$finetune_job" \
    scripts/slurm_pointcloud_ratio_report.sh | cut -d';' -f1)

echo "fine-tuning array job: $finetune_job"
echo "updated report job:    $report_job (runs after all fine-tuning tasks succeed)"
