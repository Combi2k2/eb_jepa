#!/usr/bin/env bash

set -euo pipefail
REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
train_job=$(sbatch --parsable scripts/slurm_pointcloud_test_rotate_only_train.sh | cut -d';' -f1)
train_report=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_test_rotate_only_train_report.sh | cut -d';' -f1)
test_job=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_test_rotate_only_eval.sh | cut -d';' -f1)
test_report=$(sbatch --parsable --dependency=afterok:"$test_job" \
    scripts/slurm_pointcloud_test_rotate_only_report.sh | cut -d';' -f1)
echo "training array:      $train_job"
echo "clean report:        $train_report"
echo "rotation-test array: $test_job"
echo "rotation report:     $test_report"
