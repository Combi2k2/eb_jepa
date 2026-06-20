#!/usr/bin/env bash
set -euo pipefail
REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
train_job=$(sbatch --parsable scripts/slurm_pointcloud_new_class_sweep_train.sh | cut -d';' -f1)
clean_report=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_new_class_sweep_clean_report.sh | cut -d';' -f1)
rotation_job=$(sbatch --parsable --dependency=afterok:"$train_job" \
    scripts/slurm_pointcloud_new_class_rotation_test.sh | cut -d';' -f1)
rotation_report=$(sbatch --parsable --dependency=afterok:"$rotation_job" \
    scripts/slurm_pointcloud_new_class_rotation_report.sh | cut -d';' -f1)
echo "clean training array: $train_job"
echo "clean report:         $clean_report"
echo "rotation-test array:  $rotation_job"
echo "rotation report:      $rotation_report"
