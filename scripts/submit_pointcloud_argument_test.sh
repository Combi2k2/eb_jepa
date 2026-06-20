#!/usr/bin/env bash

set -euo pipefail

REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

dependency_args=()
if [[ $# -gt 0 && -n "$1" ]]; then
    dependency_args=(--dependency="afterok:$1")
fi

evaluation_job=$(sbatch --parsable "${dependency_args[@]}" \
    scripts/slurm_pointcloud_argument_test.sh | cut -d';' -f1)
report_job=$(sbatch --parsable --dependency=afterok:"$evaluation_job" \
    scripts/slurm_pointcloud_argument_test_report.sh | cut -d';' -f1)

echo "argument-test array job: $evaluation_job"
echo "argument-test report:    $report_job"
