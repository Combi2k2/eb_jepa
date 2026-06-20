#!/usr/bin/env bash
set -euo pipefail
REPO="${EBJEPA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
job=$(sbatch --parsable scripts/slurm_pointcloud_new_class.sh | cut -d';' -f1)
echo "new-class experiment: $job"
