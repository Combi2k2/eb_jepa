#!/bin/bash

name="sweep_methods"
time="0-23:59:50"
script_link="scripts/sweep_methods.sh"

partition="defq"
reservation="Vivatech"

mkdir -p ./output_terminal/${partition}

sbatch --job-name=${name} \
       --output=./output_terminal/${partition}/${partition}_${name}_%j.out \
       --error=./output_terminal/${partition}/${partition}_${name}_%j.err \
       --time=${time} \
       --partition=${partition} \
       --reservation=${reservation} \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=8 \
       --gres=gpu:1 \
       --wrap="set -e; set -x; \
                REPO=\${EBJEPA_REPO:-\${SLURM_SUBMIT_DIR:-\$(pwd)}}; \
                cd \$REPO; \
                source \$REPO/env.sh; \
                module load python312; \
                export UV_CACHE_DIR=/lustre/work/vivatech-ipparis/lnguyen/shared/.cache/uv; \
                export PIP_CACHE_DIR=/lustre/work/vivatech-ipparis/lnguyen/shared/.cache/pip; \
                export UV_PROJECT_ENVIRONMENT=\$REPO/.venv; \
                mkdir -p \$UV_CACHE_DIR \$PIP_CACHE_DIR; \
                uv sync --dev --project \$REPO; \
                uv run --project \$REPO bash \$REPO/${script_link}"