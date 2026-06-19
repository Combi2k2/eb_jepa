#!/bin/bash
#SBATCH --job-name=pointcloud_backbones
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=pointcloud_backbone_test_%j.out
#SBATCH --error=pointcloud_backbone_test_%j.err

set -e

REPO="$SLURM_SUBMIT_DIR"
source "$REPO/env.sh"
module load python312
uv sync --dev --project "$REPO"

uv run --project "$REPO" python - <<'PY'
import torch
from omegaconf import OmegaConf

from examples.pointcloud.main import build_encoder


device = torch.device("cuda")
for backbone in ("pointnet", "pointnet2"):
    encoder = build_encoder(OmegaConf.create({
        "backbone": backbone,
        "in_channels": 3,
        "out_dim": 1024,
        "feature_transform": True,
    })).to(device)
    points = torch.randn(2, 3, 1024, device=device, requires_grad=True)
    output = encoder(points)
    output.square().mean().backward()

    assert output.shape == (2, 1024)
    assert points.grad is not None and torch.isfinite(points.grad).all()
    print(
        f"backbone={backbone} input={tuple(points.shape)} "
        f"output={tuple(output.shape)} forward=PASS backward=PASS"
    )
PY
