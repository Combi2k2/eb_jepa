"""PointCloud — SSL pretraining entrypoint (view-invariant 3D shape SSL).

Research question: can a two-view SSL objective learn a VIEW-INVARIANT shape
representation on an unordered/irregular modality (point clouds), and how does the
linear-probe accuracy degrade as we demand more rotation invariance (none -> z ->
SO(3))?

Point clouds have no temporal frames, so the objective is a two-view VICReg (the
image-JEPA / audio / EEG recipe), NOT a predictive JEPA. Two independent augmented
samplings + rotations of the same object are the two views.

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked `# TODO` below — that is the whole point of the track:
  1. the PointNet encoder over [B, 3, N]
  2. the two-view VICReg objective

Run:  python -m examples.pointcloud.main --fname examples/pointcloud/cfgs/train.yaml
"""
import os
import sys

import torch
from torch import nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP from a '256-512-128'-style spec string)
#   eb_jepa.losses:        VICRegLoss (invariance + variance + covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
class PointNetEncoder(nn.Module):
    """Encode an unordered point cloud into one global feature vector."""

    def __init__(self, in_channels=3, out_dim=1024):
        super().__init__()
        self.out_dim = int(out_dim)
        channels = (int(in_channels), 64, 64, 128, self.out_dim)
        layers = []
        for in_dim, out_dim in zip(channels[:-1], channels[1:]):
            layers.extend((
                nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ))
        self.point_mlp = nn.Sequential(*layers)

    def represent(self, x):
        """Map a point cloud shaped ``[B, C, N]`` to ``[B, out_dim]``."""
        if x.ndim != 3:
            raise ValueError(f"expected point cloud [B, C, N], got shape {tuple(x.shape)}")
        if x.shape[-1] == 0:
            raise ValueError("a point cloud must contain at least one point")
        point_features = self.point_mlp(x)
        return point_features.amax(dim=-1)

    def forward(self, x):
        return self.represent(x)


def build_encoder(cfg):
    """Return a PointNet encoder mapping a point cloud [B, 3, N] to a global
    representation [B, D]. Expose `.represent(x) -> [B, D]` (the frozen-feature API
    eval.py calls) and an `.out_dim` attribute.

    Hints: a shared per-point MLP of 1x1 Conv1d layers (3 -> 64 -> 64 -> 128 ->
    out_dim, each Conv1d + BatchNorm1d + ReLU) followed by a symmetric max-pool
    over the N points gives a PERMUTATION-INVARIANT global feature (PointNet, Qi
    et al. 2017; no T-Net needed). The max-pool is what makes it order-agnostic;
    rotation invariance, in contrast, has to be LEARNED from the augmented views."""
    return PointNetEncoder(
        in_channels=cfg.get("in_channels", 3),
        out_dim=cfg.out_dim,
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`,
    where `batch = (v1, v2, n1, n2, label)` are the two augmented views plus the
    number of real (non-padded) points per view (label unused for SSL).

    Build a two-view VICReg head:
      v1, v2 -> encoder.represent -> eb_jepa.architectures.Projector ->
      eb_jepa.losses.VICRegLoss(std_coeff, cov_coeff) on the two projections.
    The variance + covariance terms are the anti-collapse ingredient; the
    invariance (MSE) term is what pulls the two views of the same object together
    and makes the representation VIEW-INVARIANT. Return the scalar loss and a logs
    dict (e.g. the VICRegLoss component breakdown)."""
    raise NotImplementedError("TODO: assemble the two-view VICReg objective (see docstring)")


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/pointcloud/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.split = "train"
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[pointcloud] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/pointcloud/cfgs/train.yaml"
    run(fname=fname)
