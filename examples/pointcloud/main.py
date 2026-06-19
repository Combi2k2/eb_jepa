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
def _validate_point_cloud(x):
    if x.ndim != 3:
        raise ValueError(f"expected point cloud [B, C, N], got shape {tuple(x.shape)}")
    if x.shape[-1] == 0:
        raise ValueError("a point cloud must contain at least one point")


class TransformNet(nn.Module):
    """Learn a k x k alignment matrix, initialized around the identity."""

    def __init__(self, k):
        super().__init__()
        self.k = int(k)
        self.convs = nn.Sequential(
            nn.Conv1d(self.k, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.fcs = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, self.k * self.k),
        )
        nn.init.zeros_(self.fcs[-1].weight)
        nn.init.zeros_(self.fcs[-1].bias)

    def forward(self, x):
        features = self.convs(x).amax(dim=-1)
        transform = self.fcs(features).reshape(-1, self.k, self.k)
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0)
        return transform + identity


class PointNetEncoder(nn.Module):
    """Canonical-style PointNet with learned input and feature alignment."""

    def __init__(self, in_channels=3, out_dim=1024, feature_transform=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.input_transform = TransformNet(3)
        self.conv1 = nn.Conv1d(self.in_channels, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.feature_transform = TransformNet(64) if feature_transform else None
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, self.out_dim, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.out_dim)
        self.last_feature_transform = None

    def represent(self, x):
        """Map a point cloud shaped ``[B, C, N]`` to ``[B, out_dim]``."""
        _validate_point_cloud(x)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")

        xyz = x[:, :3, :]
        transform = self.input_transform(xyz)
        xyz = torch.bmm(xyz.transpose(1, 2), transform).transpose(1, 2)
        x = torch.cat((xyz, x[:, 3:, :]), dim=1) if self.in_channels > 3 else xyz
        x = torch.relu(self.bn1(self.conv1(x)))

        if self.feature_transform is not None:
            self.last_feature_transform = self.feature_transform(x)
            x = torch.bmm(x.transpose(1, 2), self.last_feature_transform).transpose(1, 2)
        else:
            self.last_feature_transform = None

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x.amax(dim=-1)

    def transform_regularization(self):
        transform = self.last_feature_transform
        if transform is None:
            return next(self.parameters()).new_zeros(())
        identity = torch.eye(
            transform.shape[-1], device=transform.device, dtype=transform.dtype,
        ).unsqueeze(0)
        residual = torch.bmm(transform, transform.transpose(1, 2)) - identity
        return residual.square().sum(dim=(1, 2)).mean()

    def forward(self, x):
        return self.represent(x)


def _square_distance(src, dst):
    return (
        src.square().sum(dim=-1, keepdim=True)
        + dst.square().sum(dim=-1).unsqueeze(1)
        - 2 * torch.matmul(src, dst.transpose(1, 2))
    ).clamp_min_(0)


def _index_points(points, index):
    batch_shape = (points.shape[0],) + (1,) * (index.ndim - 1)
    batch = torch.arange(points.shape[0], device=points.device).reshape(batch_shape)
    return points[batch.expand_as(index), index]


def _farthest_point_sample(xyz, npoint):
    """Deterministic batched farthest-point sampling over ``[B, N, 3]``."""
    batch_size, n_points, _ = xyz.shape
    npoint = min(int(npoint), n_points)
    centroids = torch.empty(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distances = torch.full((batch_size, n_points), float("inf"), device=xyz.device)
    center = xyz.mean(dim=1, keepdim=True)
    farthest = ((xyz - center) ** 2).sum(dim=-1).argmax(dim=1)
    batch = torch.arange(batch_size, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distance = ((xyz - centroid) ** 2).sum(dim=-1)
        distances = torch.minimum(distances, distance)
        farthest = distances.argmax(dim=1)
    return centroids


def _query_ball_point(radius, nsample, xyz, centroids):
    distances = _square_distance(centroids, xyz)
    k = min(int(nsample), xyz.shape[1])
    masked = distances.masked_fill(distances > radius * radius, float("inf"))
    values, indices = masked.topk(k, dim=-1, largest=False, sorted=False)
    nearest = distances.argmin(dim=-1, keepdim=True)
    return torch.where(torch.isinf(values), nearest.expand_as(indices), indices)


def _sample_and_group(npoint, radius, nsample, xyz, points):
    centroid_indices = _farthest_point_sample(xyz, npoint)
    new_xyz = _index_points(xyz, centroid_indices)
    group_indices = _query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = _index_points(xyz, group_indices) - new_xyz.unsqueeze(2)
    if points is None:
        new_points = grouped_xyz
    else:
        new_points = torch.cat((grouped_xyz, _index_points(points, group_indices)), dim=-1)
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet++ single-scale grouping and local PointNet aggregation."""

    def __init__(self, npoint, radius, nsample, in_channels, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        channels = (in_channels, *mlp)
        layers = []
        for input_dim, output_dim in zip(channels[:-1], channels[1:]):
            layers.extend((
                nn.Conv2d(input_dim, output_dim, 1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        xyz = xyz.transpose(1, 2)
        points = None if points is None else points.transpose(1, 2)
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)
            new_points = grouped_xyz if points is None else torch.cat(
                (grouped_xyz, points.unsqueeze(1)), dim=-1,
            )
        else:
            new_xyz, new_points = _sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points,
            )
        new_points = self.mlp(new_points.permute(0, 3, 2, 1))
        new_points = new_points.amax(dim=2)
        return new_xyz.transpose(1, 2), new_points


class PointNet2Encoder(nn.Module):
    """PointNet++ SSG encoder with hierarchical local neighborhood aggregation."""

    def __init__(self, in_channels=3, out_dim=1024):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, self.in_channels, (64, 64, 128))
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, (128, 128, 256))
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3,
                                          (256, 512, self.out_dim), group_all=True)

    def represent(self, x):
        _validate_point_cloud(x)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        xyz = x[:, :3, :]
        points = x[:, 3:, :] if self.in_channels > 3 else None
        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        _, points = self.sa3(xyz, points)
        return points.squeeze(-1)

    def forward(self, x):
        return self.represent(x)


def build_encoder(cfg):
    """Build the selected ``pointnet`` or ``pointnet2`` global encoder."""
    backbone = cfg.get("backbone", "pointnet").lower()
    common = {"in_channels": cfg.get("in_channels", 3), "out_dim": cfg.out_dim}
    if backbone == "pointnet":
        return PointNetEncoder(
            **common, feature_transform=cfg.get("feature_transform", True),
        )
    if backbone == "pointnet2":
        return PointNet2Encoder(**common)
    raise ValueError(
        f"unknown point-cloud backbone {backbone!r}; expected 'pointnet' or 'pointnet2'"
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`,
    where `batch = (v1, v2, label)` are the two augmented views (label unused for
    SSL).

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
