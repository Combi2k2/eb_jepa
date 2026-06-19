"""Point-cloud SSL entrypoint containing only the standard PointNet backbone.

PointNet maps ``[B, 3, N]`` to ``[B, out_dim]`` using input/feature transform
networks, shared 1x1 convolutions, and symmetric max pooling. PointNet++ is
intentionally not included in this file.
"""
import os
import sys

import torch
from omegaconf import OmegaConf
from torch import nn

from eb_jepa.architectures import Projector
from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, make_loader
from eb_jepa.losses import VICRegLoss


class TransformNet(nn.Module):
    """Learn a k x k PointNet alignment matrix around the identity."""

    def __init__(self, k):
        super().__init__()
        self.k = int(k)
        self.convs = nn.Sequential(
            nn.Conv1d(self.k, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
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
    """Standard PointNet encoder with input and feature T-Nets."""

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
        if x.ndim != 3 or x.shape[-1] == 0:
            raise ValueError(f"expected non-empty point cloud [B, C, N], got {tuple(x.shape)}")
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


def build_encoder(cfg):
    """Build only standard PointNet; no backbone selection is performed."""
    return PointNetEncoder(
        in_channels=cfg.get("in_channels", 3),
        out_dim=cfg.out_dim,
        feature_transform=cfg.get("feature_transform", True),
    )


class PointCloudVICReg(nn.Module):
    def __init__(self, encoder, projector_spec, std_coeff, cov_coeff,
                 transform_reg_weight=0.0):
        super().__init__()
        dimensions = tuple(map(int, projector_spec.split("-")))
        if dimensions[0] != encoder.out_dim:
            raise ValueError("projector input dimension must match encoder.out_dim")
        self.encoder = encoder
        self.projector = Projector(projector_spec)
        self.loss_fn = VICRegLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
        self.transform_reg_weight = float(transform_reg_weight)

    def _encode(self, view):
        projection = self.projector(self.encoder.represent(view))
        regularization = self.encoder.transform_regularization()
        return projection, regularization

    def compute_loss(self, batch):
        view1, view2 = batch[:2]
        projection1, regularization1 = self._encode(view1)
        projection2, regularization2 = self._encode(view2)
        components = self.loss_fn(projection1, projection2)
        transform_loss = 0.5 * (regularization1 + regularization2)
        loss = components["loss"] + self.transform_reg_weight * transform_loss
        logs = {
            "invariance_loss": components["invariance_loss"].detach().item(),
            "var_loss": components["var_loss"].detach().item(),
            "cov_loss": components["cov_loss"].detach().item(),
            "transform_reg_loss": transform_loss.detach().item(),
        }
        return loss, logs

    def forward(self, batch):
        return self.compute_loss(batch)


def build_ssl(encoder, cfg):
    projector_spec = cfg.get(
        "projector", f"{encoder.out_dim}-{2 * encoder.out_dim}-{2 * encoder.out_dim}",
    )
    return PointCloudVICReg(
        encoder,
        projector_spec=projector_spec,
        std_coeff=cfg.get("std_coeff", 25.0),
        cov_coeff=cfg.get("cov_coeff", 1.0),
        transform_reg_weight=cfg.get("transform_reg_weight", 0.0),
    )


def run(fname="examples/pointcloud/cfgs/train.yaml"):
    cfg = OmegaConf.load(fname)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    data_cfg = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg.split = "train"
    data_cfg.mode = "ssl"
    loader = make_loader(data_cfg)
    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
    )

    os.makedirs(cfg.meta.ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward()
            optimizer.step()
        print(f"[pointnet] epoch={epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save(
            {"epoch": epoch, "encoder": encoder.state_dict(),
             "cfg": OmegaConf.to_container(cfg, resolve=True)},
            os.path.join(cfg.meta.ckpt_dir, "latest.pth.tar"),
        )


if __name__ == "__main__":
    config = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv else \
        "examples/pointcloud/cfgs/train.yaml"
    run(config)
