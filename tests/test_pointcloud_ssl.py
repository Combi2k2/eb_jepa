import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud.main import build_encoder, build_ssl


def test_pointcloud_vicreg_loss_and_gradients():
    cfg = OmegaConf.create({
        "backbone": "pointnet",
        "in_channels": 3,
        "out_dim": 32,
        "feature_transform": False,
        "projector": "32-64-16",
        "std_coeff": 25.0,
        "cov_coeff": 1.0,
    })
    encoder = build_encoder(cfg)
    ssl = build_ssl(encoder, cfg)
    view1 = torch.randn(4, 3, 32)
    view2 = torch.randn(4, 3, 32)
    labels = torch.randint(0, 40, (4,))

    loss, logs = ssl.compute_loss((view1, view2, labels))
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(logs) == {
        "invariance_loss", "var_loss", "cov_loss", "transform_reg_loss",
    }
    assert all(p.grad is not None for p in ssl.projector.parameters())
    assert any(p.grad is not None for p in encoder.parameters())


def test_projector_input_must_match_encoder_output():
    cfg = OmegaConf.create({
        "backbone": "pointnet", "out_dim": 32,
        "feature_transform": False, "projector": "64-128-64",
    })
    with pytest.raises(ValueError, match="does not match"):
        build_ssl(build_encoder(cfg), cfg)


def test_dataset_produces_two_independent_ssl_views():
    dataset = PointCloudDataset.__new__(PointCloudDataset)
    dataset.cfg = PointCloudConfig(n_points=64, rotate="so3")
    cloud = np.random.default_rng(0).normal(size=(256, 3)).astype(np.float32)

    view1 = dataset._augment(cloud, np.random.default_rng(1)).T
    view2 = dataset._augment(cloud, np.random.default_rng(2)).T

    assert view1.shape == (3, 64)
    assert view2.shape == (3, 64)
    assert not np.allclose(view1, view2)
