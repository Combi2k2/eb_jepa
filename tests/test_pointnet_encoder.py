import pytest
import torch
from omegaconf import OmegaConf

from examples.pointcloud.main import build_encoder


def test_pointnet_output_shape_and_api():
    encoder = build_encoder(OmegaConf.create({
        "backbone": "pointnet", "in_channels": 3, "out_dim": 256,
    }))
    x = torch.randn(4, 3, 128)

    assert encoder.out_dim == 256
    assert encoder(x).shape == (4, 256)
    assert encoder.represent(x).shape == (4, 256)


def test_pointnet_is_permutation_invariant():
    encoder = build_encoder(OmegaConf.create({
        "backbone": "pointnet", "in_channels": 3, "out_dim": 128,
    })).eval()
    x = torch.randn(2, 3, 64)
    permutation = torch.randperm(x.shape[-1])

    with torch.no_grad():
        expected = encoder(x)
        actual = encoder(x[:, :, permutation])

    torch.testing.assert_close(actual, expected)


def test_pointnet_supports_backpropagation():
    encoder = build_encoder(OmegaConf.create({
        "backbone": "pointnet", "in_channels": 3, "out_dim": 64,
    }))
    x = torch.randn(2, 3, 32, requires_grad=True)

    encoder(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pointnet_rejects_invalid_shapes():
    encoder = build_encoder(OmegaConf.create({"in_channels": 3, "out_dim": 64}))

    with pytest.raises(ValueError, match=r"expected point cloud \[B, C, N\]"):
        encoder(torch.randn(3, 32))
    with pytest.raises(ValueError, match="at least one point"):
        encoder(torch.empty(2, 3, 0))


def test_pointnet2_output_shape_and_backpropagation():
    encoder = build_encoder(OmegaConf.create({
        "backbone": "pointnet2", "in_channels": 3, "out_dim": 128,
    }))
    x = torch.randn(2, 3, 64, requires_grad=True)

    output = encoder(x)
    output.square().mean().backward()

    assert output.shape == (2, 128)
    assert encoder.out_dim == 128
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="unknown point-cloud backbone"):
        build_encoder(OmegaConf.create({"backbone": "other", "out_dim": 64}))
