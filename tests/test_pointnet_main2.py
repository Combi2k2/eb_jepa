import torch
from omegaconf import OmegaConf

import examples.pointcloud.main2 as main2


def test_main2_contains_only_standard_pointnet():
    assert hasattr(main2, "PointNetEncoder")
    assert not hasattr(main2, "PointNet2Encoder")


def test_main2_pointnet_shape_and_permutation_invariance():
    encoder = main2.build_encoder(OmegaConf.create({
        "in_channels": 3, "out_dim": 128, "feature_transform": True,
    })).eval()
    points = torch.randn(2, 3, 64)
    with torch.no_grad():
        output = encoder(points)
        shuffled = encoder(points[:, :, torch.randperm(64)])

    assert output.shape == (2, 128)
    torch.testing.assert_close(output, shuffled)
