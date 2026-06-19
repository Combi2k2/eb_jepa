import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud.main import PointNetEncoder
from examples.pointcloud.supervised import (
    PointNetClassifier,
    fixed_train_val_indices,
    run_epoch,
)


def test_supervised_classifier_shape_and_training_step():
    model = PointNetClassifier(PointNetEncoder(out_dim=32), n_classes=4)
    points = torch.randn(8, 3, 16)
    labels = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(points, labels), batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    assert model(points).shape == (8, 4)
    metrics = run_epoch(model, loader, torch.device("cpu"), optimizer)

    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1


def test_supervised_augmentation_is_train_only():
    cloud = np.random.default_rng(0).normal(size=(256, 3)).astype(np.float32)
    dataset = PointCloudDataset.__new__(PointCloudDataset)
    dataset.data = np.expand_dims(cloud, 0)
    dataset.label = np.array([3])
    dataset.cfg = PointCloudConfig(
        split="train", mode="supervised", n_points=64,
        augment_supervised=True, rotate="z", jitter=0.01,
    )

    train_view1, label1 = dataset[0]
    train_view2, label2 = dataset[0]
    assert train_view1.shape == (3, 64)
    assert label1 == label2 == 3
    assert not torch.allclose(train_view1, train_view2)

    dataset.cfg.split = "test"
    test_view1, _ = dataset[0]
    test_view2, _ = dataset[0]
    torch.testing.assert_close(test_view1, test_view2)


def test_fixed_train_validation_split_is_reproducible_and_disjoint():
    train1, val1 = fixed_train_val_indices(100, val_fraction=0.2, split_seed=17)
    train2, val2 = fixed_train_val_indices(100, val_fraction=0.2, split_seed=17)
    train3, val3 = fixed_train_val_indices(100, val_fraction=0.2, split_seed=18)

    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(val1, val2)
    assert len(train1) == 80
    assert len(val1) == 20
    assert set(train1).isdisjoint(val1)
    assert set(train1) | set(val1) == set(range(100))
    assert not np.array_equal(val1, val3)


def test_rotated_test_view_is_fixed_by_seed_and_sample_index():
    clouds = np.random.default_rng(0).normal(size=(2, 256, 3)).astype(np.float32)

    def make_dataset(seed):
        dataset = PointCloudDataset.__new__(PointCloudDataset)
        dataset.data = clouds
        dataset.label = np.array([0, 1])
        dataset.cfg = PointCloudConfig(
            split="test", mode="supervised", n_points=64,
            test_rotate="so3", test_seed=seed,
        )
        return dataset

    dataset1 = make_dataset(23)
    dataset2 = make_dataset(23)
    view1, _ = dataset1[0]
    view2, _ = dataset1[0]
    view3, _ = dataset2[0]
    different_seed_view, _ = make_dataset(24)[0]
    different_sample_view, _ = dataset1[1]

    torch.testing.assert_close(view1, view2)
    torch.testing.assert_close(view1, view3)
    assert not torch.allclose(view1, different_seed_view)
    assert not torch.allclose(view1, different_sample_view)
