import numpy as np

from examples.pointcloud.eval import probe


def test_linear_probe_separates_frozen_features():
    rng = np.random.default_rng(0)
    centers = np.eye(3, dtype=np.float32)
    train_labels = np.repeat(np.arange(3), 20)
    test_labels = np.repeat(np.arange(3), 10)
    train_features = centers[train_labels] + 0.05 * rng.normal(size=(60, 3))
    test_features = centers[test_labels] + 0.05 * rng.normal(size=(30, 3))

    metrics = probe(train_features, train_labels, test_features, test_labels, 3)

    assert metrics["accuracy"] > 95.0
    assert metrics["chance"] == 100.0 / 3
