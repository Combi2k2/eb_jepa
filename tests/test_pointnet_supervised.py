import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.pointcloud.main import PointNetEncoder
from examples.pointcloud.supervised import PointNetClassifier, run_epoch


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
