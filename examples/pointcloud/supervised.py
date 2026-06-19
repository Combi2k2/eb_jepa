"""End-to-end supervised PointNet training and evaluation on ModelNet40.

This entrypoint does not use VICReg, a projector, or any other self-supervised
component. It trains the PointNet encoder and a linear classification head
together using class labels and cross-entropy.
"""
import argparse
import os

import torch
from omegaconf import OmegaConf
from torch import nn

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, make_loader
from examples.pointcloud.main import build_encoder


class PointNetClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, int(n_classes))

    def forward(self, x):
        return self.classifier(self.encoder.represent(x))


def build_loader(data_cfg, split):
    cfg = PointCloudConfig(**OmegaConf.to_container(data_cfg, resolve=True))
    cfg.split = split
    cfg.mode = "supervised"
    return make_loader(cfg, shuffle=split == "train")


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for points, labels in loader:
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(points)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(path, model, cfg, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, path)


def train(cfg):
    torch.manual_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(cfg.data, "train")
    encoder = build_encoder(cfg.model)
    model = PointNetClassifier(encoder, cfg.data.n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.optim.epochs,
    )

    for epoch in range(1, cfg.optim.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={100 * train_metrics['accuracy']:.2f}%",
            flush=True,
        )

        latest_path = os.path.join(cfg.meta.ckpt_dir, "latest.pth.tar")
        save_checkpoint(latest_path, model, cfg, epoch)

    return latest_path


def evaluate(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    model = PointNetClassifier(
        build_encoder(cfg.model), cfg.data.n_classes,
    ).to(device)
    model.load_state_dict(state["model"])
    metrics = run_epoch(model, build_loader(cfg.data, "test"), device)
    print(
        f"checkpoint={checkpoint} test_loss={metrics['loss']:.4f} "
        f"test_accuracy={100 * metrics['accuracy']:.2f}%",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", default="examples/pointcloud/cfgs/supervised.yaml",
        help="supervised training configuration",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="evaluate a previously trained checkpoint",
    )
    parser.add_argument("--ckpt", help="checkpoint used with --eval-only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        if not args.ckpt:
            raise ValueError("--ckpt is required with --eval-only")
        evaluate(args.ckpt)
    else:
        checkpoint = train(OmegaConf.load(args.fname))
        evaluate(checkpoint)
