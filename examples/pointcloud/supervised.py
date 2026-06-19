"""End-to-end supervised PointNet training and evaluation on ModelNet40.

This entrypoint does not use VICReg, a projector, or any other self-supervised
component. It trains the PointNet encoder and a linear classification head
together using class labels and cross-entropy.
"""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Subset

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud.main import build_encoder


class PointNetClassifier(nn.Module):
    def __init__(self, encoder, n_classes, transform_reg_weight=1.0e-3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, int(n_classes))
        self.transform_reg_weight = float(transform_reg_weight)

    def forward(self, x):
        return self.classifier(self.encoder.represent(x))


def fixed_train_val_indices(n_items, val_fraction=0.2, split_seed=0):
    """Return reproducible, disjoint train/validation indices."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    indices = np.random.default_rng(split_seed).permutation(n_items)
    n_val = int(round(n_items * val_fraction))
    val_indices = np.sort(indices[:n_val])
    train_indices = np.sort(indices[n_val:])
    return train_indices, val_indices


def build_loader(data_cfg, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}; expected train, val, or test")
    cfg = PointCloudConfig(**OmegaConf.to_container(data_cfg, resolve=True))
    cfg.split = "train" if split in {"train", "val"} else "test"
    cfg.mode = "supervised"
    if split != "train":
        cfg.augment_supervised = False
    dataset = PointCloudDataset(cfg)
    if split in {"train", "val"}:
        train_indices, val_indices = fixed_train_val_indices(
            len(dataset), cfg.val_fraction, cfg.split_seed,
        )
        selected = train_indices if split == "train" else val_indices
        dataset = Subset(dataset, selected.tolist())
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=split == "train",
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


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
            if training and hasattr(model.encoder, "transform_regularization"):
                loss = loss + model.transform_reg_weight * model.encoder.transform_regularization()
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
    val_loader = build_loader(cfg.data, "val")
    encoder = build_encoder(cfg.model)
    model = PointNetClassifier(
        encoder, cfg.data.n_classes,
        transform_reg_weight=cfg.model.get("transform_reg_weight", 1.0e-3),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.optim.epochs,
    )

    best_accuracy = -1.0
    best_path = os.path.join(cfg.meta.ckpt_dir, "best.pth.tar")
    for epoch in range(1, cfg.optim.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        scheduler.step()
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={100 * train_metrics['accuracy']:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={100 * val_metrics['accuracy']:.2f}%",
            flush=True,
        )

        latest_path = os.path.join(cfg.meta.ckpt_dir, "latest.pth.tar")
        save_checkpoint(latest_path, model, cfg, epoch)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(best_path, model, cfg, epoch)

    print(f"best_val_accuracy={100 * best_accuracy:.2f}%", flush=True)
    return best_path


def evaluate(checkpoint, test_rotate=None, test_seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    if test_rotate is not None:
        cfg.data.test_rotate = test_rotate
    if test_seed is not None:
        cfg.data.test_seed = test_seed
    model = PointNetClassifier(
        build_encoder(cfg.model), cfg.data.n_classes,
        transform_reg_weight=cfg.model.get("transform_reg_weight", 1.0e-3),
    ).to(device)
    model.load_state_dict(state["model"])
    metrics = run_epoch(model, build_loader(cfg.data, "test"), device)
    print(
        f"checkpoint={checkpoint} test_rotate={cfg.data.get('test_rotate', 'none')} "
        f"test_seed={cfg.data.get('test_seed', 0)} test_loss={metrics['loss']:.4f} "
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
    parser.add_argument(
        "--test-rotate", choices=("none", "z", "so3"),
        help="override checkpoint test rotation during evaluation",
    )
    parser.add_argument(
        "--test-seed", type=int,
        help="override checkpoint seed for deterministic test rotations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        if not args.ckpt:
            raise ValueError("--ckpt is required with --eval-only")
        evaluate(args.ckpt, test_rotate=args.test_rotate, test_seed=args.test_seed)
    else:
        checkpoint = train(OmegaConf.load(args.fname))
        evaluate(checkpoint)
