"""Supervised-data ratio sweep for ModelNet40 PointNet representations.

For every (augmentation, supervised ratio) pair this experiment compares:
  1. a PointNet classifier trained from scratch on the supervised partition;
  2. VICReg pretraining on the disjoint pretrain partition, followed by a frozen
     PointNet linear probe trained on exactly the same supervised partition.

All partitions are stratified and deterministic from ``split_seed``. Validation
is derived from the official training split; the official test split remains
clean and is touched only for final model evaluation.
"""

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector
from eb_jepa.datasets.pointcloud.dataset import (
    PointCloudConfig,
    PointCloudDataset,
    PointCloudIndexedDataset,
    PointCloudRotatedTestDataset,
    seed_worker,
    stratified_train_partitions,
)
from eb_jepa.losses import VICRegLoss
from eb_jepa.training_utils import setup_wandb


class TransformNet(nn.Module):
    """PointNet T-Net matching yanx27's STN3d/STNkd architecture."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.conv1 = nn.Conv1d(self.channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.channels * self.channels)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, points):
        features = torch.relu(self.bn1(self.conv1(points)))
        features = torch.relu(self.bn2(self.conv2(features)))
        features = torch.relu(self.bn3(self.conv3(features)))
        features = features.amax(dim=-1)
        features = torch.relu(self.bn4(self.fc1(features)))
        features = torch.relu(self.bn5(self.fc2(features)))
        transform = self.fc3(features).reshape(-1, self.channels, self.channels)
        identity = torch.eye(
            self.channels, device=points.device, dtype=points.dtype
        ).unsqueeze(0)
        return transform + identity


class YanxPointNetEncoder(nn.Module):
    """yanx27 PointNet encoder, exposing only its global 1024-D feature."""

    def __init__(self, in_channels=3, out_dim=1024, feature_transform=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        if self.in_channels != 3:
            raise ValueError("yanx27 PointNet input STN currently requires XYZ input")
        self.input_transform = TransformNet(self.in_channels)
        self.conv1 = nn.Conv1d(self.in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.out_dim)
        self.feature_transform = TransformNet(64) if feature_transform else None
        self.last_feature_transform = None

    def represent(self, point_cloud):
        transform = self.input_transform(point_cloud)
        features = torch.bmm(point_cloud.transpose(1, 2), transform).transpose(1, 2)
        features = torch.relu(self.bn1(self.conv1(features)))
        if self.feature_transform is not None:
            self.last_feature_transform = self.feature_transform(features)
            features = torch.bmm(
                features.transpose(1, 2), self.last_feature_transform
            ).transpose(1, 2)
        else:
            self.last_feature_transform = None
        features = torch.relu(self.bn2(self.conv2(features)))
        features = self.bn3(self.conv3(features))
        return features.amax(dim=-1)

    def transform_regularization(self):
        transform = self.last_feature_transform
        if transform is None:
            return next(self.parameters()).new_zeros(())
        identity = torch.eye(
            transform.shape[-1], device=transform.device, dtype=transform.dtype
        ).unsqueeze(0)
        residual = torch.bmm(transform, transform.transpose(1, 2)) - identity
        return torch.linalg.matrix_norm(residual).mean()

    def forward(self, point_cloud):
        return self.represent(point_cloud)


class SimplePointNetEncoder(nn.Module):
    """Original repository PointNet: shared MLP followed by global max pooling."""

    def __init__(self, in_channels=3, out_dim=1024):
        super().__init__()
        self.out_dim = int(out_dim)
        channels = (int(in_channels), 64, 64, 128, self.out_dim)
        layers = []
        for input_dim, output_dim in zip(channels[:-1], channels[1:]):
            layers.extend(
                (
                    nn.Conv1d(input_dim, output_dim, 1, bias=False),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.point_mlp = nn.Sequential(*layers)

    def represent(self, point_cloud):
        return self.point_mlp(point_cloud).amax(dim=-1)

    def transform_regularization(self):
        return next(self.parameters()).new_zeros(())

    def forward(self, point_cloud):
        return self.represent(point_cloud)


def build_pointnet_encoder(model_cfg):
    backbone = str(model_cfg.get("backbone", "simple"))
    if backbone == "simple":
        return SimplePointNetEncoder(model_cfg.in_channels, model_cfg.out_dim)
    if backbone == "yanx27":
        return YanxPointNetEncoder(
            model_cfg.in_channels,
            model_cfg.out_dim,
            feature_transform=model_cfg.get("feature_transform", True),
        )
    raise ValueError(f"unsupported PointNet backbone: {backbone!r}")


class PointNetClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, int(n_classes))

    def forward(self, point_cloud):
        return self.head(self.encoder.represent(point_cloud))


class PointCloudVICReg(nn.Module):
    def __init__(
        self, encoder, projector_spec, std_coeff, cov_coeff, transform_reg_weight=0.0
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(projector_spec)
        self.loss_fn = VICRegLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
        self.transform_reg_weight = float(transform_reg_weight)

    def compute_loss(self, batch):
        view1, view2 = batch[:2]
        projection1 = self.projector(self.encoder.represent(view1))
        regularization1 = self.encoder.transform_regularization()
        projection2 = self.projector(self.encoder.represent(view2))
        regularization2 = self.encoder.transform_regularization()
        components = self.loss_fn(projection1, projection2)
        transform_loss = 0.5 * (regularization1 + regularization2)
        components["transform_reg_loss"] = transform_loss
        components["loss"] = (
            components["loss"] + self.transform_reg_weight * transform_loss
        )
        return components


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, num_workers, shuffle, seed, drop_last=False):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=bool(drop_last),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def accuracy_from_logits(logits, labels):
    return int((logits.argmax(dim=1) == labels).sum().item())


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for point_cloud, labels in loader:
        point_cloud = point_cloud.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(point_cloud)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += accuracy_from_logits(logits, labels)
        total += len(labels)
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate_probe(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for point_cloud, labels in loader:
        point_cloud = point_cloud.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = head(encoder.represent(point_cloud))
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += accuracy_from_logits(logits, labels)
        total += len(labels)
    return {"loss": total_loss / total, "acc": correct / total}


def train_scratch(encoder, train_loader, val_loader, test_loader, cfg, device, run):
    model = PointNetClassifier(encoder, cfg.data.n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.supervised_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None

    for epoch in range(int(cfg.optim.supervised_epochs)):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for point_cloud, labels in train_loader:
            point_cloud = point_cloud.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(point_cloud)
            loss = criterion(logits, labels) + float(
                cfg.model.get("transform_reg_weight", 0.001)
            ) * model.encoder.transform_regularization()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += accuracy_from_logits(logits, labels)
            total += len(labels)

        validation = evaluate_model(model, val_loader, device)
        if validation["acc"] > best_val:
            best_val = validation["acc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if run is not None:
            run.log(
                {
                    "scratch/epoch": epoch,
                    "scratch/train_loss": loss_sum / total,
                    "scratch/train_acc": correct / total,
                    "scratch/val_loss": validation["loss"],
                    "scratch/val_acc": validation["acc"],
                }
            )

    model.load_state_dict(best_state)
    test = evaluate_model(model, test_loader, device)
    return best_val, test["acc"], model


def train_vicreg(encoder, loader, cfg, device, run):
    projector_spec = cfg.model.get(
        "projector", f"{encoder.out_dim}-{2 * encoder.out_dim}-{2 * encoder.out_dim}"
    )
    model = PointCloudVICReg(
        encoder,
        projector_spec,
        cfg.model.std_coeff,
        cfg.model.cov_coeff,
        cfg.model.get("transform_reg_weight", 0.001),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.ssl_lr, weight_decay=cfg.optim.weight_decay
    )
    for epoch in range(int(cfg.optim.ssl_epochs)):
        model.train()
        totals = {
            "loss": 0.0,
            "invariance_loss": 0.0,
            "var_loss": 0.0,
            "cov_loss": 0.0,
            "transform_reg_loss": 0.0,
        }
        samples = 0
        for batch in loader:
            batch = [item.to(device, non_blocking=True) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            components = model.compute_loss(batch)
            components["loss"].backward()
            optimizer.step()
            batch_size = len(batch[0])
            samples += batch_size
            for key in totals:
                totals[key] += components[key].detach().item() * batch_size
        if run is not None:
            run.log(
                {f"ssl/{key}": value / samples for key, value in totals.items()}
                | {"ssl/epoch": epoch}
            )
    return encoder


def train_linear_probe(
    encoder, train_loader, val_loader, test_loader, cfg, device, run
):
    encoder.eval()
    encoder.requires_grad_(False)
    head = nn.Linear(encoder.out_dim, int(cfg.data.n_classes)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.optim.probe_lr, weight_decay=cfg.optim.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None

    for epoch in range(int(cfg.optim.probe_epochs)):
        head.train()
        correct = total = 0
        loss_sum = 0.0
        for point_cloud, labels in train_loader:
            point_cloud = point_cloud.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                features = encoder.represent(point_cloud)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += accuracy_from_logits(logits, labels)
            total += len(labels)

        validation = evaluate_probe(encoder, head, val_loader, device)
        if validation["acc"] > best_val:
            best_val = validation["acc"]
            best_state = copy.deepcopy(head.state_dict())
        if run is not None:
            run.log(
                {
                    "probe/epoch": epoch,
                    "probe/train_loss": loss_sum / total,
                    "probe/train_acc": correct / total,
                    "probe/val_loss": validation["loss"],
                    "probe/val_acc": validation["acc"],
                }
            )

    head.load_state_dict(best_state)
    test = evaluate_probe(encoder, head, test_loader, device)
    return best_val, test["acc"], head


def train_finetune(
    encoder,
    train_loader,
    val_loader,
    test_loader,
    cfg,
    device,
    run,
    encoder_lr,
    head_lr,
    log_prefix,
):
    """Jointly optimize a pretrained encoder and a fresh classification head."""
    encoder.requires_grad_(True)
    model = PointNetClassifier(encoder, cfg.data.n_classes).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": float(encoder_lr)},
            {"params": model.head.parameters(), "lr": float(head_lr)},
        ],
        weight_decay=cfg.optim.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None

    for epoch in range(int(cfg.optim.finetune_epochs)):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for point_cloud, labels in train_loader:
            point_cloud = point_cloud.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(point_cloud)
            loss = criterion(logits, labels) + float(
                cfg.model.get("transform_reg_weight", 0.001)
            ) * model.encoder.transform_regularization()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += accuracy_from_logits(logits, labels)
            total += len(labels)

        validation = evaluate_model(model, val_loader, device)
        if validation["acc"] > best_val:
            best_val = validation["acc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if run is not None:
            run.log(
                {
                    f"{log_prefix}/epoch": epoch,
                    f"{log_prefix}/train_loss": loss_sum / total,
                    f"{log_prefix}/train_acc": correct / total,
                    f"{log_prefix}/val_loss": validation["loss"],
                    f"{log_prefix}/val_acc": validation["acc"],
                }
            )

    model.load_state_dict(best_state)
    test = evaluate_model(model, test_loader, device)
    return best_val, test["acc"], model


def assert_class_coverage(labels, partitions, n_classes):
    expected = set(range(int(n_classes)))
    for name in ("supervised_train", "supervised_val"):
        indices = getattr(partitions, name)
        present = set(np.unique(labels[indices]).tolist())
        if present != expected:
            raise RuntimeError(
                f"{name} is missing classes: {sorted(expected - present)}"
            )


def build_datasets(cfg, ratio, augmentation):
    ssl_augmentation = cfg.sweep.get("ssl_augmentation", augmentation)
    supervised_augmentation = cfg.sweep.get("supervised_augmentation", augmentation)
    data_cfg = PointCloudConfig(
        data_root=cfg.data.data_root,
        split="train",
        mode="supervised",
        n_classes=cfg.data.n_classes,
        n_points=cfg.data.n_points,
        rotate=ssl_augmentation,
        jitter=cfg.data.jitter,
        scale_lo=cfg.data.scale_lo,
        scale_hi=cfg.data.scale_hi,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    base_train = PointCloudDataset(data_cfg)
    partitions = stratified_train_partitions(
        base_train.label, ratio, cfg.sweep.val_ratio, cfg.sweep.split_seed
    )
    assert_class_coverage(base_train.label, partitions, cfg.data.n_classes)

    datasets = {
        "pretrain": PointCloudIndexedDataset(
            base_train,
            partitions.pretrain,
            "ssl",
            ssl_augmentation,
            seed=cfg.sweep.split_seed,
        ),
        "supervised_train": PointCloudIndexedDataset(
            base_train,
            partitions.supervised_train,
            "supervised",
            supervised_augmentation,
            seed=cfg.sweep.split_seed,
            rotation_only=cfg.sweep.get("supervised_rotation_only", False),
        ),
        "supervised_val": PointCloudIndexedDataset(
            base_train,
            partitions.supervised_val,
            "supervised",
            supervised_augmentation,
            seed=cfg.sweep.split_seed,
            deterministic_augmentation=True,
            rotation_only=cfg.sweep.get("supervised_rotation_only", False),
        ),
    }
    test_cfg = copy.deepcopy(data_cfg)
    test_cfg.split = "test"
    test_cfg.mode = "supervised"
    clean_test = PointCloudDataset(test_cfg)
    if cfg.sweep.get("test_matches_supervised", False):
        datasets["test"] = PointCloudRotatedTestDataset(
            clean_test, supervised_augmentation, seed=cfg.sweep.split_seed
        )
    else:
        datasets["test"] = clean_test
    return datasets, partitions, base_train.label


def run_task(cfg, task_id, output_dir):
    ratios = [float(value) for value in cfg.sweep.supervised_ratios]
    augmentations = list(cfg.sweep.augmentations)
    combinations = [
        (augmentation, ratio) for augmentation in augmentations for ratio in ratios
    ]
    if not 0 <= task_id < len(combinations):
        raise ValueError(f"task_id must be in [0, {len(combinations) - 1}]")
    augmentation, ratio = combinations[task_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    run_dir = output_dir / "runs" / f"aug_{augmentation}_ratio_{ratio:g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets, partitions, labels = build_datasets(cfg, ratio, augmentation)
    loader_args = (cfg.data.batch_size, cfg.data.num_workers)
    pretrain_loader = make_loader(
        datasets["pretrain"], *loader_args, True, cfg.meta.seed + 10, drop_last=True
    )
    scratch_train_loader = make_loader(
        datasets["supervised_train"], *loader_args, True, cfg.meta.seed + 20
    )
    probe_train_loader = make_loader(
        datasets["supervised_train"], *loader_args, True, cfg.meta.seed + 20
    )
    val_loader = make_loader(
        datasets["supervised_val"], *loader_args, False, cfg.meta.seed + 40
    )
    test_loader = make_loader(datasets["test"], *loader_args, False, cfg.meta.seed + 50)

    run_config = OmegaConf.to_container(cfg, resolve=True) | {
        "task_id": task_id,
        "augmentation": augmentation,
        "supervised_ratio": ratio,
        "split_fingerprint": partitions.fingerprint(),
    }
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config=run_config,
        run_dir=run_dir,
        run_name=f"pointcloud_{augmentation}_sup{ratio:g}_seed{cfg.meta.seed}",
        tags=["pointcloud", "ratio-sweep", augmentation],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )

    # Reset before each branch so both start from the same encoder initialization.
    set_seed(cfg.meta.seed)
    scratch_encoder = build_pointnet_encoder(cfg.model)
    scratch_val, scratch_test, scratch_model = train_scratch(
        scratch_encoder,
        scratch_train_loader,
        val_loader,
        test_loader,
        cfg,
        device,
        wandb_run,
    )

    set_seed(cfg.meta.seed)
    pretrained_encoder = build_pointnet_encoder(cfg.model).to(device)
    pretrained_encoder = train_vicreg(
        pretrained_encoder, pretrain_loader, cfg, device, wandb_run
    )
    pretrained_state = {
        key: value.detach().cpu().clone()
        for key, value in pretrained_encoder.state_dict().items()
    }
    probe_val, probe_test, probe_head = train_linear_probe(
        pretrained_encoder,
        probe_train_loader,
        val_loader,
        test_loader,
        cfg,
        device,
        wandb_run,
    )
    finetune_results = {}
    finetuned_models = {}
    for variant_name, variant_cfg in cfg.optim.finetune_variants.items():
        set_seed(cfg.meta.seed)
        finetune_encoder = build_pointnet_encoder(cfg.model)
        finetune_encoder.load_state_dict(pretrained_state)
        finetune_train_loader = make_loader(
            datasets["supervised_train"], *loader_args, True, cfg.meta.seed + 20
        )
        variant_val, variant_test, variant_model = train_finetune(
            finetune_encoder,
            finetune_train_loader,
            val_loader,
            test_loader,
            cfg,
            device,
            wandb_run,
            encoder_lr=variant_cfg.encoder_lr,
            head_lr=variant_cfg.head_lr,
            log_prefix=f"finetune_{variant_name}",
        )
        finetune_results.update(
            {
                f"finetune_{variant_name}_val_acc": variant_val,
                f"finetune_{variant_name}_test_acc": variant_test,
                f"finetune_{variant_name}_test_gain": variant_test - scratch_test,
            }
        )
        finetuned_models[variant_name] = variant_model.state_dict()

    result = {
        "augmentation": augmentation,
        "supervised_ratio": ratio,
        "pretrain_size": len(partitions.pretrain),
        "supervised_train_size": len(partitions.supervised_train),
        "supervised_val_size": len(partitions.supervised_val),
        "split_seed": int(cfg.sweep.split_seed),
        "split_fingerprint": partitions.fingerprint(),
        "scratch_val_acc": scratch_val,
        "scratch_test_acc": scratch_test,
        "pretrained_probe_val_acc": probe_val,
        "pretrained_probe_test_acc": probe_test,
        "test_gain": probe_test - scratch_test,
        **finetune_results,
        "train_classes": len(np.unique(labels[partitions.supervised_train])),
        "val_classes": len(np.unique(labels[partitions.supervised_val])),
    }
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"aug_{augmentation}_ratio_{ratio:g}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "result": result,
            "scratch": scratch_model.state_dict(),
            "pretrained_encoder": pretrained_encoder.state_dict(),
            "linear_probe": probe_head.state_dict(),
            "finetuned_models": finetuned_models,
        },
        run_dir / "models.pth.tar",
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                f"final/{key}": value
                for key, value in result.items()
                if isinstance(value, (int, float))
            }
        )
        wandb_run.summary.update(result)
        wandb_run.finish()
    print(json.dumps(result, indent=2), flush=True)


def run_finetune_task(cfg, task_id, output_dir):
    """Add fine-tuning results to an existing completed sweep task."""
    ratios = [float(value) for value in cfg.sweep.supervised_ratios]
    augmentations = list(cfg.sweep.augmentations)
    combinations = [
        (augmentation, ratio) for augmentation in augmentations for ratio in ratios
    ]
    if not 0 <= task_id < len(combinations):
        raise ValueError(f"task_id must be in [0, {len(combinations) - 1}]")
    augmentation, ratio = combinations[task_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    original_run_dir = output_dir / "runs" / f"aug_{augmentation}_ratio_{ratio:g}"
    checkpoint_path = original_run_dir / "models.pth.tar"
    result_path = output_dir / "results" / f"aug_{augmentation}_ratio_{ratio:g}.json"
    if not checkpoint_path.exists() or not result_path.exists():
        raise FileNotFoundError(
            f"existing sweep task is incomplete: {checkpoint_path} / {result_path}"
        )

    datasets, partitions, _ = build_datasets(cfg, ratio, augmentation)
    loader_args = (cfg.data.batch_size, cfg.data.num_workers)
    val_loader = make_loader(
        datasets["supervised_val"], *loader_args, False, cfg.meta.seed + 40
    )
    test_loader = make_loader(datasets["test"], *loader_args, False, cfg.meta.seed + 50)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_dir = output_dir / "finetune_runs" / f"aug_{augmentation}_ratio_{ratio:g}"
    run_config = OmegaConf.to_container(cfg, resolve=True) | {
        "task_id": task_id,
        "augmentation": augmentation,
        "supervised_ratio": ratio,
        "split_fingerprint": partitions.fingerprint(),
        "experiment": "pretrained_finetune",
    }
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config=run_config,
        run_dir=run_dir,
        run_name=f"pointcloud_finetune_{augmentation}_sup{ratio:g}_seed{cfg.meta.seed}",
        tags=["pointcloud", "ratio-sweep", "finetune", augmentation],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )

    result = json.loads(result_path.read_text())
    for legacy_key in (
        "finetune_val_acc",
        "finetune_test_acc",
        "finetune_test_gain",
    ):
        result.pop(legacy_key, None)
    finetuned_models = {}
    for variant_name, variant_cfg in cfg.optim.finetune_variants.items():
        set_seed(cfg.meta.seed)
        variant_encoder = build_pointnet_encoder(cfg.model)
        variant_encoder.load_state_dict(checkpoint["pretrained_encoder"])
        train_loader = make_loader(
            datasets["supervised_train"], *loader_args, True, cfg.meta.seed + 20
        )
        variant_val, variant_test, variant_model = train_finetune(
            variant_encoder,
            train_loader,
            val_loader,
            test_loader,
            cfg,
            device,
            wandb_run,
            encoder_lr=variant_cfg.encoder_lr,
            head_lr=variant_cfg.head_lr,
            log_prefix=f"finetune_{variant_name}",
        )
        result.update(
            {
                f"finetune_{variant_name}_val_acc": variant_val,
                f"finetune_{variant_name}_test_acc": variant_test,
                f"finetune_{variant_name}_test_gain": (
                    variant_test - result["scratch_test_acc"]
                ),
            }
        )
        finetuned_models[variant_name] = variant_model.state_dict()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint.pop("finetuned_model", None)
    checkpoint["finetuned_models"] = finetuned_models
    torch.save(checkpoint, checkpoint_path)
    if wandb_run is not None:
        wandb_run.log(
            {
                f"final/{key}": value
                for key, value in result.items()
                if key.startswith("finetune_") and isinstance(value, (int, float))
            }
        )
        wandb_run.summary.update(result)
        wandb_run.finish()
    print(json.dumps(result, indent=2), flush=True)


def run_argument_test_task(cfg, task_id, output_dir):
    """Evaluate all trained model variants under deterministic test rotations."""
    ratios = [float(value) for value in cfg.sweep.supervised_ratios]
    train_augmentations = list(cfg.sweep.augmentations)
    combinations = [
        (augmentation, ratio)
        for augmentation in train_augmentations
        for ratio in ratios
    ]
    if not 0 <= task_id < len(combinations):
        raise ValueError(f"task_id must be in [0, {len(combinations) - 1}]")
    train_augmentation, ratio = combinations[task_id]
    output_dir = Path(output_dir)
    checkpoint_path = (
        output_dir
        / "runs"
        / f"aug_{train_augmentation}_ratio_{ratio:g}"
        / "models.pth.tar"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing trained models: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finetuned_models = checkpoint.get("finetuned_models", {})
    required_variants = {"equal_lr", "split_lr"}
    if set(finetuned_models) != required_variants:
        raise KeyError(
            f"checkpoint needs fine-tuned variants {sorted(required_variants)}, "
            f"found {sorted(finetuned_models)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch = PointNetClassifier(
        build_pointnet_encoder(cfg.model), cfg.data.n_classes
    ).to(device)
    scratch.load_state_dict(checkpoint["scratch"])
    probe_encoder = build_pointnet_encoder(cfg.model).to(device)
    probe_encoder.load_state_dict(checkpoint["pretrained_encoder"])
    probe_head = nn.Linear(cfg.model.out_dim, cfg.data.n_classes).to(device)
    probe_head.load_state_dict(checkpoint["linear_probe"])
    finetuned = {}
    for variant_name in sorted(required_variants):
        model = PointNetClassifier(
            build_pointnet_encoder(cfg.model),
            cfg.data.n_classes,
        ).to(device)
        model.load_state_dict(finetuned_models[variant_name])
        finetuned[variant_name] = model

    test_cfg = PointCloudConfig(
        data_root=cfg.data.data_root,
        split="test",
        mode="supervised",
        n_classes=cfg.data.n_classes,
        n_points=cfg.data.n_points,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    clean_test = PointCloudDataset(test_cfg)
    rows = []
    for test_augmentation in cfg.sweep.argument_test_augmentations:
        rotated_test = PointCloudRotatedTestDataset(
            clean_test, test_augmentation, seed=cfg.sweep.split_seed
        )
        loader = make_loader(
            rotated_test,
            cfg.data.batch_size,
            cfg.data.num_workers,
            False,
            cfg.meta.seed + 50,
        )
        scratch_acc = evaluate_model(scratch, loader, device)["acc"]
        probe_acc = evaluate_probe(probe_encoder, probe_head, loader, device)["acc"]
        equal_lr_acc = evaluate_model(finetuned["equal_lr"], loader, device)["acc"]
        split_lr_acc = evaluate_model(finetuned["split_lr"], loader, device)["acc"]
        rows.append(
            {
                "train_augmentation": train_augmentation,
                "supervised_ratio": ratio,
                "test_augmentation": str(test_augmentation),
                "test_size": len(rotated_test),
                "scratch_test_acc": scratch_acc,
                "pretrained_probe_test_acc": probe_acc,
                "finetune_equal_lr_test_acc": equal_lr_acc,
                "finetune_split_lr_test_acc": split_lr_acc,
            }
        )

    results_dir = output_dir / "argument_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"train_{train_augmentation}_ratio_{ratio:g}.json"
    result_path.write_text(json.dumps(rows, indent=2) + "\n")

    print(json.dumps(rows, indent=2), flush=True)


def collect_argument_test_results(cfg, output_dir):
    """Collect all rotation-robustness evaluations into one prefixed table."""
    output_dir = Path(output_dir)
    expected = [
        output_dir
        / "argument_test_results"
        / f"train_{augmentation}_ratio_{float(ratio):g}.json"
        for augmentation in cfg.sweep.augmentations
        for ratio in cfg.sweep.supervised_ratios
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("missing argument_test results:\n" + "\n".join(missing))
    rows = []
    for path in expected:
        rows.extend(json.loads(path.read_text()))
    metric_columns = (
        "scratch_test_acc",
        "pretrained_probe_test_acc",
        "finetune_equal_lr_test_acc",
        "finetune_split_lr_test_acc",
    )
    columns = [
        "train_augmentation",
        "supervised_ratio",
        "test_augmentation",
        "test_size",
        "scratch_test_acc",
        "pretrained_probe_test_acc",
        "finetune_equal_lr_test_acc",
        "finetune_split_lr_test_acc",
    ]
    csv_path = output_dir / "results_argument_test.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "results_argument_test.md"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = [
            f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n")

    # Mirror the 12 training runs. Each run logs one final scalar per method,
    # exactly like the original sweep. Its test rotation matches its training
    # augmentation; the full 3x3 cross-evaluation remains available in the table.
    for path in expected:
        task_rows = json.loads(path.read_text())
        train_augmentation = task_rows[0]["train_augmentation"]
        supervised_ratio = task_rows[0]["supervised_ratio"]
        matched_row = next(
            row for row in task_rows if row["test_augmentation"] == train_augmentation
        )
        task_run = setup_wandb(
            project=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True)
            | {
                "train_augmentation": train_augmentation,
                "test_augmentation": train_augmentation,
                "supervised_ratio": supervised_ratio,
                "experiment": "argument_test",
            },
            run_dir=(
                output_dir
                / "wandb_argument_test_runs"
                / f"train_{train_augmentation}_ratio_{supervised_ratio:g}"
            ),
            run_name=(
                f"pointcloud_{train_augmentation}_sup{supervised_ratio:g}"
                f"_seed{cfg.meta.seed}_argument_test"
            ),
            tags=["pointcloud", "argument-test", train_augmentation],
            group=cfg.logging.group,
            enabled=cfg.logging.enabled,
            resume=False,
        )
        if task_run is not None:
            final_metrics = {
                f"final/{metric}_argument_test": matched_row[metric]
                for metric in metric_columns
            }
            task_run.log(final_metrics)
            task_run.summary.update(
                {
                    **matched_row,
                    **final_metrics,
                }
            )
            task_run.finish()

    summary_run = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=output_dir / "wandb_summary_argument_test",
        run_name=f"pointcloud_ratio_sweep_summary_seed{cfg.meta.seed}_argument_test",
        tags=["pointcloud", "argument-test", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary_run is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary_run.log({"results_argument_test": table})
        summary_run.save(str(csv_path), base_path=str(output_dir))
        summary_run.save(str(markdown_path), base_path=str(output_dir))
        summary_run.finish()
    print(markdown_path.read_text(), flush=True)
    print(f"saved {csv_path} and {markdown_path}", flush=True)


def collect_results(cfg, output_dir):
    output_dir = Path(output_dir)
    expected = [
        output_dir / "results" / f"aug_{augmentation}_ratio_{float(ratio):g}.json"
        for augmentation in cfg.sweep.augmentations
        for ratio in cfg.sweep.supervised_ratios
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("missing sweep results:\n" + "\n".join(missing))
    rows = [json.loads(path.read_text()) for path in expected]
    columns = [
        "augmentation",
        "supervised_ratio",
        "pretrain_size",
        "supervised_train_size",
        "supervised_val_size",
        "train_classes",
        "val_classes",
        "split_seed",
        "split_fingerprint",
        "scratch_val_acc",
        "scratch_test_acc",
        "pretrained_probe_val_acc",
        "pretrained_probe_test_acc",
        "test_gain",
        "finetune_equal_lr_val_acc",
        "finetune_equal_lr_test_acc",
        "finetune_equal_lr_test_gain",
        "finetune_split_lr_val_acc",
        "finetune_split_lr_test_acc",
        "finetune_split_lr_test_gain",
    ]
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)

    markdown_path = output_dir / "results.md"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n")

    summary_dir = output_dir / "wandb_summary"
    summary_run = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=summary_dir,
        run_name=f"pointcloud_ratio_sweep_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "ratio-sweep", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary_run is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(key) for key in columns])
        summary_run.log({"results": table})
        summary_run.save(str(csv_path), base_path=str(output_dir))
        summary_run.save(str(markdown_path), base_path=str(output_dir))
        summary_run.finish()
    print(markdown_path.read_text(), flush=True)
    print(f"saved {csv_path} and {markdown_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "run",
            "finetune",
            "argument-test",
            "collect",
            "collect-argument-test",
        ),
    )
    parser.add_argument("--config", default="examples/pointcloud/cfgs/ratio_sweep.yaml")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    destination = args.output_dir or config.meta.output_dir
    if args.command in ("run", "finetune", "argument-test"):
        if args.task_id is None:
            raise ValueError(f"--task-id is required for command {args.command!r}")
        if args.command == "run":
            run_task(config, args.task_id, destination)
        elif args.command == "finetune":
            run_finetune_task(config, args.task_id, destination)
        else:
            run_argument_test_task(config, args.task_id, destination)
    elif args.command == "collect":
        collect_results(config, destination)
    else:
        collect_argument_test_results(config, destination)
