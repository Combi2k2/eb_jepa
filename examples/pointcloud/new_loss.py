"""Compare sliced-Wasserstein and CKA objectives for point-cloud SSL."""

import argparse
import copy
import csv
import hashlib
import json
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
)
from eb_jepa.losses import (
    CenteredKernelAlignmentLoss,
    CovarianceLoss,
    HingeStdLoss,
    SlicedWassersteinLoss,
)
from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.new_class import (
    RemappedLabelsDataset,
    TestClassSubset,
    build_datasets as build_disjoint_datasets,
    split_supervised_indices,
)
from examples.pointcloud.new_class_matched_rotation import MatchedRotationDataset
from examples.pointcloud.new_class_sweep import task_config
from examples.pointcloud.ratio_sweep import (
    build_pointnet_encoder,
    make_loader,
    set_seed,
    train_finetune,
    train_linear_probe,
    train_scratch,
)

LOSS_METHODS = ("sliced_wasserstein", "cka")


class PointCloudAlternativeSSL(nn.Module):
    """Primary distribution alignment plus VICReg-style anti-collapse terms."""

    def __init__(self, encoder, cfg, method):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(cfg.model.projector)
        if method == "sliced_wasserstein":
            self.primary_loss = SlicedWassersteinLoss(
                cfg.loss.wasserstein_projections
            )
        elif method == "cka":
            self.primary_loss = CenteredKernelAlignmentLoss()
        else:
            raise ValueError(f"unsupported loss method: {method}")
        self.std_loss = HingeStdLoss(std_margin=1.0)
        self.cov_loss = CovarianceLoss()
        self.primary_coeff = float(cfg.loss.primary_coeff)
        self.std_coeff = float(cfg.loss.std_coeff)
        self.cov_coeff = float(cfg.loss.cov_coeff)

    def compute_loss(self, batch):
        view1, view2 = batch[:2]
        projection1 = self.projector(self.encoder.represent(view1))
        projection2 = self.projector(self.encoder.represent(view2))
        primary = self.primary_loss(projection1, projection2)
        variance = self.std_loss(projection1) + self.std_loss(projection2)
        covariance = self.cov_loss(projection1) + self.cov_loss(projection2)
        total = (
            self.primary_coeff * primary
            + self.std_coeff * variance
            + self.cov_coeff * covariance
        )
        return {
            "loss": total,
            "primary_loss": primary,
            "var_loss": variance,
            "cov_loss": covariance,
        }


def train_alternative_ssl(encoder, loader, cfg, method, device, run):
    model = PointCloudAlternativeSSL(encoder, cfg, method).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.ssl_lr, weight_decay=cfg.optim.weight_decay
    )
    for epoch in range(int(cfg.optim.ssl_epochs)):
        model.train()
        totals = {"loss": 0.0, "primary_loss": 0.0, "var_loss": 0.0, "cov_loss": 0.0}
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


def _all_class_datasets(cfg):
    data_cfg = PointCloudConfig(
        data_root=cfg.data.data_root,
        split="train",
        mode="supervised",
        n_classes=40,
        n_points=cfg.data.n_points,
        rotate="so3",
        jitter=cfg.data.jitter,
        scale_lo=cfg.data.scale_lo,
        scale_hi=cfg.data.scale_hi,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    base_train = PointCloudDataset(data_cfg)
    all_indices = np.arange(len(base_train), dtype=np.int64)
    train_indices, val_indices = split_supervised_indices(
        base_train.label, np.arange(40), cfg.split.val_ratio, cfg.split.sample_seed
    )
    identity_map = {class_id: class_id for class_id in range(40)}
    pretrain = PointCloudIndexedDataset(
        base_train,
        all_indices,
        mode="ssl",
        augmentation="so3",
        seed=cfg.split.sample_seed,
    )
    supervised_train = RemappedLabelsDataset(
        PointCloudIndexedDataset(
            base_train, train_indices, "supervised", "none", seed=cfg.split.sample_seed
        ),
        identity_map,
    )
    supervised_val = RemappedLabelsDataset(
        PointCloudIndexedDataset(
            base_train,
            val_indices,
            "supervised",
            "none",
            seed=cfg.split.sample_seed,
            deterministic_augmentation=True,
        ),
        identity_map,
    )
    test_cfg = copy.deepcopy(data_cfg)
    test_cfg.split = "test"
    clean_test = TestClassSubset(
        PointCloudDataset(test_cfg), np.arange(40), identity_map
    )
    digest = hashlib.sha256()
    for values in (all_indices, train_indices, val_indices):
        digest.update(values.tobytes())
    metadata = {
        "class_setup": "all40",
        "pretrain_classes": 40,
        "supervised_classes": 40,
        "pretrain_size": len(pretrain),
        "supervised_train_size": len(supervised_train),
        "supervised_val_size": len(supervised_val),
        "test_size": len(clean_test),
        "split_fingerprint": digest.hexdigest()[:16],
        "pretrain_supervised_overlap": True,
    }
    return pretrain, supervised_train, supervised_val, clean_test, metadata


def build_setup_datasets(cfg, pretrain_classes):
    if pretrain_classes == "all40":
        current_cfg = copy.deepcopy(cfg)
        current_cfg.data.n_classes = 40
        datasets = _all_class_datasets(current_cfg)
    else:
        current_cfg = task_config(cfg, int(pretrain_classes))
        pretrain, train, val, test, metadata = build_disjoint_datasets(current_cfg)
        metadata["class_setup"] = f"{int(pretrain_classes)}_{current_cfg.data.n_classes}"
        metadata["pretrain_supervised_overlap"] = False
        datasets = pretrain, train, val, test, metadata
    pretrain, clean_train, clean_val, clean_test, metadata = datasets
    rotation = str(cfg.sweep.supervised_test_rotation)
    train = MatchedRotationDataset(
        clean_train, rotation, cfg.sweep.rotation_seed, deterministic=False
    )
    val = MatchedRotationDataset(
        clean_val, rotation, cfg.sweep.rotation_seed, deterministic=True
    )
    test = MatchedRotationDataset(
        clean_test, rotation, cfg.sweep.rotation_seed, deterministic=True
    )
    return current_cfg, pretrain, train, val, test, metadata


def run_task(cfg, task_id, output_dir):
    combinations = [
        (setup, method)
        for setup in cfg.sweep.pretrain_class_setups
        for method in LOSS_METHODS
    ]
    if not 0 <= int(task_id) < len(combinations):
        raise ValueError(f"task_id must be in [0, {len(combinations) - 1}]")
    setup, method = combinations[int(task_id)]
    setup = str(setup)
    setup_value = setup if setup == "all40" else int(setup)
    current_cfg, pretrain, train, val, test, metadata = build_setup_datasets(
        cfg, setup_value
    )
    loader_args = (current_cfg.data.batch_size, current_cfg.data.num_workers)
    pretrain_loader = make_loader(
        pretrain, *loader_args, True, current_cfg.meta.seed + 10, drop_last=True
    )
    scratch_loader = make_loader(train, *loader_args, True, current_cfg.meta.seed + 20)
    probe_loader = make_loader(train, *loader_args, True, current_cfg.meta.seed + 20)
    val_loader = make_loader(val, *loader_args, False, current_cfg.meta.seed + 40)
    test_loader = make_loader(test, *loader_args, False, current_cfg.meta.seed + 50)
    run_dir = Path(output_dir) / "runs" / f"setup_{metadata['class_setup']}_loss_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config=OmegaConf.to_container(current_cfg, resolve=True)
        | metadata
        | {"ssl_loss_method": method},
        run_dir=run_dir / "wandb",
        run_name=f"pointcloud_{metadata['class_setup']}_{method}_seed{cfg.meta.seed}",
        tags=["pointcloud", "new-loss", method, metadata["class_setup"]],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(current_cfg.meta.seed)
    scratch_val, scratch_test, scratch_model = train_scratch(
        build_pointnet_encoder(current_cfg.model),
        scratch_loader,
        val_loader,
        test_loader,
        current_cfg,
        device,
        wandb_run,
    )
    set_seed(current_cfg.meta.seed)
    pretrained_encoder = train_alternative_ssl(
        build_pointnet_encoder(current_cfg.model).to(device),
        pretrain_loader,
        current_cfg,
        method,
        device,
        wandb_run,
    )
    pretrained_state = {
        key: value.detach().cpu().clone()
        for key, value in pretrained_encoder.state_dict().items()
    }
    probe_val, probe_test, probe_head = train_linear_probe(
        pretrained_encoder,
        probe_loader,
        val_loader,
        test_loader,
        current_cfg,
        device,
        wandb_run,
    )
    finetune_results = {}
    finetuned_models = {}
    for variant_name, variant_cfg in current_cfg.optim.finetune_variants.items():
        set_seed(current_cfg.meta.seed)
        encoder = build_pointnet_encoder(current_cfg.model)
        encoder.load_state_dict(pretrained_state)
        train_loader = make_loader(
            train, *loader_args, True, current_cfg.meta.seed + 20
        )
        val_acc, test_acc, model = train_finetune(
            encoder,
            train_loader,
            val_loader,
            test_loader,
            current_cfg,
            device,
            wandb_run,
            encoder_lr=variant_cfg.encoder_lr,
            head_lr=variant_cfg.head_lr,
            log_prefix=f"finetune_{variant_name}",
        )
        finetune_results[f"finetune_{variant_name}_val_acc"] = val_acc
        finetune_results[f"finetune_{variant_name}_test_acc"] = test_acc
        finetuned_models[variant_name] = model.state_dict()

    result = metadata | {
        "ssl_loss_method": method,
        "ssl_augmentation": "so3",
        "supervised_test_augmentation": cfg.sweep.supervised_test_rotation,
        "scratch_val_acc": scratch_val,
        "scratch_test_acc": scratch_test,
        "pretrained_probe_val_acc": probe_val,
        "pretrained_probe_test_acc": probe_test,
        **finetune_results,
    }
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"setup_{metadata['class_setup']}_loss_{method}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "result": result,
            "scratch": scratch_model.state_dict(),
            "pretrained_encoder": pretrained_state,
            "linear_probe": probe_head.state_dict(),
            "finetuned_models": finetuned_models,
        },
        run_dir / "models.pth.tar",
    )
    if wandb_run is not None:
        final_metrics = {
            f"final/{key}": value
            for key, value in result.items()
            if key.endswith("_acc")
        }
        wandb_run.log(final_metrics)
        wandb_run.summary.update(result | final_metrics)
        wandb_run.finish()


def collect_results(cfg, output_dir):
    output_dir = Path(output_dir)
    rows = []
    for setup in cfg.sweep.pretrain_class_setups:
        setup_name = "all40" if str(setup) == "all40" else f"{int(setup)}_{40-int(setup)}"
        for method in LOSS_METHODS:
            path = output_dir / "results" / f"setup_{setup_name}_loss_{method}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing result: {path}")
            rows.append(json.loads(path.read_text()))
    columns = [
        "class_setup",
        "pretrain_classes",
        "supervised_classes",
        "pretrain_supervised_overlap",
        "pretrain_size",
        "supervised_train_size",
        "supervised_val_size",
        "test_size",
        "split_fingerprint",
        "ssl_loss_method",
        "ssl_augmentation",
        "supervised_test_augmentation",
        "scratch_test_acc",
        "pretrained_probe_test_acc",
        "finetune_equal_lr_test_acc",
        "finetune_split_lr_test_acc",
    ]
    csv_path = output_dir / "results_new_loss.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)
    summary = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=output_dir / "wandb_summary",
        run_name=f"pointcloud_new_loss_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "new-loss", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary.log({"results_new_loss": table})
        summary.save(str(csv_path), base_path=str(output_dir))
        summary.finish()
    print(csv_path.read_text(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--config", default="examples/pointcloud/cfgs/new_loss.yaml")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    destination = args.output_dir or config.meta.output_dir
    if args.command == "run":
        if args.task_id is None:
            raise ValueError("--task-id is required for run")
        run_task(config, args.task_id, destination)
    else:
        collect_results(config, destination)
