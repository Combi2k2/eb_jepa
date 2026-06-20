"""Train 10/30 and 30/10 class splits, then rotate-test 10/20/30 splits."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import _rand_rot
from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.new_class import build_datasets, run as run_clean_experiment
from examples.pointcloud.ratio_sweep import (
    PointNetClassifier,
    build_pointnet_encoder,
    evaluate_model,
    evaluate_probe,
    make_loader,
)

METHOD_METRICS = {
    "scratch": "scratch_test_acc",
    "pretrained_probe": "pretrained_probe_test_acc",
    "finetune_equal_lr": "finetune_equal_lr_test_acc",
    "finetune_split_lr": "finetune_split_lr_test_acc",
}
ROTATIONS = ("none", "z", "so3")


class DeterministicRotatedDataset(torch.utils.data.Dataset):
    """Apply one stable rotation to each already-clean test point cloud."""

    def __init__(self, dataset, rotation, seed):
        if rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}")
        self.dataset = dataset
        self.rotation = rotation
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        point_cloud, label = self.dataset[index]
        if self.rotation == "none":
            return point_cloud, label
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        rotation = torch.from_numpy(_rand_rot(rng, self.rotation)).to(point_cloud)
        return rotation @ point_cloud, label


def task_config(cfg, n_pretrain_classes):
    task_cfg = copy.deepcopy(cfg)
    task_cfg.split.n_pretrain_classes = int(n_pretrain_classes)
    task_cfg.data.n_classes = int(task_cfg.data.total_classes) - int(
        n_pretrain_classes
    )
    return task_cfg


def clean_run_dir(output_dir, n_pretrain_classes):
    return Path(output_dir) / "runs" / f"pretrain_{int(n_pretrain_classes)}"


def run_clean_task(cfg, task_id, output_dir):
    counts = [int(value) for value in cfg.sweep.new_pretrain_class_counts]
    if not 0 <= int(task_id) < len(counts):
        raise ValueError(f"task_id must be in [0, {len(counts) - 1}]")
    count = counts[int(task_id)]
    run_clean_experiment(task_config(cfg, count), clean_run_dir(output_dir, count))


def _load_models(cfg, checkpoint, device):
    scratch = PointNetClassifier(
        build_pointnet_encoder(cfg.model), cfg.data.n_classes
    ).to(device)
    scratch.load_state_dict(checkpoint["scratch"])

    probe_encoder = build_pointnet_encoder(cfg.model).to(device)
    probe_encoder.load_state_dict(checkpoint["pretrained_encoder"])
    probe_head = nn.Linear(cfg.model.out_dim, cfg.data.n_classes).to(device)
    probe_head.load_state_dict(checkpoint["linear_probe"])

    finetuned = {}
    for variant in ("equal_lr", "split_lr"):
        model = PointNetClassifier(
            build_pointnet_encoder(cfg.model), cfg.data.n_classes
        ).to(device)
        model.load_state_dict(checkpoint["finetuned_models"][variant])
        finetuned[variant] = model
    return scratch, probe_encoder, probe_head, finetuned


def checkpoint_dir(output_dir, existing_20_dir, n_pretrain_classes):
    if int(n_pretrain_classes) == 20:
        return Path(existing_20_dir)
    return clean_run_dir(output_dir, n_pretrain_classes)


def run_rotation_task(cfg, task_id, output_dir, existing_20_dir):
    counts = [int(value) for value in cfg.sweep.rotation_pretrain_class_counts]
    if not 0 <= int(task_id) < len(counts):
        raise ValueError(f"task_id must be in [0, {len(counts) - 1}]")
    count = counts[int(task_id)]
    current_cfg = task_config(cfg, count)
    model_dir = checkpoint_dir(output_dir, existing_20_dir, count)
    checkpoint_path = model_dir / "models.pth.tar"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    _, _, _, clean_test, metadata = build_datasets(current_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch, probe_encoder, probe_head, finetuned = _load_models(
        current_cfg, checkpoint, device
    )

    rows = []
    for rotation in ROTATIONS:
        dataset = DeterministicRotatedDataset(
            clean_test, rotation, current_cfg.sweep.rotation_seed
        )
        loader = make_loader(
            dataset,
            current_cfg.data.batch_size,
            current_cfg.data.num_workers,
            False,
            current_cfg.meta.seed + 50,
        )
        rows.append(
            {
                "pretrain_classes": count,
                "supervised_classes": int(current_cfg.data.n_classes),
                "test_rotation": rotation,
                "test_size": len(dataset),
                "split_fingerprint": metadata["split_fingerprint"],
                "scratch_test_acc": evaluate_model(
                    scratch, loader, device
                )["acc"],
                "pretrained_probe_test_acc": evaluate_probe(
                    probe_encoder, probe_head, loader, device
                )["acc"],
                "finetune_equal_lr_test_acc": evaluate_model(
                    finetuned["equal_lr"], loader, device
                )["acc"],
                "finetune_split_lr_test_acc": evaluate_model(
                    finetuned["split_lr"], loader, device
                )["acc"],
            }
        )

    result_dir = Path(output_dir) / "rotation_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"pretrain_{count}.json"
    result_path.write_text(json.dumps(rows, indent=2) + "\n")

    run = setup_wandb(
        project=current_cfg.logging.rotation_project,
        config=OmegaConf.to_container(current_cfg, resolve=True) | metadata,
        run_dir=Path(output_dir) / "wandb_rotation" / f"pretrain_{count}",
        run_name=(
            f"pointcloud_pretrain{count}_supervised{current_cfg.data.n_classes}"
            f"_rotation_test_seed{current_cfg.meta.seed}"
        ),
        tags=["pointcloud", "disjoint-class", "rotation-test"],
        group=current_cfg.logging.rotation_group,
        enabled=current_cfg.logging.enabled,
        resume=False,
    )
    if run is not None:
        import wandb

        columns = list(rows[0])
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        metrics = {}
        for row in rows:
            for method, key in METHOD_METRICS.items():
                metrics[f"final/{method}_test_acc_{row['test_rotation']}"] = row[key]
        run.log({"results_rotation_test": table, **metrics})
        run.summary.update(metrics)
        run.finish()
    print(json.dumps(rows, indent=2), flush=True)


def _write_table(rows, output_dir, stem):
    output_dir = Path(output_dir)
    columns = list(rows[0])
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [
            f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n")
    return columns, csv_path, markdown_path


def collect_clean(cfg, output_dir):
    rows = []
    for count in cfg.sweep.new_pretrain_class_counts:
        result = json.loads(
            (clean_run_dir(output_dir, count) / "results.json").read_text()
        )
        rows.append(
            {
                "pretrain_classes": int(count),
                "supervised_classes": int(cfg.data.total_classes) - int(count),
                "pretrain_size": result["pretrain_size"],
                "supervised_train_size": result["supervised_train_size"],
                "supervised_val_size": result["supervised_val_size"],
                "test_size": result["test_size"],
                "split_fingerprint": result["split_fingerprint"],
                **{key: result[key] for key in METHOD_METRICS.values()},
            }
        )
    columns, csv_path, markdown_path = _write_table(
        rows, output_dir, "results_new_class_10_30"
    )
    run = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=Path(output_dir) / "wandb_clean_summary",
        run_name=f"pointcloud_new_class_10_30_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "disjoint-class", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if run is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        run.log({"results_new_class_10_30": table})
        run.save(str(csv_path), base_path=str(output_dir))
        run.save(str(markdown_path), base_path=str(output_dir))
        run.finish()


def collect_rotation(cfg, output_dir):
    rows = []
    for count in cfg.sweep.rotation_pretrain_class_counts:
        rows.extend(
            json.loads(
                (
                    Path(output_dir)
                    / "rotation_results"
                    / f"pretrain_{int(count)}.json"
                ).read_text()
            )
        )
    columns, csv_path, markdown_path = _write_table(
        rows, output_dir, "results_new_class_rotation_test"
    )
    run = setup_wandb(
        project=cfg.logging.rotation_project,
        config=cfg,
        run_dir=Path(output_dir) / "wandb_rotation_summary",
        run_name=f"pointcloud_new_class_rotation_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "disjoint-class", "rotation-test", "summary"],
        group=cfg.logging.rotation_group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if run is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        run.log({"results_new_class_rotation_test": table})
        run.save(str(csv_path), base_path=str(output_dir))
        run.save(str(markdown_path), base_path=str(output_dir))
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("train", "rotate-test", "collect-clean", "collect-rotation")
    )
    parser.add_argument(
        "--config", default="examples/pointcloud/cfgs/new_class_sweep.yaml"
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--existing-20-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    destination = args.output_dir or config.meta.output_dir
    if args.command in ("train", "rotate-test") and args.task_id is None:
        raise ValueError(f"--task-id is required for {args.command}")
    if args.command == "train":
        run_clean_task(config, args.task_id, destination)
    elif args.command == "rotate-test":
        existing = args.existing_20_dir or config.meta.existing_20_dir
        run_rotation_task(config, args.task_id, destination, existing)
    elif args.command == "collect-clean":
        collect_clean(config, destination)
    else:
        collect_rotation(config, destination)
