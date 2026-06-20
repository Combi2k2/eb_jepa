"""Matched supervised-train and test rotation for disjoint-class experiments."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import _rand_rot
from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.new_class import build_datasets
from examples.pointcloud.new_class_sweep import checkpoint_dir, task_config
from examples.pointcloud.ratio_sweep import (
    build_pointnet_encoder,
    make_loader,
    set_seed,
    train_finetune,
    train_linear_probe,
    train_scratch,
)

ROTATIONS = ("none", "z", "so3")
METHOD_KEYS = (
    "scratch_test_acc",
    "pretrained_probe_test_acc",
    "finetune_equal_lr_test_acc",
    "finetune_split_lr_test_acc",
)


class MatchedRotationDataset(torch.utils.data.Dataset):
    """Rotate clean point clouds with a stochastic or deterministic draw."""

    def __init__(self, dataset, rotation, seed, deterministic):
        if rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}")
        self.dataset = dataset
        self.rotation = rotation
        self.seed = int(seed)
        self.deterministic = bool(deterministic)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        point_cloud, label = self.dataset[index]
        if self.rotation == "none":
            return point_cloud, label
        if self.deterministic:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(index)])
            )
        else:
            random_seed = torch.randint(0, 2**31 - 1, (1,)).item()
            rng = np.random.default_rng(random_seed)
        rotation = torch.from_numpy(_rand_rot(rng, self.rotation)).to(point_cloud)
        return rotation @ point_cloud, label


def run_task(cfg, task_id, output_dir, existing_20_dir):
    combinations = [
        (int(count), rotation)
        for count in cfg.sweep.pretrain_class_counts
        for rotation in ROTATIONS
    ]
    if not 0 <= int(task_id) < len(combinations):
        raise ValueError(f"task_id must be in [0, {len(combinations) - 1}]")
    count, rotation = combinations[int(task_id)]
    current_cfg = task_config(cfg, count)
    model_dir = checkpoint_dir(output_dir, existing_20_dir, count)
    checkpoint_path = model_dir / "models.pth.tar"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained_state = checkpoint["pretrained_encoder"]

    _, clean_train, clean_val, clean_test, metadata = build_datasets(current_cfg)
    train_dataset = MatchedRotationDataset(
        clean_train, rotation, current_cfg.sweep.rotation_seed, deterministic=False
    )
    val_dataset = MatchedRotationDataset(
        clean_val, rotation, current_cfg.sweep.rotation_seed, deterministic=True
    )
    test_dataset = MatchedRotationDataset(
        clean_test, rotation, current_cfg.sweep.rotation_seed, deterministic=True
    )
    loader_args = (current_cfg.data.batch_size, current_cfg.data.num_workers)
    scratch_loader = make_loader(
        train_dataset, *loader_args, True, current_cfg.meta.seed + 20
    )
    probe_loader = make_loader(
        train_dataset, *loader_args, True, current_cfg.meta.seed + 20
    )
    val_loader = make_loader(
        val_dataset, *loader_args, False, current_cfg.meta.seed + 40
    )
    test_loader = make_loader(
        test_dataset, *loader_args, False, current_cfg.meta.seed + 50
    )

    task_dir = (
        Path(output_dir)
        / "matched_rotation_runs"
        / f"pretrain_{count}_rotation_{rotation}"
    )
    run = setup_wandb(
        project=current_cfg.logging.matched_rotation_project,
        config=OmegaConf.to_container(current_cfg, resolve=True)
        | metadata
        | {
            "train_rotation": rotation,
            "test_rotation": rotation,
            "pretrained_checkpoint": str(checkpoint_path),
        },
        run_dir=task_dir / "wandb",
        run_name=(
            f"pointcloud_pretrain{count}_supervised{current_cfg.data.n_classes}"
            f"_train-{rotation}_test-{rotation}_seed{current_cfg.meta.seed}"
        ),
        tags=["pointcloud", "disjoint-class", "matched-rotation", rotation],
        group=current_cfg.logging.matched_rotation_group,
        enabled=current_cfg.logging.enabled,
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
        run,
    )

    set_seed(current_cfg.meta.seed)
    probe_encoder = build_pointnet_encoder(current_cfg.model).to(device)
    probe_encoder.load_state_dict(pretrained_state)
    probe_val, probe_test, probe_head = train_linear_probe(
        probe_encoder,
        probe_loader,
        val_loader,
        test_loader,
        current_cfg,
        device,
        run,
    )

    finetune_results = {}
    finetuned_models = {}
    for variant_name, variant_cfg in current_cfg.optim.finetune_variants.items():
        set_seed(current_cfg.meta.seed)
        encoder = build_pointnet_encoder(current_cfg.model)
        encoder.load_state_dict(pretrained_state)
        train_loader = make_loader(
            train_dataset, *loader_args, True, current_cfg.meta.seed + 20
        )
        val_acc, test_acc, model = train_finetune(
            encoder,
            train_loader,
            val_loader,
            test_loader,
            current_cfg,
            device,
            run,
            encoder_lr=variant_cfg.encoder_lr,
            head_lr=variant_cfg.head_lr,
            log_prefix=f"finetune_{variant_name}",
        )
        finetune_results[f"finetune_{variant_name}_val_acc"] = val_acc
        finetune_results[f"finetune_{variant_name}_test_acc"] = test_acc
        finetuned_models[variant_name] = model.state_dict()

    result = metadata | {
        "pretrain_classes": count,
        "supervised_classes": int(current_cfg.data.n_classes),
        "ssl_rotation": "so3",
        "supervised_train_rotation": rotation,
        "supervised_val_rotation": rotation,
        "test_rotation": rotation,
        "scratch_val_acc": scratch_val,
        "scratch_test_acc": scratch_test,
        "pretrained_probe_val_acc": probe_val,
        "pretrained_probe_test_acc": probe_test,
        **finetune_results,
    }
    results_dir = Path(output_dir) / "matched_rotation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"pretrain_{count}_rotation_{rotation}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "result": result,
            "scratch": scratch_model.state_dict(),
            "pretrained_encoder": pretrained_state,
            "linear_probe": probe_head.state_dict(),
            "finetuned_models": finetuned_models,
        },
        task_dir / "models.pth.tar",
    )
    if run is not None:
        final_metrics = {
            f"final/{key}": result[key]
            for key in result
            if key.endswith("_acc")
        }
        run.log(final_metrics)
        run.summary.update(result | final_metrics)
        run.finish()
    print(json.dumps(result, indent=2), flush=True)


def collect_results(cfg, output_dir):
    output_dir = Path(output_dir)
    rows = []
    for count in cfg.sweep.pretrain_class_counts:
        for rotation in ROTATIONS:
            path = (
                output_dir
                / "matched_rotation_results"
                / f"pretrain_{int(count)}_rotation_{rotation}.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"missing result: {path}")
            rows.append(json.loads(path.read_text()))
    columns = [
        "pretrain_classes",
        "supervised_classes",
        "pretrain_size",
        "supervised_train_size",
        "supervised_val_size",
        "test_size",
        "split_fingerprint",
        "ssl_rotation",
        "supervised_train_rotation",
        "supervised_val_rotation",
        "test_rotation",
        "scratch_test_acc",
        "pretrained_probe_test_acc",
        "finetune_equal_lr_test_acc",
        "finetune_split_lr_test_acc",
    ]
    csv_path = output_dir / "results_new_class_matched_rotation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)

    summary = setup_wandb(
        project=cfg.logging.matched_rotation_project,
        config=cfg,
        run_dir=output_dir / "wandb_matched_rotation_summary",
        run_name=f"pointcloud_new_class_matched_rotation_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "disjoint-class", "matched-rotation", "summary"],
        group=cfg.logging.matched_rotation_group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary.log({"results_new_class_matched_rotation": table})
        summary.save(str(csv_path), base_path=str(output_dir))
        summary.finish()
    print(csv_path.read_text(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "collect"))
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
    if args.command == "run":
        if args.task_id is None:
            raise ValueError("--task-id is required for run")
        existing = args.existing_20_dir or config.meta.existing_20_dir
        run_task(config, args.task_id, destination, existing)
    else:
        collect_results(config, destination)
