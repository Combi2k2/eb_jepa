"""No-rotation training sweep with deterministic rotation-only test evaluation."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import (
    PointCloudConfig,
    PointCloudDataset,
    PointCloudRotatedTestDataset,
)
from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.ratio_sweep import (
    PointNetClassifier,
    build_pointnet_encoder,
    evaluate_model,
    evaluate_probe,
    make_loader,
    run_task,
)

TEST_METHODS = (("none", "none"), ("z", "rotation"), ("so3", "SO3"))
MODEL_METRICS = (
    "scratch_test_acc",
    "pretrained_probe_test_acc",
    "finetune_equal_lr_test_acc",
    "finetune_split_lr_test_acc",
)


def run_train_task(cfg, task_id, output_dir):
    if cfg.sweep.get("ssl_augmentation") not in ("z", "so3"):
        raise ValueError("SSL augmentation must use z or so3 rotation")
    if cfg.sweep.get("supervised_augmentation") != "none":
        raise ValueError("supervised train/validation must use clean views")
    run_task(cfg, task_id, output_dir)


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
    for variant_name in ("equal_lr", "split_lr"):
        model = PointNetClassifier(
            build_pointnet_encoder(cfg.model),
            cfg.data.n_classes,
        ).to(device)
        model.load_state_dict(checkpoint["finetuned_models"][variant_name])
        finetuned[variant_name] = model
    return scratch, probe_encoder, probe_head, finetuned


def run_test_task(cfg, task_id, output_dir):
    ratios = [float(value) for value in cfg.sweep.supervised_ratios]
    if not 0 <= task_id < len(ratios):
        raise ValueError(f"task_id must be in [0, {len(ratios) - 1}]")
    ratio = ratios[task_id]
    output_dir = Path(output_dir)
    ssl_augmentation = str(cfg.sweep.ssl_augmentation)
    checkpoint_path = (
        output_dir
        / "runs"
        / f"aug_{ssl_augmentation}_ratio_{ratio:g}"
        / "models.pth.tar"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing trained models: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(checkpoint.get("finetuned_models", {})) != {"equal_lr", "split_lr"}:
        raise KeyError("checkpoint does not contain both fine-tuning variants")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scratch, probe_encoder, probe_head, finetuned = _load_models(
        cfg, checkpoint, device
    )
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
    for rotation, method_suffix in TEST_METHODS:
        test_dataset = PointCloudRotatedTestDataset(
            clean_test, rotation, seed=cfg.sweep.split_seed
        )
        loader = make_loader(
            test_dataset,
            cfg.data.batch_size,
            cfg.data.num_workers,
            False,
            cfg.meta.seed + 50,
        )
        row = {
            "supervised_ratio": ratio,
            "train_rotation": "none",
            "ssl_rotation": ssl_augmentation,
            "test_rotation": rotation,
            "test_method": method_suffix,
            "test_size": len(test_dataset),
            "scratch_test_acc": evaluate_model(scratch, loader, device)["acc"],
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
        rows.append(row)

        suffix = f"test_rotate_only_{method_suffix}"
        run = setup_wandb(
            project=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True)
            | {
                "supervised_ratio": ratio,
                "train_rotation": "none",
                "ssl_rotation": ssl_augmentation,
                "test_rotation": rotation,
                "test_method": method_suffix,
                "experiment": "test_rotate_only",
            },
            run_dir=(
                output_dir
                / "wandb_test_rotate_only_runs"
                / f"ratio_{ratio:g}_{method_suffix}"
            ),
            run_name=(
                f"pointcloud_ssl-{ssl_augmentation}_supervised-none_sup{ratio:g}"
                f"_seed{cfg.meta.seed}_{suffix}"
            ),
            tags=["pointcloud", "test-rotate-only", method_suffix],
            group=cfg.logging.group,
            enabled=cfg.logging.enabled,
            resume=False,
        )
        if run is not None:
            final_metrics = {
                f"final/{metric}_{suffix}": row[metric] for metric in MODEL_METRICS
            }
            run.log(final_metrics)
            run.summary.update({**row, **final_metrics})
            run.finish()

    results_dir = output_dir / "test_rotate_only_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"ratio_{ratio:g}.json"
    result_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


def collect_test_results(cfg, output_dir):
    output_dir = Path(output_dir)
    expected = [
        output_dir / "test_rotate_only_results" / f"ratio_{float(ratio):g}.json"
        for ratio in cfg.sweep.supervised_ratios
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing test_rotate_only results:\n" + "\n".join(missing)
        )
    rows = []
    for path in expected:
        rows.extend(json.loads(path.read_text()))
    columns = [
        "supervised_ratio",
        "train_rotation",
        "ssl_rotation",
        "test_rotation",
        "test_method",
        "test_size",
        *MODEL_METRICS,
    ]
    csv_path = output_dir / "results_test_rotate_only.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "results_test_rotate_only.md"
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

    summary_run = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=output_dir / "wandb_summary_test_rotate_only",
        run_name=f"pointcloud_test_rotate_only_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "test-rotate-only", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=True,
    )
    if summary_run is not None:
        import wandb

        table_payload = {}
        all_results = wandb.Table(columns=columns)
        for row in rows:
            all_results.add_data(*[row[column] for column in columns])
        table_payload["results_test_rotate_only"] = all_results

        for _, method_suffix in TEST_METHODS:
            method_table = wandb.Table(columns=columns)
            for row in rows:
                if row["test_method"] == method_suffix:
                    method_table.add_data(*[row[column] for column in columns])
            table_payload[f"results_test_rotate_only_{method_suffix}"] = method_table

        summary_run.log(table_payload)
        summary_run.save(str(csv_path), base_path=str(output_dir))
        summary_run.save(str(markdown_path), base_path=str(output_dir))
        summary_run.finish()
    print(markdown_path.read_text(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("train", "test", "collect-train", "collect-test")
    )
    parser.add_argument(
        "--config", default="examples/pointcloud/cfgs/test_rotate_only.yaml"
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    destination = args.output_dir or config.meta.output_dir
    if args.command in ("train", "test") and args.task_id is None:
        raise ValueError(f"--task-id is required for {args.command!r}")
    if args.command == "train":
        run_train_task(config, args.task_id, destination)
    elif args.command == "test":
        run_test_task(config, args.task_id, destination)
    elif args.command == "collect-train":
        from examples.pointcloud.ratio_sweep import collect_results

        collect_results(config, destination)
    else:
        collect_test_results(config, destination)
