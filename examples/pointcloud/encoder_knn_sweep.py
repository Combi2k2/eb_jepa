"""k-NN benchmark over every matched-rotation disjoint-class encoder."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.visualize_encoder_knn import ENCODER_VARIANTS, run as run_knn

ROTATIONS = ("none", "z", "so3")


def combinations(cfg):
    return [
        (int(count), rotation)
        for count in cfg.sweep.pretrain_class_counts
        for rotation in ROTATIONS
    ]


def condition_dir(output_dir, count, rotation):
    return (
        Path(output_dir)
        / "knn_benchmark"
        / f"pretrain_{int(count)}_rotation_{rotation}"
    )


def run_task(cfg, task_id, output_dir):
    conditions = combinations(cfg)
    if not 0 <= int(task_id) < len(conditions):
        raise ValueError(f"task_id must be in [0, {len(conditions) - 1}]")
    count, rotation = conditions[int(task_id)]
    checkpoint = (
        Path(output_dir)
        / "matched_rotation_runs"
        / f"pretrain_{count}_rotation_{rotation}"
        / "models.pth.tar"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing matched-rotation checkpoint: {checkpoint}")

    run_dir = condition_dir(output_dir, count, rotation)
    results = []
    for variant in ENCODER_VARIANTS:
        variant_dir = run_dir / variant
        args = SimpleNamespace(
            checkpoint=str(checkpoint),
            config="examples/pointcloud/cfgs/new_class_sweep.yaml",
            pretrain_classes=count,
            rotation=rotation,
            rotation_seed=int(cfg.sweep.rotation_seed),
            encoder_variant=variant,
            neighbors=int(cfg.knn.neighbors),
            metric=str(cfg.knn.metric),
            weights=str(cfg.knn.weights),
            projection=str(cfg.knn.projection),
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            max_train_samples=None,
            max_test_samples=None,
            output_dir=str(variant_dir),
        )
        results.append(run_knn(args))

    wandb_run = setup_wandb(
        project=cfg.logging.knn_project,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {
            "pretrain_classes": count,
            "supervised_classes": 40 - count,
            "rotation": rotation,
            "checkpoint": str(checkpoint),
        },
        run_dir=run_dir / "wandb",
        run_name=(
            f"pointcloud_knn_pretrain{count}_supervised{40-count}"
            f"_rotation-{rotation}_seed{cfg.meta.seed}"
        ),
        tags=["pointcloud", "encoder-knn", rotation],
        group=cfg.logging.knn_group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if wandb_run is not None:
        import wandb

        table_columns = [
            "encoder_variant",
            "pretrain_classes",
            "supervised_classes",
            "rotation",
            "neighbors",
            "metric",
            "train_samples",
            "test_samples",
            "accuracy",
            "balanced_accuracy",
            "split_fingerprint",
        ]
        table = wandb.Table(columns=table_columns)
        payload = {"knn_results": table}
        for result in results:
            table.add_data(*[result[column] for column in table_columns])
            variant = result["encoder_variant"]
            payload[f"final/{variant}_knn_test_acc"] = result["accuracy"]
            payload[f"final/{variant}_knn_balanced_acc"] = result[
                "balanced_accuracy"
            ]
            payload[f"visualization/{variant}_embedding"] = wandb.Image(
                str(run_dir / variant / "knn_embedding.png")
            )
            payload[f"visualization/{variant}_confusion"] = wandb.Image(
                str(run_dir / variant / "knn_confusion_matrix.png")
            )
        wandb_run.log(payload)
        wandb_run.summary.update(
            {key: value for key, value in payload.items() if key.startswith("final/")}
        )
        wandb_run.finish()


def collect_results(cfg, output_dir):
    output_dir = Path(output_dir)
    rows = []
    for count, rotation in combinations(cfg):
        for variant in ENCODER_VARIANTS:
            path = (
                condition_dir(output_dir, count, rotation)
                / variant
                / "knn_metrics.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"missing k-NN result: {path}")
            rows.append(json.loads(path.read_text()))
    columns = [
        "pretrain_classes",
        "supervised_classes",
        "rotation",
        "encoder_variant",
        "neighbors",
        "metric",
        "weights",
        "projection",
        "train_samples",
        "test_samples",
        "accuracy",
        "balanced_accuracy",
        "split_fingerprint",
        "checkpoint",
    ]
    csv_path = output_dir / "results_encoder_knn.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)

    summary = setup_wandb(
        project=cfg.logging.knn_project,
        config=cfg,
        run_dir=output_dir / "wandb_knn_summary",
        run_name=f"pointcloud_encoder_knn_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "encoder-knn", "summary"],
        group=cfg.logging.knn_group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary.log({"results_encoder_knn": table})
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
