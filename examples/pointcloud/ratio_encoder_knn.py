"""k-NN and PCA evaluation for all-class pretrain/supervised ratio checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.ratio_sweep import build_datasets, make_loader
from examples.pointcloud.visualize_encoder_knn import (
    ENCODER_VARIANTS,
    extract_embeddings,
    load_encoder,
    plot_confusion,
    plot_embeddings,
    project_embeddings,
)


HEAD_METRICS = {
    "scratch": "scratch_test_acc",
    "pretrained": "pretrained_probe_test_acc",
    "finetune_equal_lr": "finetune_equal_lr_test_acc",
    "finetune_split_lr": "finetune_split_lr_test_acc",
}


def combinations(cfg):
    return [
        (str(augmentation), float(ratio))
        for augmentation in cfg.sweep.augmentations
        for ratio in cfg.sweep.supervised_ratios
    ]


def condition_dir(output_dir, augmentation, ratio):
    return (
        Path(output_dir)
        / "ratio_knn_benchmark"
        / f"aug_{augmentation}_ratio_{float(ratio):g}"
    )


def run_task(cfg, task_id, checkpoint_root, output_dir):
    conditions = combinations(cfg)
    if not 0 <= int(task_id) < len(conditions):
        raise ValueError(f"task_id must be in [0, {len(conditions) - 1}]")
    augmentation, ratio = conditions[int(task_id)]
    checkpoint_path = (
        Path(checkpoint_root)
        / "runs"
        / f"aug_{augmentation}_ratio_{ratio:g}"
        / "models.pth.tar"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    datasets, partitions, _ = build_datasets(cfg, ratio, augmentation)
    # k-NN needs a stable reference bank. Draw one reproducible augmentation per
    # supervised-train object instead of changing it between encoder variants.
    datasets["supervised_train"].deterministic_augmentation = True
    loader_args = (cfg.data.batch_size, cfg.data.num_workers)
    train_loader = make_loader(
        datasets["supervised_train"], *loader_args, False, cfg.meta.seed + 70
    )
    test_loader = make_loader(datasets["test"], *loader_args, False, cfg.meta.seed + 71)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    task_dir = condition_dir(output_dir, augmentation, ratio)
    rows = []

    for variant in ENCODER_VARIANTS:
        encoder = load_encoder(cfg, checkpoint, variant, device)
        train_embeddings, train_labels = extract_embeddings(encoder, train_loader, device)
        test_embeddings, test_labels = extract_embeddings(encoder, test_loader, device)
        train_knn = normalize(train_embeddings)
        test_knn = normalize(test_embeddings)
        classifier = KNeighborsClassifier(
            n_neighbors=int(cfg.knn.neighbors),
            metric=str(cfg.knn.metric),
            weights=str(cfg.knn.weights),
            n_jobs=-1,
        )
        classifier.fit(train_knn, train_labels)
        predictions = classifier.predict(test_knn)
        train_2d, test_2d = project_embeddings(
            train_knn, test_knn, str(cfg.knn.projection), cfg.meta.seed
        )
        variant_dir = task_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        plot_embeddings(
            train_2d,
            train_labels,
            test_2d,
            test_labels,
            predictions,
            variant_dir / "knn_embedding_pca.png",
        )
        plot_confusion(
            test_labels,
            predictions,
            cfg.data.n_classes,
            variant_dir / "knn_confusion_matrix.png",
        )
        row = {
            "supervised_ratio": ratio,
            "pretrain_ratio": 1.0 - ratio,
            "pretrain_size": len(partitions.pretrain),
            "supervised_train_size": len(partitions.supervised_train),
            "supervised_val_size": len(partitions.supervised_val),
            "test_size": len(datasets["test"]),
            "split_fingerprint": partitions.fingerprint(),
            "ssl_augmentation": str(cfg.sweep.ssl_augmentation),
            "supervised_augmentation": augmentation,
            "test_augmentation": augmentation,
            "encoder_variant": variant,
            "prediction_head": "Linear(1024,40)",
            "head_test_acc": float(checkpoint["result"][HEAD_METRICS[variant]]),
            "knn_neighbors": int(cfg.knn.neighbors),
            "knn_metric": str(cfg.knn.metric),
            "knn_test_acc": float(accuracy_score(test_labels, predictions)),
            "knn_balanced_acc": float(
                balanced_accuracy_score(test_labels, predictions)
            ),
        }
        rows.append(row)
        (variant_dir / "knn_metrics.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb_run = setup_wandb(
        project=cfg.logging.knn_project,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {
            "supervised_ratio": ratio,
            "pretrain_ratio": 1.0 - ratio,
            "ssl_augmentation": str(cfg.sweep.ssl_augmentation),
            "supervised_augmentation": augmentation,
            "test_augmentation": augmentation,
            "split_fingerprint": partitions.fingerprint(),
        },
        run_dir=task_dir / "wandb",
        run_name=f"pointcloud_ratio{ratio:g}_{augmentation}_encoder_knn_seed{cfg.meta.seed}",
        tags=["pointcloud", "ratio-sweep", "encoder-knn", augmentation],
        group=cfg.logging.knn_group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if wandb_run is not None:
        import wandb

        columns = list(rows[0])
        table = wandb.Table(columns=columns)
        payload = {"knn_results": table}
        for row in rows:
            table.add_data(*[row[column] for column in columns])
            variant = row["encoder_variant"]
            payload[f"final/{variant}_knn_test_acc"] = row["knn_test_acc"]
            payload[f"final/{variant}_head_test_acc"] = row["head_test_acc"]
            payload[f"pca/{variant}_embedding"] = wandb.Image(
                str(task_dir / variant / "knn_embedding_pca.png")
            )
            payload[f"confusion/{variant}"] = wandb.Image(
                str(task_dir / variant / "knn_confusion_matrix.png")
            )
        wandb_run.log(payload)
        wandb_run.summary.update(
            {key: value for key, value in payload.items() if key.startswith("final/")}
        )
        wandb_run.finish()


def collect_results(cfg, output_dir):
    output_dir = Path(output_dir)
    rows = []
    for augmentation, ratio in combinations(cfg):
        for variant in ENCODER_VARIANTS:
            path = (
                condition_dir(output_dir, augmentation, ratio)
                / variant
                / "knn_metrics.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"missing result: {path}")
            rows.append(json.loads(path.read_text()))
    columns = list(rows[0])
    csv_path = output_dir / "results_ratio_encoder_knn.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = setup_wandb(
        project=cfg.logging.knn_project,
        config=cfg,
        run_dir=output_dir / "wandb_ratio_knn_summary",
        run_name=f"pointcloud_ratio_encoder_knn_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "ratio-sweep", "encoder-knn", "summary"],
        group=cfg.logging.knn_group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary.log({"results_ratio_encoder_knn": table})
        summary.save(str(csv_path), base_path=str(output_dir))
        summary.finish()
    print(csv_path.read_text(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument(
        "--config", default="examples/pointcloud/cfgs/ratio_encoder_knn.yaml"
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    destination = args.output_dir or config.meta.output_dir
    if args.command == "run":
        if args.task_id is None:
            raise ValueError("--task-id is required for run")
        source = args.checkpoint_root or config.meta.checkpoint_root
        run_task(config, args.task_id, source, destination)
    else:
        collect_results(config, destination)
