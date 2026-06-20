"""Visualize PointNet encoder embeddings and classify them with k-nearest neighbors."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torch.utils.data import Subset

from examples.pointcloud.new_class import build_datasets
from examples.pointcloud.new_class_matched_rotation import MatchedRotationDataset
from examples.pointcloud.new_class_sweep import task_config
from examples.pointcloud.ratio_sweep import PointNetClassifier, build_pointnet_encoder, make_loader

ENCODER_VARIANTS = (
    "pretrained",
    "scratch",
    "finetune_equal_lr",
    "finetune_split_lr",
)


def load_encoder(cfg, checkpoint, variant, device):
    if variant == "pretrained":
        encoder = build_pointnet_encoder(cfg.model)
        encoder.load_state_dict(checkpoint["pretrained_encoder"])
        return encoder.to(device)

    model = PointNetClassifier(build_pointnet_encoder(cfg.model), cfg.data.n_classes)
    if variant == "scratch":
        model.load_state_dict(checkpoint["scratch"])
    else:
        key = variant.removeprefix("finetune_")
        model.load_state_dict(checkpoint["finetuned_models"][key])
    return model.encoder.to(device)


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    embeddings, labels = [], []
    for point_cloud, label in loader:
        point_cloud = point_cloud.to(device, non_blocking=True)
        embeddings.append(encoder.represent(point_cloud).cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def project_embeddings(train_embeddings, test_embeddings, method, seed):
    if method == "pca":
        projector = PCA(n_components=2, random_state=seed)
        train_2d = projector.fit_transform(train_embeddings)
        test_2d = projector.transform(test_embeddings)
        return train_2d, test_2d
    combined = np.concatenate([train_embeddings, test_embeddings])
    perplexity = min(30.0, max(5.0, (len(combined) - 1) / 3.0))
    combined_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(combined)
    return combined_2d[: len(train_embeddings)], combined_2d[len(train_embeddings) :]


def plot_embeddings(train_2d, train_labels, test_2d, test_labels, predictions, output):
    n_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
    cmap = plt.get_cmap("turbo", n_classes)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=train_labels,
        cmap=cmap,
        s=8,
        alpha=0.35,
        linewidths=0,
    )
    test_plot = axes[0].scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c=test_labels,
        cmap=cmap,
        s=18,
        edgecolors="black",
        linewidths=0.2,
    )
    axes[0].set_title("Encoder space: train (faint) and test (outlined), true labels")
    axes[0].set_xlabel("component 1")
    axes[0].set_ylabel("component 2")
    fig.colorbar(test_plot, ax=axes[0], label="remapped class")

    correct = predictions == test_labels
    predicted_plot = axes[1].scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c=predictions,
        cmap=cmap,
        s=20,
        alpha=0.85,
        linewidths=0,
    )
    if np.any(~correct):
        axes[1].scatter(
            test_2d[~correct, 0],
            test_2d[~correct, 1],
            facecolors="none",
            edgecolors="black",
            marker="x",
            s=35,
            linewidths=0.8,
            label="incorrect",
        )
        axes[1].legend(loc="best")
    axes[1].set_title("k-NN predictions; black x indicates an error")
    axes[1].set_xlabel("component 1")
    axes[1].set_ylabel("component 2")
    fig.colorbar(predicted_plot, ax=axes[1], label="predicted remapped class")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confusion(labels, predictions, n_classes, output):
    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(n_classes), normalize="true"
    )
    size = max(8, min(18, n_classes * 0.55))
    fig, axis = plt.subplots(figsize=(size, size))
    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=np.arange(n_classes),
        yticklabels=np.arange(n_classes),
        ax=axis,
    )
    axis.set_xlabel("predicted remapped class")
    axis.set_ylabel("true remapped class")
    axis.set_title("Row-normalized k-NN confusion matrix")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args):
    cfg = task_config(OmegaConf.load(args.config), args.pretrain_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, train_dataset, _, test_dataset, metadata = build_datasets(cfg)
    train_dataset = MatchedRotationDataset(
        train_dataset, args.rotation, args.rotation_seed, deterministic=True
    )
    test_dataset = MatchedRotationDataset(
        test_dataset, args.rotation, args.rotation_seed, deterministic=True
    )
    if args.max_train_samples:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_test_samples:
        test_dataset = Subset(
            test_dataset, range(min(args.max_test_samples, len(test_dataset)))
        )

    loader_args = (args.batch_size or cfg.data.batch_size, args.num_workers)
    train_loader = make_loader(train_dataset, *loader_args, False, cfg.meta.seed + 70)
    test_loader = make_loader(test_dataset, *loader_args, False, cfg.meta.seed + 71)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(cfg, checkpoint, args.encoder_variant, device)
    train_embeddings, train_labels = extract_embeddings(encoder, train_loader, device)
    test_embeddings, test_labels = extract_embeddings(encoder, test_loader, device)

    if args.metric == "cosine":
        train_for_knn = normalize(train_embeddings)
        test_for_knn = normalize(test_embeddings)
    else:
        train_for_knn, test_for_knn = train_embeddings, test_embeddings
    classifier = KNeighborsClassifier(
        n_neighbors=args.neighbors,
        metric=args.metric,
        weights=args.weights,
        n_jobs=-1,
    )
    classifier.fit(train_for_knn, train_labels)
    predictions = classifier.predict(test_for_knn)
    train_2d, test_2d = project_embeddings(
        train_for_knn, test_for_knn, args.projection, cfg.meta.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inverse_label_map = {
        mapped: int(original)
        for original, mapped in metadata["label_map"].items()
    }
    predictions_path = output_dir / "knn_predictions.csv"
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "true_remapped_label",
                "predicted_remapped_label",
                "true_original_label",
                "predicted_original_label",
                "correct",
                "embedding_x",
                "embedding_y",
            ],
        )
        writer.writeheader()
        for index, (truth, prediction, position) in enumerate(
            zip(test_labels, predictions, test_2d)
        ):
            writer.writerow(
                {
                    "sample_index": index,
                    "true_remapped_label": int(truth),
                    "predicted_remapped_label": int(prediction),
                    "true_original_label": inverse_label_map[int(truth)],
                    "predicted_original_label": inverse_label_map[int(prediction)],
                    "correct": bool(truth == prediction),
                    "embedding_x": float(position[0]),
                    "embedding_y": float(position[1]),
                }
            )

    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "encoder_variant": args.encoder_variant,
        "pretrain_classes": int(args.pretrain_classes),
        "supervised_classes": int(cfg.data.n_classes),
        "rotation": args.rotation,
        "rotation_seed": int(args.rotation_seed),
        "neighbors": int(args.neighbors),
        "metric": args.metric,
        "weights": args.weights,
        "projection": args.projection,
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, predictions)
        ),
        "split_fingerprint": metadata["split_fingerprint"],
    }
    (output_dir / "knn_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_embeddings(
        train_2d,
        train_labels,
        test_2d,
        test_labels,
        predictions,
        output_dir / "knn_embedding.png",
    )
    plot_confusion(
        test_labels,
        predictions,
        int(cfg.data.n_classes),
        output_dir / "knn_confusion_matrix.png",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="examples/pointcloud/cfgs/new_class_sweep.yaml"
    )
    parser.add_argument("--pretrain-classes", type=int, choices=(10, 20, 30), required=True)
    parser.add_argument("--rotation", choices=("none", "z", "so3"), default="none")
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--encoder-variant", choices=ENCODER_VARIANTS, default="pretrained")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--weights", choices=("uniform", "distance"), default="distance")
    parser.add_argument("--projection", choices=("pca", "tsne"), default="pca")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
