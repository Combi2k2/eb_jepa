"""Disjoint-class SSL pretraining and supervised evaluation on ModelNet40."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import (
    PointCloudConfig,
    PointCloudDataset,
    PointCloudIndexedDataset,
)
from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.ratio_sweep import (
    build_pointnet_encoder,
    make_loader,
    set_seed,
    train_finetune,
    train_linear_probe,
    train_scratch,
    train_vicreg,
)


class RemappedLabelsDataset(torch.utils.data.Dataset):
    """Wrap a supervised dataset and map original ModelNet40 labels to 0..K-1."""

    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = {int(key): int(value) for key, value in label_map.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        point_cloud, original_label = self.dataset[index]
        return point_cloud, self.label_map[int(original_label)]


class TestClassSubset(torch.utils.data.Dataset):
    """Clean official-test subset restricted to selected original class IDs."""

    def __init__(self, dataset, class_ids, label_map):
        allowed = np.asarray(sorted(class_ids), dtype=np.int64)
        self.dataset = dataset
        self.indices = np.flatnonzero(np.isin(dataset.label, allowed))
        self.label_map = {int(key): int(value) for key, value in label_map.items()}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        point_cloud = self.dataset._clean(self.dataset.data[original_index])
        original_label = int(self.dataset.label[original_index])
        return torch.from_numpy(point_cloud.T.copy()), self.label_map[original_label]


def split_classes(n_classes, n_pretrain_classes, seed):
    if not 0 < int(n_pretrain_classes) < int(n_classes):
        raise ValueError("n_pretrain_classes must be between 1 and n_classes - 1")
    permutation = np.random.default_rng(int(seed)).permutation(int(n_classes))
    pretrain = np.sort(permutation[: int(n_pretrain_classes)])
    supervised = np.sort(permutation[int(n_pretrain_classes) :])
    return pretrain, supervised


def split_supervised_indices(labels, class_ids, val_ratio, seed):
    """Use all selected-class samples, with a deterministic per-class val split."""
    rng = np.random.default_rng(int(seed))
    train_indices, val_indices = [], []
    for class_id in class_ids:
        indices = rng.permutation(np.flatnonzero(labels == int(class_id)))
        n_val = min(len(indices) - 1, max(1, int(round(len(indices) * val_ratio))))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return (
        rng.permutation(np.asarray(train_indices, dtype=np.int64)),
        rng.permutation(np.asarray(val_indices, dtype=np.int64)),
    )


def split_fingerprint(pretrain_classes, supervised_classes, train_indices, val_indices):
    digest = hashlib.sha256()
    for values in (
        pretrain_classes,
        supervised_classes,
        train_indices,
        val_indices,
    ):
        digest.update(np.asarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


def build_datasets(cfg):
    train_cfg = PointCloudConfig(
        data_root=cfg.data.data_root,
        split="train",
        mode="supervised",
        n_classes=cfg.data.total_classes,
        n_points=cfg.data.n_points,
        rotate="so3",
        jitter=cfg.data.jitter,
        scale_lo=cfg.data.scale_lo,
        scale_hi=cfg.data.scale_hi,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    base_train = PointCloudDataset(train_cfg)
    pretrain_classes, supervised_classes = split_classes(
        cfg.data.total_classes, cfg.split.n_pretrain_classes, cfg.split.class_seed
    )
    pretrain_indices = np.flatnonzero(
        np.isin(base_train.label, pretrain_classes)
    ).astype(np.int64)
    supervised_train_indices, supervised_val_indices = split_supervised_indices(
        base_train.label,
        supervised_classes,
        cfg.split.val_ratio,
        cfg.split.sample_seed,
    )
    label_map = {
        int(original): mapped
        for mapped, original in enumerate(supervised_classes.tolist())
    }

    pretrain = PointCloudIndexedDataset(
        base_train,
        pretrain_indices,
        mode="ssl",
        augmentation="so3",
        seed=cfg.split.sample_seed,
    )
    supervised_train = RemappedLabelsDataset(
        PointCloudIndexedDataset(
            base_train,
            supervised_train_indices,
            mode="supervised",
            augmentation="none",
            seed=cfg.split.sample_seed,
        ),
        label_map,
    )
    supervised_val = RemappedLabelsDataset(
        PointCloudIndexedDataset(
            base_train,
            supervised_val_indices,
            mode="supervised",
            augmentation="none",
            seed=cfg.split.sample_seed,
            deterministic_augmentation=True,
        ),
        label_map,
    )

    test_cfg = copy.deepcopy(train_cfg)
    test_cfg.split = "test"
    clean_test = PointCloudDataset(test_cfg)
    test = TestClassSubset(clean_test, supervised_classes, label_map)

    expected = set(range(len(supervised_classes)))
    for name, dataset in (
        ("supervised_train", supervised_train),
        ("supervised_val", supervised_val),
        ("test", test),
    ):
        present = {dataset[index][1] for index in range(len(dataset))}
        if present != expected:
            raise RuntimeError(f"{name} is missing remapped labels: {expected - present}")

    metadata = {
        "pretrain_class_ids": pretrain_classes.tolist(),
        "supervised_class_ids": supervised_classes.tolist(),
        "label_map": {str(key): value for key, value in label_map.items()},
        "pretrain_size": len(pretrain),
        "supervised_train_size": len(supervised_train),
        "supervised_val_size": len(supervised_val),
        "test_size": len(test),
        "split_fingerprint": split_fingerprint(
            pretrain_classes,
            supervised_classes,
            supervised_train_indices,
            supervised_val_indices,
        ),
    }
    return pretrain, supervised_train, supervised_val, test, metadata


def run(cfg, output_dir):
    if int(cfg.data.n_classes) != int(cfg.data.total_classes) - int(
        cfg.split.n_pretrain_classes
    ):
        raise ValueError("data.n_classes must equal the number of supervised classes")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain, supervised_train, supervised_val, test, metadata = build_datasets(cfg)
    loader_args = (cfg.data.batch_size, cfg.data.num_workers)
    pretrain_loader = make_loader(
        pretrain, *loader_args, True, cfg.meta.seed + 10, drop_last=True
    )
    scratch_loader = make_loader(
        supervised_train, *loader_args, True, cfg.meta.seed + 20
    )
    probe_loader = make_loader(
        supervised_train, *loader_args, True, cfg.meta.seed + 20
    )
    val_loader = make_loader(
        supervised_val, *loader_args, False, cfg.meta.seed + 40
    )
    test_loader = make_loader(test, *loader_args, False, cfg.meta.seed + 50)

    run_config = OmegaConf.to_container(cfg, resolve=True) | metadata
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config=run_config,
        run_dir=output_dir / "wandb",
        run_name=(
            f"pointcloud_pretrain{cfg.split.n_pretrain_classes}"
            f"_supervised{cfg.data.n_classes}_seed{cfg.meta.seed}"
        ),
        tags=[
            "pointcloud",
            "disjoint-class",
            "ssl-so3",
            f"pretrain-{cfg.split.n_pretrain_classes}-classes",
        ],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.meta.seed)
    scratch_val, scratch_test, scratch_model = train_scratch(
        build_pointnet_encoder(cfg.model),
        scratch_loader,
        val_loader,
        test_loader,
        cfg,
        device,
        wandb_run,
    )

    set_seed(cfg.meta.seed)
    pretrained_encoder = train_vicreg(
        build_pointnet_encoder(cfg.model).to(device),
        pretrain_loader,
        cfg,
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
        cfg,
        device,
        wandb_run,
    )

    finetune_results = {}
    finetuned_models = {}
    for variant_name, variant_cfg in cfg.optim.finetune_variants.items():
        set_seed(cfg.meta.seed)
        encoder = build_pointnet_encoder(cfg.model)
        encoder.load_state_dict(pretrained_state)
        train_loader = make_loader(
            supervised_train, *loader_args, True, cfg.meta.seed + 20
        )
        val_acc, test_acc, model = train_finetune(
            encoder,
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
        finetune_results[f"finetune_{variant_name}_val_acc"] = val_acc
        finetune_results[f"finetune_{variant_name}_test_acc"] = test_acc
        finetuned_models[variant_name] = model.state_dict()

    result = metadata | {
        "scratch_val_acc": scratch_val,
        "scratch_test_acc": scratch_test,
        "pretrained_probe_val_acc": probe_val,
        "pretrained_probe_test_acc": probe_test,
        **finetune_results,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "result": result,
            "scratch": scratch_model.state_dict(),
            "pretrained_encoder": pretrained_state,
            "linear_probe": probe_head.state_dict(),
            "finetuned_models": finetuned_models,
        },
        output_dir / "models.pth.tar",
    )

    if wandb_run is not None:
        import wandb

        rows = [
            ("scratch", scratch_val, scratch_test),
            ("pretrained_probe", probe_val, probe_test),
            (
                "finetune_equal_lr",
                result["finetune_equal_lr_val_acc"],
                result["finetune_equal_lr_test_acc"],
            ),
            (
                "finetune_split_lr",
                result["finetune_split_lr_val_acc"],
                result["finetune_split_lr_test_acc"],
            ),
        ]
        table = wandb.Table(columns=["method", "val_acc", "test_acc"], data=rows)
        final_metrics = {
            f"final/{key}": value
            for key, value in result.items()
            if key.endswith("_acc")
        }
        wandb_run.log({"results_new_class": table, **final_metrics})
        wandb_run.summary.update(result | final_metrics)
        wandb_run.finish()
    print(json.dumps(result, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="examples/pointcloud/cfgs/new_class.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    run(config, args.output_dir or config.meta.output_dir)
