"""Action-conditioned JEPA for predicting rotated point-cloud representations."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import (
    PointCloudConfig,
    PointCloudDataset,
    PointCloudIndexedDataset,
    _rand_rot,
    stratified_train_partitions,
)
from eb_jepa.losses import CovarianceLoss, HingeStdLoss
from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.ratio_sweep import (
    assert_class_coverage,
    build_pointnet_encoder,
    make_loader,
    set_seed,
    train_finetune,
    train_linear_probe,
    train_scratch,
)


class PointCloudActionDataset(torch.utils.data.Dataset):
    """Original view, rotated view, and the exact rotation action matrix."""

    def __init__(self, dataset, indices, rotation="so3", seed=0):
        if dataset.cfg.split != "train":
            raise ValueError("PointCloudActionDataset requires the train split")
        if rotation not in ("z", "z180", "so3"):
            raise ValueError("action rotation must be z, z180, or so3")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rotation = rotation
        self.seed = int(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        source = self.dataset._clean(self.dataset.data[original_index])
        random_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        rng = np.random.default_rng(random_seed)
        rotation = _rand_rot(rng, self.rotation)
        target = source @ rotation.T
        target = self.dataset._normalize(target).astype(np.float32)
        label = int(self.dataset.label[original_index])
        return (
            torch.from_numpy(source.T.copy()),
            torch.from_numpy(target.T.copy()),
            torch.from_numpy(rotation.reshape(-1).copy()),
            label,
        )


class ActionPredictor(nn.Module):
    def __init__(self, representation_dim, action_dim=9, action_hidden=128, hidden=2048):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.LayerNorm(action_hidden),
            nn.GELU(),
            nn.Linear(action_hidden, action_hidden),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(representation_dim + action_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, representation_dim),
        )

    def forward(self, representation, action):
        encoded_action = self.action_encoder(action)
        return self.predictor(torch.cat((representation, encoded_action), dim=-1))


class ActionConditionedJEPA(nn.Module):
    def __init__(self, context_encoder, cfg):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = copy.deepcopy(context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = ActionPredictor(
            representation_dim=context_encoder.out_dim,
            action_dim=9,
            action_hidden=cfg.action_hidden,
            hidden=cfg.predictor_hidden,
        )
        self.prediction_coeff = float(cfg.prediction_coeff)
        self.std_coeff = float(cfg.std_coeff)
        self.cov_coeff = float(cfg.cov_coeff)
        self.ema_momentum = float(cfg.ema_momentum)
        self.std_loss = HingeStdLoss(std_margin=1.0)
        self.cov_loss = CovarianceLoss()

    def compute_loss(self, batch):
        source, target, action = batch[:3]
        context_representation = self.context_encoder.represent(source)
        prediction = self.predictor(context_representation, action)
        self.target_encoder.eval()
        with torch.no_grad():
            target_representation = self.target_encoder.represent(target)
        prediction_loss = F.mse_loss(
            F.normalize(prediction, dim=-1),
            F.normalize(target_representation, dim=-1),
        )
        var_loss = self.std_loss(context_representation) + self.std_loss(prediction)
        cov_loss = self.cov_loss(context_representation) + self.cov_loss(prediction)
        loss = (
            self.prediction_coeff * prediction_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )
        return {
            "loss": loss,
            "prediction_loss": prediction_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }

    @torch.no_grad()
    def update_target_encoder(self):
        momentum = self.ema_momentum
        for target, context in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target.mul_(momentum).add_(context, alpha=1.0 - momentum)
        for target, context in zip(
            self.target_encoder.buffers(), self.context_encoder.buffers()
        ):
            if target.dtype.is_floating_point:
                target.mul_(momentum).add_(context, alpha=1.0 - momentum)
            else:
                target.copy_(context)


def train_action_jepa(encoder, loader, cfg, device, run):
    model = ActionConditionedJEPA(encoder, cfg.action_jepa).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.optim.ssl_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    for epoch in range(int(cfg.optim.ssl_epochs)):
        model.train()
        totals = {"loss": 0.0, "prediction_loss": 0.0, "var_loss": 0.0, "cov_loss": 0.0}
        samples = 0
        for batch in loader:
            batch = [item.to(device, non_blocking=True) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            components = model.compute_loss(batch)
            components["loss"].backward()
            optimizer.step()
            model.update_target_encoder()
            batch_size = len(batch[0])
            samples += batch_size
            for key in totals:
                totals[key] += components[key].detach().item() * batch_size
        if run is not None:
            run.log(
                {f"action_jepa/{key}": value / samples for key, value in totals.items()}
                | {"action_jepa/epoch": epoch}
            )
    return model.context_encoder, model.target_encoder, model.predictor


def build_datasets(cfg, ratio):
    data_cfg = PointCloudConfig(
        data_root=cfg.data.data_root,
        split="train",
        mode="supervised",
        n_classes=cfg.data.n_classes,
        n_points=cfg.data.n_points,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    base_train = PointCloudDataset(data_cfg)
    partitions = stratified_train_partitions(
        base_train.label, ratio, cfg.sweep.val_ratio, cfg.sweep.split_seed
    )
    assert_class_coverage(base_train.label, partitions, cfg.data.n_classes)
    pretrain = PointCloudActionDataset(
        base_train,
        partitions.pretrain,
        rotation=cfg.action_jepa.rotation_action,
        seed=cfg.sweep.split_seed,
    )
    supervised_train = PointCloudIndexedDataset(
        base_train,
        partitions.supervised_train,
        mode="supervised",
        augmentation="none",
        seed=cfg.sweep.split_seed,
    )
    supervised_val = PointCloudIndexedDataset(
        base_train,
        partitions.supervised_val,
        mode="supervised",
        augmentation="none",
        seed=cfg.sweep.split_seed,
        deterministic_augmentation=True,
    )
    test_cfg = copy.deepcopy(data_cfg)
    test_cfg.split = "test"
    test = PointCloudDataset(test_cfg)
    return pretrain, supervised_train, supervised_val, test, partitions, base_train.label


def run_task(cfg, task_id, output_dir):
    ratios = [float(value) for value in cfg.sweep.supervised_ratios]
    if not 0 <= int(task_id) < len(ratios):
        raise ValueError(f"task_id must be in [0, {len(ratios) - 1}]")
    ratio = ratios[int(task_id)]
    pretrain, train, val, test, partitions, labels = build_datasets(cfg, ratio)
    loader_args = (cfg.data.batch_size, cfg.data.num_workers)
    pretrain_loader = make_loader(pretrain, *loader_args, True, cfg.meta.seed + 10, drop_last=True)
    scratch_loader = make_loader(train, *loader_args, True, cfg.meta.seed + 20)
    probe_loader = make_loader(train, *loader_args, True, cfg.meta.seed + 20)
    val_loader = make_loader(val, *loader_args, False, cfg.meta.seed + 40)
    test_loader = make_loader(test, *loader_args, False, cfg.meta.seed + 50)
    run_dir = Path(output_dir) / "runs" / f"ratio_{ratio:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {
            "supervised_ratio": ratio,
            "pretrain_ratio": 1.0 - ratio,
            "split_fingerprint": partitions.fingerprint(),
            "action_representation": "flattened_3x3_rotation_matrix",
        },
        run_dir=run_dir / "wandb",
        run_name=f"pointcloud_action_jepa_ratio{ratio:g}_seed{cfg.meta.seed}",
        tags=["pointcloud", "action-jepa", "rotation-prediction"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.meta.seed)
    scratch_val, scratch_test, scratch_model = train_scratch(
        build_pointnet_encoder(cfg.model), scratch_loader, val_loader, test_loader, cfg, device, wandb_run
    )
    set_seed(cfg.meta.seed)
    encoder, target_encoder, predictor = train_action_jepa(
        build_pointnet_encoder(cfg.model).to(device), pretrain_loader, cfg, device, wandb_run
    )
    pretrained_state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    probe_val, probe_test, probe_head = train_linear_probe(
        encoder, probe_loader, val_loader, test_loader, cfg, device, wandb_run
    )
    finetune_results = {}
    finetuned_models = {}
    for variant_name, variant_cfg in cfg.optim.finetune_variants.items():
        set_seed(cfg.meta.seed)
        finetune_encoder = build_pointnet_encoder(cfg.model)
        finetune_encoder.load_state_dict(pretrained_state)
        train_loader = make_loader(train, *loader_args, True, cfg.meta.seed + 20)
        val_acc, test_acc, model = train_finetune(
            finetune_encoder,
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
    result = {
        "supervised_ratio": ratio,
        "pretrain_ratio": 1.0 - ratio,
        "pretrain_size": len(partitions.pretrain),
        "supervised_train_size": len(partitions.supervised_train),
        "supervised_val_size": len(partitions.supervised_val),
        "test_size": len(test),
        "split_fingerprint": partitions.fingerprint(),
        "train_classes": len(np.unique(labels[partitions.supervised_train])),
        "val_classes": len(np.unique(labels[partitions.supervised_val])),
        "rotation_action": cfg.action_jepa.rotation_action,
        "scratch_val_acc": scratch_val,
        "scratch_test_acc": scratch_test,
        "pretrained_probe_val_acc": probe_val,
        "pretrained_probe_test_acc": probe_test,
        **finetune_results,
    }
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"ratio_{ratio:g}.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "result": result,
            "scratch": scratch_model.state_dict(),
            "pretrained_encoder": pretrained_state,
            "target_encoder": target_encoder.state_dict(),
            "action_predictor": predictor.state_dict(),
            "linear_probe": probe_head.state_dict(),
            "finetuned_models": finetuned_models,
        },
        run_dir / "models.pth.tar",
    )
    if wandb_run is not None:
        final_metrics = {f"final/{key}": value for key, value in result.items() if key.endswith("_acc")}
        wandb_run.log(final_metrics)
        wandb_run.summary.update(result | final_metrics)
        wandb_run.finish()


def collect_results(cfg, output_dir):
    output_dir = Path(output_dir)
    rows = [
        json.loads((output_dir / "results" / f"ratio_{float(ratio):g}.json").read_text())
        for ratio in cfg.sweep.supervised_ratios
    ]
    columns = [
        "supervised_ratio", "pretrain_ratio", "pretrain_size", "supervised_train_size",
        "supervised_val_size", "test_size", "split_fingerprint", "rotation_action",
        "scratch_test_acc", "pretrained_probe_test_acc", "finetune_equal_lr_test_acc",
        "finetune_split_lr_test_acc",
    ]
    csv_path = output_dir / "results_action_jepa.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)
    summary = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=output_dir / "wandb_summary",
        run_name=f"pointcloud_action_jepa_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "action-jepa", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary is not None:
        import wandb
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary.log({"results_action_jepa": table})
        summary.save(str(csv_path), base_path=str(output_dir))
        summary.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--config", default="examples/pointcloud/cfgs/action_jepa.yaml")
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
