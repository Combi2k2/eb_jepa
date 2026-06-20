"""Tuned vanilla VICReg across augmentation protocols and class splits."""

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eb_jepa.training_utils import setup_wandb
from examples.pointcloud.new_class import build_datasets as build_disjoint_datasets
from examples.pointcloud.new_class_matched_rotation import MatchedRotationDataset
from examples.pointcloud.new_class_sweep import task_config
from examples.pointcloud.new_loss import _all_class_datasets
from examples.pointcloud.ratio_sweep import (
    build_pointnet_encoder,
    make_loader,
    set_seed,
    train_finetune,
    train_linear_probe,
    train_scratch,
    train_vicreg,
)

PROTOCOLS = ("clean_to_so3", "so3_to_so3")


def build_protocol_datasets(cfg, pretrain_setup, protocol):
    if pretrain_setup == "all40":
        current_cfg = copy.deepcopy(cfg)
        current_cfg.data.n_classes = 40
        pretrain, clean_train, clean_val, clean_test, metadata = _all_class_datasets(
            current_cfg
        )
    else:
        current_cfg = task_config(cfg, int(pretrain_setup))
        pretrain, clean_train, clean_val, clean_test, metadata = (
            build_disjoint_datasets(current_cfg)
        )
        metadata["class_setup"] = (
            f"{int(pretrain_setup)}_{int(current_cfg.data.n_classes)}"
        )
        metadata["pretrain_supervised_overlap"] = False

    train_rotation = "none" if protocol == "clean_to_so3" else "so3"
    train = MatchedRotationDataset(
        clean_train, train_rotation, cfg.sweep.rotation_seed, deterministic=False
    )
    val = MatchedRotationDataset(
        clean_val, train_rotation, cfg.sweep.rotation_seed, deterministic=True
    )
    test = MatchedRotationDataset(
        clean_test, "so3", cfg.sweep.rotation_seed, deterministic=True
    )
    metadata["supervised_protocol"] = protocol
    metadata["supervised_train_rotation"] = train_rotation
    metadata["supervised_val_rotation"] = train_rotation
    metadata["test_rotation"] = "so3"
    return current_cfg, pretrain, train, val, test, metadata


def run_task(cfg, task_id, output_dir):
    combinations = [
        (str(setup), protocol)
        for setup in cfg.sweep.pretrain_class_setups
        for protocol in PROTOCOLS
    ]
    if not 0 <= int(task_id) < len(combinations):
        raise ValueError(f"task_id must be in [0, {len(combinations) - 1}]")
    setup, protocol = combinations[int(task_id)]
    setup_value = setup if setup == "all40" else int(setup)
    current_cfg, pretrain, train, val, test, metadata = build_protocol_datasets(
        cfg, setup_value, protocol
    )
    class_setup = metadata.get(
        "class_setup",
        f"{metadata['pretrain_classes']}_{metadata['supervised_classes']}",
    )
    loader_args = (current_cfg.data.batch_size, current_cfg.data.num_workers)
    pretrain_loader = make_loader(
        pretrain, *loader_args, True, current_cfg.meta.seed + 10, drop_last=True
    )
    scratch_loader = make_loader(train, *loader_args, True, current_cfg.meta.seed + 20)
    probe_loader = make_loader(train, *loader_args, True, current_cfg.meta.seed + 20)
    val_loader = make_loader(val, *loader_args, False, current_cfg.meta.seed + 40)
    test_loader = make_loader(test, *loader_args, False, current_cfg.meta.seed + 50)

    run_dir = Path(output_dir) / "runs" / f"setup_{class_setup}_{protocol}"
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config=OmegaConf.to_container(current_cfg, resolve=True) | metadata,
        run_dir=run_dir / "wandb",
        run_name=f"pointcloud_{class_setup}_{protocol}_seed{cfg.meta.seed}",
        tags=["pointcloud", "vicreg", "tuned-loss", protocol, class_setup],
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
    pretrained_encoder = train_vicreg(
        build_pointnet_encoder(current_cfg.model).to(device),
        pretrain_loader,
        current_cfg,
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
        "class_setup": class_setup,
        "ssl_loss_method": "vicreg",
        "sim_coeff": float(current_cfg.model.sim_coeff),
        "std_coeff": float(current_cfg.model.std_coeff),
        "cov_coeff": float(current_cfg.model.cov_coeff),
        "ssl_augmentation": "so3",
        "scratch_val_acc": scratch_val,
        "scratch_test_acc": scratch_test,
        "pretrained_probe_val_acc": probe_val,
        "pretrained_probe_test_acc": probe_test,
        **finetune_results,
    }
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"setup_{class_setup}_{protocol}.json"
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
        class_setup = "all40" if str(setup) == "all40" else f"{int(setup)}_{40-int(setup)}"
        for protocol in PROTOCOLS:
            path = output_dir / "results" / f"setup_{class_setup}_{protocol}.json"
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
        "supervised_protocol",
        "ssl_augmentation",
        "supervised_train_rotation",
        "supervised_val_rotation",
        "test_rotation",
        "sim_coeff",
        "std_coeff",
        "cov_coeff",
        "scratch_test_acc",
        "pretrained_probe_test_acc",
        "finetune_equal_lr_test_acc",
        "finetune_split_lr_test_acc",
    ]
    csv_path = output_dir / "results_vicreg_tuned_setups.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)
    summary = setup_wandb(
        project=cfg.logging.project,
        config=cfg,
        run_dir=output_dir / "wandb_summary",
        run_name=f"pointcloud_vicreg_tuned_setups_summary_seed{cfg.meta.seed}",
        tags=["pointcloud", "vicreg", "tuned-loss", "summary"],
        group=cfg.logging.group,
        enabled=cfg.logging.enabled,
        resume=False,
    )
    if summary is not None:
        import wandb

        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[column] for column in columns])
        summary.log({"results_vicreg_tuned_setups": table})
        summary.save(str(csv_path), base_path=str(output_dir))
        summary.finish()
    print(csv_path.read_text(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument(
        "--config", default="examples/pointcloud/cfgs/vicreg_tuned_setups.yaml"
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
