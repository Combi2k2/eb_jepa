"""Normalize all completed point-cloud result tables into one long-form CSV."""

import csv
import json
from pathlib import Path


ROOT = Path("/lustre/work/vivatech-ipparis/lnguyen/checkpoints/pointcloud")
OUTPUT = Path("examples/pointcloud/all_experiment_results.csv")

COLUMNS = [
    "experiment_id",
    "source_table",
    "wandb_project",
    "experiment_note",
    "backbone",
    "class_protocol",
    "pretrain_classes",
    "supervised_classes",
    "supervised_ratio",
    "ssl_augmentation",
    "supervised_augmentation",
    "test_augmentation",
    "method",
    "val_acc",
    "test_acc",
    "pretrain_size",
    "supervised_train_size",
    "supervised_val_size",
    "test_size",
    "split_seed",
    "split_fingerprint",
    "status",
]

METHODS = (
    ("scratch", "scratch_val_acc", "scratch_test_acc"),
    ("pretrained_probe", "pretrained_probe_val_acc", "pretrained_probe_test_acc"),
    ("finetune", "finetune_val_acc", "finetune_test_acc"),
    (
        "finetune_equal_lr",
        "finetune_equal_lr_val_acc",
        "finetune_equal_lr_test_acc",
    ),
    (
        "finetune_split_lr",
        "finetune_split_lr_val_acc",
        "finetune_split_lr_test_acc",
    ),
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row, key, default=""):
    result = row.get(key, default)
    return default if result is None else result


def append_wide(rows, source, spec, raw_rows):
    for raw in raw_rows:
        for method, val_key, test_key in METHODS:
            if test_key not in raw or raw[test_key] == "":
                continue
            rows.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "source_table": str(source),
                    "wandb_project": spec["wandb_project"],
                    "experiment_note": spec["note"],
                    "backbone": spec["backbone"],
                    "class_protocol": spec.get("class_protocol", "all_40_classes"),
                    "pretrain_classes": value(raw, "pretrain_classes", spec.get("pretrain_classes", 40)),
                    "supervised_classes": value(raw, "supervised_classes", spec.get("supervised_classes", 40)),
                    "supervised_ratio": value(raw, "supervised_ratio"),
                    "ssl_augmentation": spec["ssl_aug"](raw),
                    "supervised_augmentation": spec["supervised_aug"](raw),
                    "test_augmentation": spec["test_aug"](raw),
                    "method": method,
                    "val_acc": value(raw, val_key),
                    "test_acc": raw[test_key],
                    "pretrain_size": value(raw, "pretrain_size"),
                    "supervised_train_size": value(raw, "supervised_train_size"),
                    "supervised_val_size": value(raw, "supervised_val_size"),
                    "test_size": value(raw, "test_size", spec.get("test_size", 2468)),
                    "split_seed": value(raw, "split_seed", spec.get("split_seed", 0)),
                    "split_fingerprint": value(raw, "split_fingerprint"),
                    "status": "completed",
                }
            )


def fixed(text):
    return lambda _: text


def field(name, default=""):
    return lambda row: value(row, name, default)


def add_csv(rows, relative_path, spec):
    path = ROOT / relative_path
    append_wide(rows, path, spec, read_csv(path))


def main():
    rows = []

    add_csv(
        rows,
        "ratio_sweep/results.csv",
        {
            "experiment_id": "ratio_sweep_legacy_clean_test",
            "wandb_project": "eb_jepa",
            "note": "Legacy 4-ratio sweep on all 40 classes; SSL and supervised data use the selected none/z/SO3 augmentation; official test is clean.",
            "backbone": "simple_pointnet",
            "ssl_aug": field("augmentation"),
            "supervised_aug": field("augmentation"),
            "test_aug": fixed("none"),
        },
    )
    add_csv(
        rows,
        "ratio_sweep_v2/results.csv",
        {
            "experiment_id": "ratio_sweep_v2_clean_test",
            "wandb_project": "eb_jepa",
            "note": "Final 4-ratio all-class sweep with scratch, frozen probe, equal-LR fine-tuning and split-LR fine-tuning; official test is clean.",
            "backbone": "simple_pointnet",
            "ssl_aug": field("augmentation"),
            "supervised_aug": field("augmentation"),
            "test_aug": fixed("none"),
        },
    )
    add_csv(
        rows,
        "ratio_sweep_v2/results_argument_test.csv",
        {
            "experiment_id": "ratio_sweep_v2_cross_rotation_test",
            "wandb_project": "eb_jepa",
            "note": "Cross-rotation benchmark: every all-class training augmentation and ratio is evaluated on deterministic none/z/SO3 versions of the complete official test set.",
            "backbone": "simple_pointnet",
            "ssl_aug": field("train_augmentation"),
            "supervised_aug": field("train_augmentation"),
            "test_aug": field("test_augmentation"),
        },
    )
    add_csv(
        rows,
        "test_rotate_only/results.csv",
        {
            "experiment_id": "legacy_no_rotation_training_clean_test",
            "wandb_project": "eb_jepa",
            "note": "Legacy no-rotation training experiment retained for provenance; clean official-test results before the corrected SO3-SSL protocol.",
            "backbone": "simple_pointnet",
            "ssl_aug": fixed("no_rotation"),
            "supervised_aug": fixed("none"),
            "test_aug": fixed("none"),
        },
    )
    add_csv(
        rows,
        "test_rotate_only/results_test_rotate_only.csv",
        {
            "experiment_id": "legacy_no_rotation_training_rotation_test",
            "wandb_project": "eb_jepa",
            "note": "Legacy no-rotation model evaluated on deterministic none/z/SO3 official-test views; retained for provenance.",
            "backbone": "simple_pointnet",
            "ssl_aug": fixed("no_rotation"),
            "supervised_aug": field("train_rotation", "none"),
            "test_aug": field("test_rotation"),
        },
    )
    add_csv(
        rows,
        "eb_jepa2_rotation_protocol/results.csv",
        {
            "experiment_id": "clean_supervised_so3_ssl_clean_test",
            "wandb_project": "eb_jepa2",
            "note": "Corrected protocol: clean supervised train/val, two-view SO3 VICReg pretraining, four ratios, and clean official-test evaluation.",
            "backbone": "simple_pointnet",
            "ssl_aug": fixed("so3"),
            "supervised_aug": fixed("none"),
            "test_aug": fixed("none"),
        },
    )
    add_csv(
        rows,
        "eb_jepa2_rotation_protocol/results_test_rotate_only.csv",
        {
            "experiment_id": "clean_supervised_so3_ssl_rotation_test",
            "wandb_project": "eb_jepa2",
            "note": "Corrected clean-supervised/SO3-SSL models evaluated on deterministic none/z/SO3 versions of the unchanged official test set.",
            "backbone": "simple_pointnet",
            "ssl_aug": field("ssl_rotation", "so3"),
            "supervised_aug": field("train_rotation", "none"),
            "test_aug": field("test_rotation"),
        },
    )
    add_csv(
        rows,
        "eb_jepa_new_pointnet_rotation_protocol/results.csv",
        {
            "experiment_id": "yanx27_pointnet_clean_test",
            "wandb_project": "eb_jepa_new_pointnet",
            "note": "Same clean-supervised/SO3-SSL ratio protocol using the yanx27-style PointNet encoder with input and feature T-Nets; classifier remains Linear(1024,40).",
            "backbone": "yanx27_pointnet_stn",
            "ssl_aug": fixed("so3"),
            "supervised_aug": fixed("none"),
            "test_aug": fixed("none"),
        },
    )
    add_csv(
        rows,
        "eb_jepa_new_pointnet_rotation_protocol/results_test_rotate_only.csv",
        {
            "experiment_id": "yanx27_pointnet_rotation_test",
            "wandb_project": "eb_jepa_new_pointnet",
            "note": "Yanx27-style PointNet models from the clean-supervised/SO3-SSL sweep evaluated on deterministic none/z/SO3 test rotations.",
            "backbone": "yanx27_pointnet_stn",
            "ssl_aug": field("ssl_rotation", "so3"),
            "supervised_aug": field("train_rotation", "none"),
            "test_aug": field("test_rotation"),
        },
    )
    add_csv(
        rows,
        "eb_jepa_supervised_rotation/results.csv",
        {
            "experiment_id": "matched_supervised_test_rotation",
            "wandb_project": "eb_jepa_supervised_rotation",
            "note": "Four-ratio matched-rotation benchmark: supervised train/val and test use the same none/z/SO3 rotation-only protocol, while SSL always uses two SO3 views.",
            "backbone": "simple_pointnet",
            "ssl_aug": fixed("so3"),
            "supervised_aug": field("augmentation"),
            "test_aug": field("augmentation"),
        },
    )

    class_20_path = ROOT / "eb_jepa_new_class/results.json"
    class_20 = json.loads(class_20_path.read_text())
    append_wide(
        rows,
        class_20_path,
        {
            "experiment_id": "disjoint_class_20_20_clean_test",
            "wandb_project": "eb_jepa_new_class",
            "note": "Disjoint-class transfer: SSL pretrains on all samples from 20 classes; clean supervised train/val/test use the other 20 classes only.",
            "backbone": "simple_pointnet",
            "class_protocol": "20_pretrain_20_supervised_disjoint",
            "pretrain_classes": 20,
            "supervised_classes": 20,
            "ssl_aug": fixed("so3"),
            "supervised_aug": fixed("none"),
            "test_aug": fixed("none"),
            "test_size": class_20["test_size"],
        },
        [class_20],
    )
    add_csv(
        rows,
        "eb_jepa_new_class_10_30/results_new_class_10_30.csv",
        {
            "experiment_id": "disjoint_class_10_30_clean_test",
            "wandb_project": "eb_jepa_new_class_10_30",
            "note": "Disjoint-class transfer clean test for 10 SSL/30 supervised and 30 SSL/10 supervised class splits.",
            "backbone": "simple_pointnet",
            "class_protocol": "variable_disjoint_class_split",
            "ssl_aug": fixed("so3"),
            "supervised_aug": fixed("none"),
            "test_aug": fixed("none"),
        },
    )
    add_csv(
        rows,
        "eb_jepa_new_class_10_30/results_new_class_rotation_test.csv",
        {
            "experiment_id": "disjoint_class_10_20_30_rotation_test",
            "wandb_project": "eb_jepa_new_class_rotation_test",
            "note": "Disjoint-class models for 10/30, 20/20 and 30/10 class splits evaluated on deterministic none/z/SO3 rotations of each supervised-class test subset.",
            "backbone": "simple_pointnet",
            "class_protocol": "variable_disjoint_class_split",
            "ssl_aug": fixed("so3"),
            "supervised_aug": fixed("none"),
            "test_aug": field("test_rotation"),
        },
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in COLUMNS} for row in rows)
    print(f"wrote {len(rows)} rows from 13 tables to {OUTPUT}")


if __name__ == "__main__":
    main()
