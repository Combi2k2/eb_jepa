# Point-cloud SSL experiment notes

This document records the ModelNet40 experiments added on branch `trung`, their
data protocol, model variants, W&B logging, and Slurm entrypoints.

## Dataset and deterministic partitions

The experiments use the canonical PointNet ModelNet40 HDF5 release:

```text
/lustre/work/pdl17890/udl806719/datasets/modelnet40/modelnet40_ply_hdf5_2048
```

The official training split is partitioned independently inside each class. A
single `split_seed` determines the permutation for every class. The resulting
partitions are:

- `pretrain`: unlabeled data used by VICReg;
- `supervised_train`: labeled data used by scratch training, linear probing,
  and fine-tuning;
- `supervised_val`: labeled validation data used for checkpoint selection.

The supervised fractions are `0.10`, `0.25`, `0.50`, and `0.75`; 20% of each
supervised fraction is assigned to validation. At least one sample from every
one of the 40 classes is retained in both supervised train and validation. The
same seed produces the same indices and a `split_fingerprint` is saved in each
result. The official test split is never included in these partitions.

This is class-stratified coverage, not exact class balancing: ModelNet40 itself
has unequal class counts, and each class contributes the same fraction of its
own samples.

## Training methods reported

Every task trains and reports four methods on the same supervised partition:

1. `scratch`: encoder and one linear classification layer trained from random
   initialization;
2. `pretrained_probe`: VICReg pretraining followed by a frozen encoder and a
   fresh linear probe;
3. `finetune_equal_lr`: pretrained encoder and linear head jointly optimized,
   both with learning rate `1e-3`;
4. `finetune_split_lr`: pretrained encoder at `1e-4` and linear head at `1e-3`.

The classification head is always exactly `Linear(1024, 40)`. VICReg uses a
`1024-2048-2048` projector, standard deviation coefficient 25, covariance
coefficient 1, AdamW, and two independently generated views of each pretraining
sample.

## PointNet backbones

`examples/pointcloud/ratio_sweep.py` supports two backbones selected by
`model.backbone`:

- `simple`: the repository's original shared point MLP
  `3 -> 64 -> 64 -> 128 -> 1024`, followed by global max pooling;
- `yanx27`: the PointNet encoder structure used by
  `yanx27/Pointnet_Pointnet2_pytorch`, including input STN, optional feature STN,
  `3 -> 64 -> 128 -> 1024` point convolutions, global max pooling, and feature
  transform regularization.

The yanx27 experiment deliberately does not use that repository's multi-layer
classification head. It retains the single linear layer described above.

## Experiment A: base ratio and cross-rotation sweeps

Config: `examples/pointcloud/cfgs/ratio_sweep.yaml`

This is the original ratio-sweep implementation. It supports supervised
augmentation `none`, `z`, and `so3`, clean official-test evaluation, additional
fine-tuning, and a deterministic cross-rotation test benchmark. Outputs include
JSON per task, CSV/Markdown aggregate tables, model checkpoints, and W&B tables.

Entrypoints:

```bash
bash scripts/submit_pointcloud_ratio_sweep.sh
bash scripts/submit_pointcloud_finetune_sweep.sh
bash scripts/submit_pointcloud_argument_test.sh <training-array-job-id>
```

## Experiment B: clean supervised data and rotated test data

Config: `examples/pointcloud/cfgs/test_rotate_only.yaml`

Protocol:

- backbone: `yanx27` PointNet;
- supervised train and validation: clean, with no rotation augmentation;
- VICReg pretraining: two SO(3)-augmented views;
- test: three deterministic rotation-only versions of the same official test
  set (`none`, z-axis rotation, and SO(3));
- W&B project: `eb_jepa_new_pointnet`.

Run:

```bash
bash scripts/submit_pointcloud_test_rotate_only.sh
```

The submission creates a four-task training array, a clean-result report, a
four-task rotation evaluation array, and a final rotated-test report.

## Experiment C: matched supervised and test rotation

Config: `examples/pointcloud/cfgs/supervised_rotation.yaml`

Protocol:

- backbone: original `simple` PointNet;
- four supervised ratios and three matched rotation cases, giving 12 tasks;
- `none`: supervised train/validation and test are all unrotated;
- `z`: supervised train uses randomly sampled z rotations, while validation and
  test use deterministic per-sample z rotations;
- `so3`: supervised train uses randomly sampled SO(3) rotations, while
  validation and test use deterministic per-sample SO(3) rotations;
- supervised rotation is rotation-only: it does not add random scale, jitter,
  or point resampling;
- VICReg always retains its normal two-view SO(3) augmentation, including its
  existing sampling, scale, and jitter operations;
- W&B project: `eb_jepa_supervised_rotation`.

Run:

```bash
bash scripts/submit_pointcloud_supervised_rotation.sh
```

At the time this note was written, the active corrected jobs are:

```text
training array: 76307
report job:     76308
```

The previous jobs `76284` and `76285` were canceled after a linear-probe
`NameError`. The probe now correctly optimizes cross-entropy using only the
unfrozen linear head; a local smoke test was run before resubmission.

## W&B metrics and tables

Per-task runs log epoch-level train/validation metrics and final values under
`final/`, including:

```text
final/scratch_test_acc
final/pretrained_probe_test_acc
final/finetune_equal_lr_test_acc
final/finetune_split_lr_test_acc
```

Collector jobs upload aggregate `wandb.Table` objects and save CSV and Markdown
copies under the experiment checkpoint directory. W&B chart step axes are only
meaningful for epoch-level training metrics; the final accuracy values are
single scalar summaries.

## Reproducibility details

- Data partition membership is controlled by `sweep.split_seed`.
- DataLoader order is controlled by explicit `torch.Generator` seeds.
- Worker NumPy state is initialized from the PyTorch worker seed.
- Rotated validation and test samples derive their rotation from
  `(split_seed, original_sample_index)`, so repeated runs with the same seed see
  identical evaluation data.
- Supervised training rotations intentionally change between samples/epochs but
  remain reproducible given the same global and DataLoader seeds.
- Checkpoints and Slurm logs are runtime artifacts and are not committed.
