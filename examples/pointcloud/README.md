# PointCloud — view-invariant SSL on 3D point clouds (ModelNet40)

**Question.** Can a two-view SSL objective learn a **view-invariant** shape
representation on an *unordered, irregular* modality (3D point clouds), and how
does linear-probe accuracy degrade as we demand more **rotation invariance**
(none → z → SO(3))?

Point clouds have no temporal frames, so the objective is **two-view VICReg** (the
image-JEPA / audio / EEG recipe), *not* a predictive JEPA. The two SSL views are
two independent augmented samplings + rotations of the *same* object, so VICReg
learns a representation invariant to how the object was sampled and oriented.

## Data
**ModelNet40** in the canonical PointNet HDF5 release (`modelnet40_ply_hdf5_2048`):
9840 train / 2468 test shapes, **2048 (x, y, z) points** each, **40 classes**, at
`/lustre/work/pdl17890/udl806719/datasets/modelnet40/modelnet40_ply_hdf5_2048`.
Each shape → two augmented views, each `[3, n_points=1024]`: random **subsample**,
random **rotation** (`rotate = so3 | z | none`), random **scale** (0.8–1.25),
Gaussian **jitter** (σ=0.01), then **unit-sphere normalize** (center + scale).

## Layout
```
eb_jepa/datasets/pointcloud/   dataset.py (provided loader) + data_config.yaml
examples/pointcloud/
  main.py     SSL pretraining — TODO: build_encoder() + build_ssl()
  eval.py     downstream probe — TODO: probe() + metric
  cfgs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — a **PointNet** encoder over `[B, 3, N]`: a shared
   per-point MLP of 1×1 `Conv1d`s (`3→64→64→128→out_dim`) + a symmetric **max-pool**
   → permutation-invariant global feature. Expose `.represent()` and `.out_dim`.
2. `main.py:build_ssl` — the **two-view VICReg** objective: two views →
   `encoder.represent` → eb_jepa `Projector` → eb_jepa `VICRegLoss` (invariance +
   variance + covariance). The invariance term makes the feature *view-invariant*;
   the var/cov terms prevent collapse.
3. `eval.py:probe` — the frozen-feature linear probe → **40-way accuracy** on the
   official ModelNet40 test split, compared to a random-encoder floor (chance 2.5%).

Everything else (data loading, augmentation, training loop, feature extraction) is
provided. Reuse the eb_jepa core (`Projector`, `VICRegLoss`) — do not duplicate.

## Why this track
The max-pool gives the encoder **permutation invariance** for free (point order is
meaningless). **Rotation invariance**, by contrast, is not built in — it must be
*learned* from the augmented views. The expected (well-known) result is that
accuracy drops monotonically `none → z → SO(3)`: the more rotation invariance the
two views demand, the harder the global feature is to keep linearly separable.

## Run
```bash
python -m examples.pointcloud.main --fname examples/pointcloud/cfgs/train.yaml
python -m examples.pointcloud.eval --ckpt <.../latest.pth.tar>
# view-invariance study: rerun pretraining with data.rotate=none and data.rotate=z
```

## Supervised-data ratio sweep

`ratio_sweep.py` deterministically partitions the official training set into a
disjoint VICReg pretraining partition and a stratified supervised partition. The
supervised partition is further split into train/validation, with every class
present in both. For each supervised ratio and each `none|z|so3` augmentation it
compares three models: PointNet trained from scratch, VICReg pretraining followed
by a frozen linear probe, and VICReg pretraining followed by joint encoder+head
fine-tuning. The official test split is always clean.

Fine-tuning reports two optimizer configurations from the same pretrained
checkpoint and fresh classification-head initialization:

- `equal_lr`: encoder LR `1e-3`, head LR `1e-3`;
- `split_lr`: encoder LR `1e-4`, head LR `1e-3`.

```bash
bash scripts/submit_pointcloud_ratio_sweep.sh
```

The default sweep uses supervised ratios `0.10, 0.25, 0.50, 0.75`, a 20% split
of each supervised subset for validation, and 12 SLURM array tasks. Results are
collected under `$EBJEPA_CKPTS/pointcloud/ratio_sweep` as `results.csv` and
`results.md`, and the same table is logged to W&B project `eb_jepa`.

To add fine-tuning to results produced by an older run of this sweep without
repeating VICReg pretraining, run:

```bash
bash scripts/submit_pointcloud_finetune_sweep.sh
```

### Rotated official-test benchmark

After all models are trained, `_argument_test` evaluates scratch, frozen probe,
`equal_lr` fine-tuning, and `split_lr` fine-tuning on the complete official test
set under deterministic rotation-only `none`, `z`, and `so3` views. It does not
replace the original clean-test metrics.

```bash
bash scripts/submit_pointcloud_argument_test.sh <training-array-job-id>
```

The combined outputs are `results_argument_test.csv` and
`results_argument_test.md`. Twelve mirrored W&B runs append `_argument_test` and
log one final scalar per method, using the matching train/test rotation. This
reproduces the original sweep chart structure across four ratios and
`none|z|so3`. A thirteenth summary run stores the full 36-row cross-rotation table
under `results_argument_test`.

## Train without rotation, test rotation only

`test_rotate_only.py` retrains all four ratios with clean supervised
train/validation data. VICReg pretraining still uses the original two-view
pipeline (random sampling, SO(3) rotation, scale, and jitter). It then evaluates
the complete test set with deterministic rotation-only views named
`test_rotate_only_none`, `test_rotate_only_rotation` (z-axis), and
`test_rotate_only_SO3`.

```bash
bash scripts/submit_pointcloud_test_rotate_only.sh
```

Each `(ratio, test method)` has one W&B run and logs the four final model metrics
once. The combined table is written to `results_test_rotate_only.csv` and
`results_test_rotate_only.md`.
