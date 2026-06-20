"""PointCloud dataset — ModelNet40 3D shapes for view-invariant SSL.

The PointNet HDF5 release (``modelnet40_ply_hdf5_2048``): each shape is 2048
(x, y, z) points with a 40-class label. The two SSL views are two INDEPENDENT
augmented samplings of the SAME object — random SO(3) rotation + 1024-pt
subsample + jitter + scale, then unit-sphere normalize — so a two-view objective
(VICReg) learns a VIEW-INVARIANT shape representation.

Data loading is PROVIDED (plumbing). The modelling choices on top of these views
(encoder, SSL objective, probe) live in ``examples/pointcloud/`` and are where the
``# TODO``s are.

``mode="ssl"`` -> ``(v1, v2, label)`` (two augmented views, each ``[3, n_points]``);
``mode="supervised"`` -> ``(x[3, n_points], label)`` (one deterministic clean view).
"""

import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None


@dataclass
class PointCloudConfig:
    data_root: str = (
        "/lustre/work/pdl17890/udl806719/datasets/modelnet40/"
        "modelnet40_ply_hdf5_2048"
    )
    split: str = "train"  # train | test
    mode: str = "ssl"  # ssl (two views) | supervised ((x, y))
    n_classes: int = 40
    n_points: int = 1024
    # SSL augmentations (geometric)
    rotate: str = "so3"  # so3 (full) | z (azimuth only) | none
    jitter: float = 0.01
    scale_lo: float = 0.8
    scale_hi: float = 1.25
    batch_size: int = 128
    num_workers: int = 8


def _rand_rot(rng, mode):
    if mode == "none":
        return np.eye(3, dtype=np.float32)
    if mode == "z":
        a = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    # uniform SO(3) via a random quaternion
    u1, u2, u3 = rng.uniform(size=3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ],
        dtype=np.float64,
    )
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: PointCloudConfig):
        if h5py is None:
            raise ImportError("h5py required for the ModelNet40 HDF5 loader")
        self.cfg = cfg
        files = sorted(
            glob.glob(os.path.join(cfg.data_root, f"ply_data_{cfg.split}*.h5"))
        )
        if not files:
            raise FileNotFoundError(
                f"no ply_data_{cfg.split}*.h5 under {cfg.data_root} — "
                "download the modelnet40_ply_hdf5_2048 release first"
            )
        data, label = [], []
        for p in files:
            with h5py.File(p, "r") as f:
                data.append(f["data"][:].astype(np.float32))  # [n, 2048, 3]
                label.append(f["label"][:].astype(np.int64).reshape(-1))
        self.data = np.concatenate(data, 0)
        self.label = np.concatenate(label, 0)
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize(pc):
        pc = pc - pc.mean(0, keepdims=True)
        scale = np.max(np.linalg.norm(pc, axis=1)) + 1e-6
        return pc / scale

    def _augment(self, pc, rng, rotate=None):
        c = self.cfg
        idx = rng.choice(pc.shape[0], c.n_points, replace=c.n_points > pc.shape[0])
        p = pc[idx]
        p = p @ _rand_rot(rng, c.rotate if rotate is None else rotate).T
        p = p * rng.uniform(c.scale_lo, c.scale_hi)
        p = p + rng.normal(0, c.jitter, size=p.shape).astype(np.float32)
        return self._normalize(p).astype(np.float32)

    def _clean(self, pc):
        idx = np.linspace(0, pc.shape[0] - 1, self.cfg.n_points).astype(int)
        return self._normalize(pc[idx]).astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        pc, y = self.data[i], int(self.label[i])
        if self.cfg.mode == "supervised":
            return torch.from_numpy(self._clean(pc).T), y  # [3, N], label
        # SSL: two independent augmented views of the SAME object -> view invariance
        v1 = torch.from_numpy(self._augment(pc, rng).T)  # [3, N]
        v2 = torch.from_numpy(self._augment(pc, rng).T)  # [3, N]
        return v1, v2, y


@dataclass(frozen=True)
class PointCloudPartitions:
    """Indices into the official ModelNet40 training split."""

    pretrain: np.ndarray
    supervised_train: np.ndarray
    supervised_val: np.ndarray

    def fingerprint(self):
        """Stable identifier used to verify split reproducibility across runs."""
        digest = hashlib.sha256()
        for indices in (self.pretrain, self.supervised_train, self.supervised_val):
            digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return digest.hexdigest()[:16]


def stratified_train_partitions(labels, supervised_ratio, val_ratio, seed):
    """Create deterministic, disjoint per-class pretrain/train/validation splits.

    A single seeded permutation is generated for each class. Supervised examples
    are a prefix of that permutation, making the subsets nested when ratios are
    swept with the same seed. At least one example per class is reserved for each
    of supervised train, supervised validation, and SSL pretraining.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    supervised_ratio = float(supervised_ratio)
    val_ratio = float(val_ratio)
    if not 0.0 < supervised_ratio < 1.0:
        raise ValueError("supervised_ratio must be strictly between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    rng = np.random.default_rng(int(seed))
    pretrain, supervised_train, supervised_val = [], [], []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < 3:
            raise ValueError(f"class {class_id} needs at least three examples")
        class_indices = rng.permutation(class_indices)
        n_supervised = int(round(len(class_indices) * supervised_ratio))
        n_supervised = min(len(class_indices) - 1, max(2, n_supervised))
        n_val = int(round(n_supervised * val_ratio))
        n_val = min(n_supervised - 1, max(1, n_val))

        supervised_val.extend(class_indices[:n_val])
        supervised_train.extend(class_indices[n_val:n_supervised])
        pretrain.extend(class_indices[n_supervised:])

    # Shuffle each final partition without changing its membership.
    return PointCloudPartitions(
        pretrain=rng.permutation(np.asarray(pretrain, dtype=np.int64)),
        supervised_train=rng.permutation(np.asarray(supervised_train, dtype=np.int64)),
        supervised_val=rng.permutation(np.asarray(supervised_val, dtype=np.int64)),
    )


class PointCloudIndexedDataset(torch.utils.data.Dataset):
    """Indexed view with split-specific augmentation behavior.

    ``augmentation`` is one of ``none``, ``no_rotation``, ``z``, or ``so3``.
    ``none`` is a clean deterministic view, while ``no_rotation`` retains random
    sampling/scale/jitter but uses the identity rotation. Validation can use a
    deterministic augmentation per original sample index, while training uses
    worker-seeded random augmentation. Official test data should continue to use
    ``PointCloudDataset(mode='supervised')`` and is therefore always clean.
    """

    def __init__(
        self,
        dataset,
        indices,
        mode,
        augmentation,
        seed=0,
        deterministic_augmentation=False,
        rotation_only=False,
    ):
        if dataset.cfg.split != "train":
            raise ValueError(
                "PointCloudIndexedDataset is only for train-derived splits"
            )
        if mode not in ("ssl", "supervised"):
            raise ValueError("mode must be 'ssl' or 'supervised'")
        if augmentation not in ("none", "no_rotation", "z", "so3"):
            raise ValueError("augmentation must be one of: none, no_rotation, z, so3")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self.augmentation = augmentation
        self.seed = int(seed)
        self.deterministic_augmentation = bool(deterministic_augmentation)
        self.rotation_only = bool(rotation_only)

    def __len__(self):
        return len(self.indices)

    def _rng(self, original_index, view=0):
        if self.deterministic_augmentation:
            sequence = np.random.SeedSequence(
                [self.seed, int(original_index), int(view)]
            )
            return np.random.default_rng(sequence)
        random_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        return np.random.default_rng(random_seed)

    def _view(self, point_cloud, original_index, view=0):
        if self.augmentation == "none":
            return self.dataset._clean(point_cloud)
        rotation = "none" if self.augmentation == "no_rotation" else self.augmentation
        if self.rotation_only:
            point_cloud = self.dataset._clean(point_cloud)
            point_cloud = point_cloud @ _rand_rot(
                self._rng(original_index, view=view), rotation
            ).T
            return self.dataset._normalize(point_cloud).astype(np.float32)
        return self.dataset._augment(
            point_cloud,
            self._rng(original_index, view=view),
            rotate=rotation,
        )

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        point_cloud = self.dataset.data[original_index]
        label = int(self.dataset.label[original_index])
        if self.mode == "ssl":
            view1 = torch.from_numpy(self._view(point_cloud, original_index, view=0).T)
            view2 = torch.from_numpy(self._view(point_cloud, original_index, view=1).T)
            return view1, view2, label
        view = torch.from_numpy(self._view(point_cloud, original_index).T)
        return view, label


class PointCloudRotatedTestDataset(torch.utils.data.Dataset):
    """Deterministic rotation-only views of the complete official test split.

    The underlying point selection and unit-sphere normalization are identical to
    the clean test protocol. ``z`` and ``so3`` then apply one seeded rotation per
    sample; no jitter, scale, or random resampling is introduced. This keeps the
    benchmark focused specifically on rotation robustness.
    """

    def __init__(self, dataset, rotation, seed=0):
        if dataset.cfg.split != "test":
            raise ValueError("PointCloudRotatedTestDataset requires the test split")
        if rotation not in ("none", "z", "so3"):
            raise ValueError("rotation must be one of: none, z, so3")
        self.dataset = dataset
        self.rotation = rotation
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        point_cloud = self.dataset._clean(self.dataset.data[index])
        if self.rotation != "none":
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
            point_cloud = point_cloud @ _rand_rot(rng, self.rotation).T
            point_cloud = self.dataset._normalize(point_cloud).astype(np.float32)
        label = int(self.dataset.label[index])
        return torch.from_numpy(point_cloud.T.copy()), label


def seed_worker(worker_id):
    """Seed NumPy from the deterministic PyTorch DataLoader worker seed."""
    del worker_id
    np.random.seed(torch.initial_seed() % (2**32))


def make_loader(cfg: PointCloudConfig, shuffle=None):
    ds = PointCloudDataset(cfg)
    is_train = cfg.split == "train"
    if shuffle is None:
        shuffle = is_train
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=cfg.mode == "ssl",
        persistent_workers=cfg.num_workers > 0,
    )
