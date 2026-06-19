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
``mode="supervised"`` -> ``(x[3, n_points], label)``. Supervised training can use
one augmented view by setting ``augment_supervised=True``; evaluation stays clean.
"""
import glob
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
    data_root: str = ("/lustre/work/pdl17890/udl806719/datasets/modelnet40/"
                      "modelnet40_ply_hdf5_2048")
    split: str = "train"            # train | test
    mode: str = "ssl"               # ssl (two views) | supervised ((x, y))
    n_classes: int = 40
    n_points: int = 1024
    augment_supervised: bool = False # augment train split in supervised mode
    # Geometric augmentations
    rotate: str = "so3"             # so3 (full) | z (azimuth only) | none
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
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)], dtype=np.float64)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: PointCloudConfig):
        if h5py is None:
            raise ImportError("h5py required for the ModelNet40 HDF5 loader")
        self.cfg = cfg
        files = sorted(glob.glob(os.path.join(cfg.data_root, f"ply_data_{cfg.split}*.h5")))
        if not files:
            raise FileNotFoundError(
                f"no ply_data_{cfg.split}*.h5 under {cfg.data_root} — "
                "download the modelnet40_ply_hdf5_2048 release first")
        data, label = [], []
        for p in files:
            with h5py.File(p, "r") as f:
                data.append(f["data"][:].astype(np.float32))      # [n, 2048, 3]
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

    def _augment(self, pc, rng):
        c = self.cfg
        idx = rng.choice(pc.shape[0], c.n_points, replace=c.n_points > pc.shape[0])
        p = pc[idx]
        p = p @ _rand_rot(rng, c.rotate).T
        p = p * rng.uniform(c.scale_lo, c.scale_hi)
        p = p + rng.normal(0, c.jitter, size=p.shape).astype(np.float32)
        return self._normalize(p).astype(np.float32)

    def _clean(self, pc):
        idx = np.linspace(0, pc.shape[0] - 1, self.cfg.n_points).astype(int)
        return self._normalize(pc[idx]).astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        pc, y = self.data[i], int(self.label[i])
        if self.cfg.mode == "supervised":
            use_augmentation = self.cfg.augment_supervised and self.cfg.split == "train"
            view = self._augment(pc, rng) if use_augmentation else self._clean(pc)
            return torch.from_numpy(view.T), y                       # [3, N], label
        # SSL: two independent augmented views of the SAME object -> view invariance
        v1 = torch.from_numpy(self._augment(pc, rng).T)              # [3, N]
        v2 = torch.from_numpy(self._augment(pc, rng).T)              # [3, N]
        return v1, v2, y


def make_loader(cfg: PointCloudConfig, shuffle=None):
    ds = PointCloudDataset(cfg)
    is_train = cfg.split == "train"
    if shuffle is None:
        shuffle = is_train
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=cfg.mode == "ssl",
        persistent_workers=cfg.num_workers > 0)
