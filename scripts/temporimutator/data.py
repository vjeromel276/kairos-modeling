"""
PyTorch Dataset wrappers for Temporimutator artifacts.

Uses mmap-backed numpy loads so the 547 MB train_sequences.npy doesn't have to
sit in RAM. Per-item fetch copies a 252×4 slice into a tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]


class TMDataset(Dataset):
    """
    One row per window. Returns:
      sequence:   (252, 4) float32
      vol_scalar: (1,)     float32
      dir_label:  ()       int64    — {0=down, 1=flat, 2=up}
      force:     ()        float32
      idx:       ()        int64    — index into the parquet meta file
    """

    def __init__(self, data_dir: str | Path, split: Split, horizon: int):
        self.data_dir = Path(data_dir)
        self.split = split
        self.horizon = horizon

        self._sequences = np.load(self.data_dir / f"{split}_sequences.npy", mmap_mode="r")
        self._scalars = np.load(self.data_dir / f"{split}_scalars.npy", mmap_mode="r")
        self._dir = np.load(self.data_dir / f"{split}_labels_dir_{horizon}d.npy", mmap_mode="r")
        self._force = np.load(self.data_dir / f"{split}_labels_force_{horizon}d.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        # mmap-backed arrays are read-only; .copy() makes a writable buffer so
        # PyTorch's from_numpy doesn't emit a non-writable-tensor warning.
        seq = torch.from_numpy(np.array(self._sequences[idx], dtype=np.float32, copy=True))
        vs = torch.from_numpy(np.array(self._scalars[idx], dtype=np.float32, copy=True))
        y_dir = torch.tensor(int(self._dir[idx]), dtype=torch.long)
        y_force = torch.tensor(float(self._force[idx]), dtype=torch.float32)
        return seq, vs, y_dir, y_force, idx


def class_counts(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    out = np.zeros(n_classes, dtype=np.int64)
    vals, cnts = np.unique(labels, return_counts=True)
    for v, c in zip(vals, cnts):
        out[int(v)] = int(c)
    return out


def inverse_freq_weights(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Weight c = N / (K * n_c), the standard scikit 'balanced' formula."""
    counts = class_counts(labels, n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero
    w = counts.sum() / (n_classes * counts)
    return w.astype(np.float32)
