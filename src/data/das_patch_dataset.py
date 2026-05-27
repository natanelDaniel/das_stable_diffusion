"""
DASPatchDataset — PyTorch Dataset that yields 2D spatiotemporal patches.

Each patch shape: [1, patch_channels, patch_time]  (grayscale 2D image)
Label: integer class index and one-hot vector.

HDF5 structure:  f["Acquisition"]["Raw[0]"]["RawData"]  shape=[n_time, n_fiber_ch]
Bitmap structure: .npy shape=[n_fiber_ch, n_windows]    value 1 → event present
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from typing import List, Tuple, Optional


CLASSES = [
    "car", "construction", "fence", "longboard",
    "manipulation", "openclose", "regular", "running", "walk",
]
N_CLASSES = len(CLASSES)


class DASPatchDataset(Dataset):
    """
    Loads 2D spatiotemporal patches from the DAS HDF5 dataset.

    Args:
        data_dir:       Root directory containing one sub-folder per class.
        patch_channels: Number of fiber channels per patch (spatial extent).
        patch_time:     Number of time samples per patch.
        shift:          Stride (in samples) for the sliding time window.
        decimation:     Dict[class_name, keep_fraction] — random keep fraction.
        classes:        Ordered list of class names (must match sub-folder names).
        seed:           RNG seed for decimation.
    """

    def __init__(
        self,
        data_dir: str,
        patch_channels: int = 32,
        patch_time: int = 256,
        shift: int = 128,
        decimation: Optional[dict] = None,
        classes: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.patch_channels = patch_channels
        self.patch_time = patch_time
        self.shift = shift
        self.decimation = decimation or {}
        self.classes = classes or CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.rng = np.random.default_rng(seed)

        self.samples: List[Tuple[str, int, int, int]] = []
        # Each entry: (h5_path, window_idx, channel_idx, class_idx)
        self._index_files()

    def _index_files(self):
        """Walk the data directory and index all event positions."""
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            keep_frac = self.decimation.get(class_name, 1.0)

            for h5_path in glob(os.path.join(class_dir, "*.h5")):
                npy_path = h5_path[:-3] + ".npy"
                if not os.path.exists(npy_path):
                    continue

                bitmap = np.load(npy_path)  # [n_fiber_ch, n_windows]
                event_positions = np.argwhere(bitmap)  # [[ch, win], ...]

                if keep_frac < 1.0:
                    n_keep = max(1, int(len(event_positions) * keep_frac))
                    idx = self.rng.choice(len(event_positions), n_keep, replace=False)
                    event_positions = event_positions[idx]

                for ch_idx, win_idx in event_positions:
                    self.samples.append((h5_path, int(win_idx), int(ch_idx), class_idx))

        if not self.samples:
            raise RuntimeError(f"No samples found in {self.data_dir}. Check class folder names.")

    def _load_patch(self, h5_path: str, win_idx: int, ch_idx: int) -> np.ndarray:
        """Extract a 2D patch [patch_channels, patch_time] from the HDF5 file."""
        with h5py.File(h5_path, "r") as f:
            raw = f["Acquisition"]["Raw[0]"]["RawData"]  # [n_time, n_fiber_ch]
            n_time, n_fiber = raw.shape

            # Time slice
            t_start = win_idx * self.shift
            t_end = t_start + self.patch_time
            if t_end > n_time:
                t_start = max(0, n_time - self.patch_time)
                t_end = n_time

            # Channel slice (centred on event channel, clamped to boundaries)
            half_ch = self.patch_channels // 2
            ch_start = max(0, ch_idx - half_ch)
            ch_end = ch_start + self.patch_channels
            if ch_end > n_fiber:
                ch_end = n_fiber
                ch_start = max(0, ch_end - self.patch_channels)

            # Read the patch: shape [time_slice, ch_slice]
            patch = raw[t_start:t_end, ch_start:ch_end]  # numpy slice via h5py
            patch = np.array(patch, dtype=np.float32)

        # Pad if boundary clamp made the patch smaller than expected
        if patch.shape[0] < self.patch_time:
            pad_t = self.patch_time - patch.shape[0]
            patch = np.pad(patch, ((0, pad_t), (0, 0)))
        if patch.shape[1] < self.patch_channels:
            pad_ch = self.patch_channels - patch.shape[1]
            patch = np.pad(patch, ((0, 0), (0, pad_ch)))

        # Transpose to [patch_channels, patch_time]
        return patch.T  # [n_ch, n_time]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        h5_path, win_idx, ch_idx, class_idx = self.samples[idx]
        patch = self._load_patch(h5_path, win_idx, ch_idx)

        # Per-patch z-score normalisation
        mu = patch.mean()
        sigma = patch.std() + 1e-8
        patch = (patch - mu) / sigma

        # Add channel dim: [1, patch_channels, patch_time]
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)

        # One-hot label
        onehot = torch.zeros(N_CLASSES, dtype=torch.float32)
        onehot[class_idx] = 1.0

        return patch_tensor, class_idx, onehot
