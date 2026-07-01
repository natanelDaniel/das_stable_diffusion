"""
DASLatentPatchDataset — yields decimated 1 kHz DAS patches for latent diffusion training.

Each patch shape: [1, 8, 16384]  (~8 m x 16.384 s @ 1 kHz)
Reads pre-decimated *.dec1k.npy caches built by scripts/build_decimation_cache.py.

The 9th class `regular` plays the role of "noise" in mixed conditioning at sampling time.
No special handling at the dataset level — `regular` is just one of the 9 classes (index 6).

Key differences from DASPatchDataset:
  * 1 kHz instead of 20 kHz (decimation cache)
  * 8 channels x 16384 samples instead of 32 x 256
  * Random event anchoring within the patch (event placed somewhere in [2s, 14s])
  * Global normalization with (mean, std) supplied via config, NOT per-patch z-score
  * Returns (patch, class_idx) only — soft / mixed conditioning lives at sampling time
"""

import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


CLASSES = [
    "car", "construction", "fence", "longboard",
    "manipulation", "openclose", "regular", "running", "walk",
]
N_CLASSES = len(CLASSES)
NOISE_CLASS = "regular"
NOISE_CLASS_IDX = CLASSES.index(NOISE_CLASS)

ORIG_SR_HZ = 20_000
TARGET_SR_HZ = 1_000
DECIM_FACTOR = ORIG_SR_HZ // TARGET_SR_HZ            # 20
BITMAP_SHIFT_RAW = 2048                              # raw samples per bitmap row
BITMAP_SHIFT_DEC = BITMAP_SHIFT_RAW / DECIM_FACTOR   # 102.4 dec samples per bitmap row

DEFAULT_PATCH_CH = 8
DEFAULT_PATCH_TIME = 16_384                          # 16.384 s @ 1 kHz
DEFAULT_EVENT_OFFSET_RANGE = (2_000, 14_000)         # event center placed in [2s, 14s] within patch
CACHE_SUFFIX = ".dec1k.npy"                          # default for 1 kHz cache


def cache_suffix_for(sample_rate_hz: int) -> str:
    """File suffix used for the decimated cache at the given target rate."""
    if sample_rate_hz <= 0 or ORIG_SR_HZ % sample_rate_hz != 0:
        raise ValueError(
            f"target_sample_rate {sample_rate_hz} must divide ORIG_SR_HZ ({ORIG_SR_HZ})."
        )
    if sample_rate_hz % 1000 == 0:
        return f".dec{sample_rate_hz // 1000}k.npy"     # .dec1k.npy, .dec2k.npy
    return f".dec{sample_rate_hz}.npy"                  # .dec500.npy


class DASLatentPatchDataset(Dataset):
    """Latent-diffusion dataset: random-anchored 8x16384 patches at 1 kHz with class_idx.

    Args:
        data_dir:           Root with class subfolders.
        patch_channels:     Fiber channels per patch.
        patch_time:         Decimated time samples per patch (seconds * 1000).
        event_offset_range: (low, high) where the labeled event center lands within the
                            patch (in decimated samples). Random per __getitem__ call.
        decimation:         {class_name: keep_fraction} for class-balancing the index.
                            Note: "decimation" here is dataset subsampling, NOT signal
                            decimation (which is handled by the .dec1k.npy cache).
        classes:            Class names; defaults to module-level CLASSES.
        normalize:          (mean, std) for global normalization, applied as (x-mu)/sigma.
                            None disables normalization.
        seed:               RNG seed.
        return_mixed:       If True, __getitem__ returns (mixed_patch, class_idx, alpha)
                            where mixed_patch = event_patch + alpha * noise_patch with
                            noise drawn from the `regular` class. If False, returns
                            (event_patch, class_idx). VAE training uses False; diffusion
                            training uses True.
        mix_alpha_range:    Uniform draw range for alpha when return_mixed=True.
    """

    def __init__(
        self,
        data_dir: str,
        patch_channels: int = DEFAULT_PATCH_CH,
        patch_time: int = DEFAULT_PATCH_TIME,
        event_offset_range: Tuple[int, int] = DEFAULT_EVENT_OFFSET_RANGE,
        decimation: Optional[Dict[str, float]] = None,
        classes: Optional[List[str]] = None,
        normalize: Optional[Tuple[float, float]] = None,
        seed: int = 42,
        return_mixed: bool = False,
        mix_alpha_range: Tuple[float, float] = (0.0, 1.0),
        cache_in_ram: bool = False,
        target_sample_rate: int = TARGET_SR_HZ,
    ):
        if not (0 <= event_offset_range[0] < event_offset_range[1] <= patch_time):
            raise ValueError(
                f"event_offset_range {event_offset_range} must satisfy "
                f"0 <= lo < hi <= patch_time ({patch_time})"
            )
        if not (0.0 <= mix_alpha_range[0] <= mix_alpha_range[1] <= 1.0):
            raise ValueError(
                f"mix_alpha_range {mix_alpha_range} must satisfy "
                f"0 <= lo <= hi <= 1"
            )

        self.data_dir = data_dir
        self.patch_channels = patch_channels
        self.patch_time = patch_time
        self.event_offset_lo, self.event_offset_hi = event_offset_range
        self.decimation = decimation or {}
        self.classes = classes or CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.normalize = normalize
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.return_mixed = return_mixed
        self.mix_alpha_lo, self.mix_alpha_hi = mix_alpha_range
        self.cache_in_ram = cache_in_ram
        # Per-dataset target sample rate. Drives which .dec*.npy cache file is read and
        # how bitmap-window indices map to time samples (bitmap shift is in RAW samples).
        self.target_sample_rate = int(target_sample_rate)
        self.decim_factor = ORIG_SR_HZ // self.target_sample_rate
        self.bitmap_shift_dec = BITMAP_SHIFT_RAW / self.decim_factor
        self.cache_suffix = cache_suffix_for(self.target_sample_rate)

        # (cache_path, win_idx, ch_idx, class_idx)
        self.samples: List[Tuple[str, int, int, int]] = []
        # Subset of samples drawn from the `regular` (noise) class. Used by return_mixed.
        self._noise_samples: List[Tuple[str, int, int, int]] = []
        # cache_path -> np.memmap [n_time_dec, n_fiber]
        self._cache: Dict[str, np.ndarray] = {}

        self._index_files()

        if self.return_mixed and not self._noise_samples:
            raise RuntimeError(
                f"return_mixed=True but no `{NOISE_CLASS}` samples were indexed. "
                f"Ensure `{NOISE_CLASS}` is in `classes` and its bitmap is present."
            )

    def _index_files(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            keep_frac = self.decimation.get(class_name, 1.0)

            for h5_path in sorted(glob(os.path.join(class_dir, "*.h5"))):
                cache_path = h5_path[:-3] + self.cache_suffix
                npy_path = h5_path[:-3] + ".npy"
                if not os.path.exists(cache_path):
                    raise FileNotFoundError(
                        f"No decimation cache for {h5_path} at SR={self.target_sample_rate} Hz "
                        f"(expected {cache_path}). "
                        f"Run scripts/build_decimation_cache.py --target_sr {self.target_sample_rate} first."
                    )
                if not os.path.exists(npy_path):
                    continue

                bitmap = np.load(npy_path)  # [n_windows, n_fiber]
                event_positions = np.argwhere(bitmap)

                if keep_frac < 1.0 and len(event_positions) > 0:
                    n_keep = max(1, int(len(event_positions) * keep_frac))
                    idx = self.rng.choice(len(event_positions), n_keep, replace=False)
                    event_positions = event_positions[idx]

                for win_idx, ch_idx in event_positions:
                    sample = (cache_path, int(win_idx), int(ch_idx), class_idx)
                    self.samples.append(sample)
                    if class_name == NOISE_CLASS:
                        self._noise_samples.append(sample)

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {self.data_dir}. "
                f"Check class folders, bitmap .npy files, and .dec1k.npy caches."
            )

    def _get_cache(self, path: str) -> np.ndarray:
        if path not in self._cache:
            # cache_in_ram=True loads the full .dec1k.npy into RAM (zero disk I/O after
            # first touch). With ~6 GB of caches and 32+ GB RAM this is the right knob
            # when training data lives on an HDD. Set num_workers=0 when using this to
            # avoid duplicating the cache in every worker process.
            mode = None if self.cache_in_ram else "r"
            self._cache[path] = np.load(path, mmap_mode=mode)
        return self._cache[path]

    def _bitmap_win_to_dec_time(self, win_idx: int) -> int:
        """Center of the bitmap event window in decimated samples.

        Uses the dataset's own bitmap_shift_dec so the calculation is correct at any SR.
        """
        return int(win_idx * self.bitmap_shift_dec + self.bitmap_shift_dec / 2)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_patch(
        self,
        sample: Tuple[str, int, int, int],
        local_rng: np.random.Generator,
    ) -> np.ndarray:
        """Load and normalize a single patch from one (cache_path, win, ch, class_idx) sample.

        Returns a float32 array of shape [patch_channels, patch_time], post-normalization.
        Uses `local_rng` to draw the temporal anchor offset.
        """
        cache_path, win_idx, ch_idx, _class_idx = sample
        dec = self._get_cache(cache_path)
        n_time_dec, n_fiber = dec.shape

        event_offset = int(local_rng.integers(self.event_offset_lo, self.event_offset_hi))

        event_center = self._bitmap_win_to_dec_time(win_idx)
        t_start = event_center - event_offset
        t_end = t_start + self.patch_time

        if t_start < 0:
            t_start = 0
            t_end = self.patch_time
        if t_end > n_time_dec:
            t_end = n_time_dec
            t_start = max(0, t_end - self.patch_time)

        half_ch = self.patch_channels // 2
        ch_start = max(0, ch_idx - half_ch)
        ch_end = ch_start + self.patch_channels
        if ch_end > n_fiber:
            ch_end = n_fiber
            ch_start = max(0, ch_end - self.patch_channels)

        patch = np.array(dec[t_start:t_end, ch_start:ch_end], dtype=np.float32)  # [T, C]

        if patch.shape[0] < self.patch_time:
            patch = np.pad(patch, ((0, self.patch_time - patch.shape[0]), (0, 0)))
        if patch.shape[1] < self.patch_channels:
            patch = np.pad(patch, ((0, 0), (0, self.patch_channels - patch.shape[1])))

        patch = patch.T  # [C, T]

        if self.normalize is not None:
            mu, sigma = self.normalize
            patch = (patch - mu) / (sigma + 1e-8)

        return patch

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        class_idx = sample[3]

        # Per-sample RNG so two epochs over the same idx get different anchorings.
        event_rng = np.random.default_rng(self.seed + idx)
        event_patch = self._load_patch(sample, event_rng)  # [C, T]
        event_tensor = torch.from_numpy(event_patch).unsqueeze(0)  # [1, C, T]

        if not self.return_mixed:
            return event_tensor, class_idx

        # Mix path: draw a noise patch + alpha and synthesize mixed = event + alpha * noise
        mix_rng = np.random.default_rng(self.seed + idx + 1_000_003)  # large offset to decorrelate
        noise_sample = self._noise_samples[
            int(mix_rng.integers(0, len(self._noise_samples)))
        ]
        noise_patch = self._load_patch(noise_sample, mix_rng)
        alpha = float(mix_rng.uniform(self.mix_alpha_lo, self.mix_alpha_hi))

        mixed = event_patch + alpha * noise_patch
        mixed_tensor = torch.from_numpy(mixed).unsqueeze(0)  # [1, C, T]
        return mixed_tensor, class_idx, alpha


def compute_global_stats(
    dataset: DASLatentPatchDataset,
    n_samples: int = 256,
    seed: int = 42,
) -> Tuple[float, float]:
    """Sample patches from an un-normalized dataset and return (mean, std) over all values.

    The dataset must be constructed with normalize=None. Uses straightforward
    sum / sum-of-squares accumulation in float64.
    """
    if dataset.normalize is not None:
        raise ValueError("Dataset must be constructed with normalize=None to compute global stats.")

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(dataset))
    idx_sample = rng.choice(len(dataset), size=n_samples, replace=False)

    total_sum = 0.0
    total_sq = 0.0
    total_n = 0
    for i in idx_sample:
        patch, _ = dataset[int(i)]
        x = patch.numpy().astype(np.float64).reshape(-1)
        total_sum += float(x.sum())
        total_sq += float((x * x).sum())
        total_n += x.size
    mean = total_sum / total_n
    var = total_sq / total_n - mean * mean
    return float(mean), float(np.sqrt(max(var, 0.0)))
