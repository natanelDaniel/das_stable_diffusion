"""Tests for DASLatentPatchDataset and the decimation cache pipeline."""

import os
import sys
from glob import glob

import h5py
import numpy as np
import pytest
import torch

# Make the project root importable when running pytest from anywhere
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import (  # noqa: E402
    BITMAP_SHIFT_DEC,
    BITMAP_SHIFT_RAW,
    CLASSES,
    DECIM_FACTOR,
    DEFAULT_PATCH_CH,
    DEFAULT_PATCH_TIME,
    DASLatentPatchDataset,
    NOISE_CLASS,
    NOISE_CLASS_IDX,
    compute_global_stats,
)


@pytest.fixture
def tiny_dataset_root(tmp_path):
    """Build a tiny synthetic dataset with .dec1k.npy caches and bitmaps for two classes.

    Recording is 60 s @ 20 kHz so a 16.384 s patch fits well inside with room to spare
    for the [2 s, 14 s] random event anchoring.
    """
    n_time_raw = 1_200_000   # 60 s @ 20 kHz
    n_fiber = 50
    n_time_dec = n_time_raw // DECIM_FACTOR  # 60 000

    rng = np.random.default_rng(0)

    for class_name in ["car", "regular"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        h5_path = class_dir / f"{class_name}_test.h5"
        cache_path = class_dir / f"{class_name}_test.dec1k.npy"
        bitmap_path = class_dir / f"{class_name}_test.npy"

        # Dummy h5 just so glob() finds it; dataset reads .dec1k.npy
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("Acquisition/Raw[0]")
            grp.create_dataset(
                "RawData",
                data=rng.standard_normal((100, n_fiber), dtype=np.float32),
            )

        dec = rng.standard_normal((n_time_dec, n_fiber)).astype(np.float32)
        np.save(cache_path, dec)

        n_windows = n_time_raw // BITMAP_SHIFT_RAW
        bitmap = np.zeros((n_windows, n_fiber), dtype=bool)
        # Place events at win indices that keep the patch fully inside the recording
        # under any event_offset in [2000, 14000]:
        #   event_center = win_idx * 102.4 + 51.2
        #   t_start = event_center - offset  needs >= 0   -> event_center >= 14000  -> win_idx >= 137
        #   t_end = t_start + 16384         needs <= 60000 -> event_center <= 45616 -> win_idx <= 445
        for _ in range(5):
            w = int(rng.integers(150, n_windows - 200))
            c = int(rng.integers(10, n_fiber - 10))
            bitmap[w, c] = True
        np.save(bitmap_path, bitmap)

    return str(tmp_path)


@pytest.fixture
def tiny_dataset_root_500(tmp_path):
    """Same as tiny_dataset_root but with .dec500.npy caches (500 Hz)."""
    n_time_raw = 1_200_000   # 60 s @ 20 kHz
    n_fiber = 50
    decim_factor_500 = 40
    n_time_dec = n_time_raw // decim_factor_500  # 30 000

    rng = np.random.default_rng(0)

    for class_name in ["car", "regular"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        h5_path = class_dir / f"{class_name}_test.h5"
        cache_path = class_dir / f"{class_name}_test.dec500.npy"
        bitmap_path = class_dir / f"{class_name}_test.npy"

        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("Acquisition/Raw[0]")
            grp.create_dataset(
                "RawData",
                data=rng.standard_normal((100, n_fiber), dtype=np.float32),
            )

        dec = rng.standard_normal((n_time_dec, n_fiber)).astype(np.float32)
        np.save(cache_path, dec)

        n_windows = n_time_raw // BITMAP_SHIFT_RAW
        bitmap = np.zeros((n_windows, n_fiber), dtype=bool)
        for _ in range(5):
            # Same safe-band logic: at 500 Hz, patch_time=2048 with offset [250, 1750]
            # needs event_center >= 1750 and <= n_time_dec - 298 = 29702
            # center = win * 51.2 + 25.6 -> win range ~[34, 580]
            w = int(rng.integers(40, 500))
            c = int(rng.integers(10, n_fiber - 10))
            bitmap[w, c] = True
        np.save(bitmap_path, bitmap)

    return str(tmp_path)


def test_decimation_factor_constant():
    assert DECIM_FACTOR == 20
    assert BITMAP_SHIFT_DEC == BITMAP_SHIFT_RAW / DECIM_FACTOR
    assert BITMAP_SHIFT_DEC == 102.4


def test_noise_class_is_regular():
    assert NOISE_CLASS == "regular"
    assert NOISE_CLASS_IDX == CLASSES.index("regular")
    assert NOISE_CLASS_IDX == 6


def test_class_count_is_nine():
    assert len(CLASSES) == 9


def test_patch_shape(tiny_dataset_root):
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
    )
    patch, label = ds[0]
    assert patch.shape == (1, DEFAULT_PATCH_CH, DEFAULT_PATCH_TIME)
    assert patch.dtype.is_floating_point
    assert isinstance(label, int)
    assert 0 <= label < 2


def test_all_specified_classes_loaded(tiny_dataset_root):
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
    )
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1}


def test_random_anchoring_varies_with_seed(tiny_dataset_root):
    """Different seeds should produce different patch contents for the same sample index."""
    contents = []
    for seed in (10, 20, 30, 40, 50):
        ds = DASLatentPatchDataset(
            data_dir=tiny_dataset_root,
            classes=["car", "regular"],
            seed=seed,
        )
        patch, _ = ds[0]
        contents.append(float(patch.numpy().sum()))
    assert len(set(contents)) > 1


def test_normalization_shifts_distribution(tiny_dataset_root):
    """With cache values ~N(0,1) and normalize=(5, 2), output mean must be roughly -2.5."""
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        normalize=(5.0, 2.0),
    )
    patch, _ = ds[0]
    assert patch.numpy().mean() < -1.0


def test_event_offset_range_validation():
    with pytest.raises(ValueError, match="event_offset_range"):
        DASLatentPatchDataset(
            data_dir="/nonexistent",
            event_offset_range=(0, 20_000),
            patch_time=16_384,
        )


def test_event_offset_range_must_be_ordered():
    with pytest.raises(ValueError, match="event_offset_range"):
        DASLatentPatchDataset(
            data_dir="/nonexistent",
            event_offset_range=(5000, 5000),
            patch_time=16_384,
        )


def test_missing_cache_raises_clear_error(tmp_path):
    """If .dec1k.npy is missing, the error must point to the cache-building script."""
    class_dir = tmp_path / "car"
    class_dir.mkdir()
    # Create an h5 + bitmap but NO .dec1k.npy
    (class_dir / "x.h5").write_bytes(b"\x00")
    np.save(class_dir / "x.npy", np.zeros((10, 10), dtype=bool))
    with pytest.raises(FileNotFoundError, match="build_decimation_cache"):
        DASLatentPatchDataset(data_dir=str(tmp_path), classes=["car"])


def test_compute_global_stats(tiny_dataset_root):
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        normalize=None,
    )
    mu, sigma = compute_global_stats(ds, n_samples=4)
    # Cache is N(0, 1) → stats should land near (0, 1).
    assert abs(mu) < 0.5
    assert 0.5 < sigma < 1.5


def test_compute_global_stats_requires_unnormalized(tiny_dataset_root):
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        normalize=(0.0, 1.0),
    )
    with pytest.raises(ValueError):
        compute_global_stats(ds, n_samples=4)


def test_class_decimation_keep_fraction(tiny_dataset_root):
    """decimation={class: 0.5} should roughly halve that class's sample count."""
    ds_full = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
    )
    ds_half = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        decimation={"car": 0.5, "regular": 1.0},
    )
    n_car_full = sum(1 for s in ds_full.samples if s[3] == 0)
    n_car_half = sum(1 for s in ds_half.samples if s[3] == 0)
    n_reg_full = sum(1 for s in ds_full.samples if s[3] == 1)
    n_reg_half = sum(1 for s in ds_half.samples if s[3] == 1)
    assert n_car_half <= n_car_full
    assert n_reg_half == n_reg_full


def test_bitmap_to_dec_time_mapping(tiny_dataset_root):
    """Default 1 kHz instance: center @ win * 102.4 + 51.2."""
    ds = DASLatentPatchDataset(data_dir=tiny_dataset_root, classes=["car", "regular"])
    assert ds._bitmap_win_to_dec_time(10) == 1075
    assert ds._bitmap_win_to_dec_time(0) == 51


def test_bitmap_to_dec_time_scales_with_sample_rate(tiny_dataset_root_500):
    """At 500 Hz, decim factor 40, bitmap shift = 51.2 dec samples per row."""
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root_500,
        classes=["car", "regular"],
        target_sample_rate=500,
        patch_time=2048,
        event_offset_range=(250, 1750),
    )
    # win=10: center = 10 * 51.2 + 25.6 = 537.6 -> 537
    assert ds._bitmap_win_to_dec_time(10) == 537
    assert ds._bitmap_win_to_dec_time(0) == 25  # 25.6 -> 25


def test_cache_suffix_for_helper():
    from src.data.das_latent_patch_dataset import cache_suffix_for
    assert cache_suffix_for(1000) == ".dec1k.npy"
    assert cache_suffix_for(2000) == ".dec2k.npy"
    assert cache_suffix_for(500) == ".dec500.npy"
    with pytest.raises(ValueError):
        cache_suffix_for(3000)  # doesn't divide 20000
    with pytest.raises(ValueError):
        cache_suffix_for(0)


def test_missing_500hz_cache_message_points_to_target_sr_flag(tmp_path):
    """The FileNotFoundError must reference the SR-specific build command."""
    class_dir = tmp_path / "car"
    class_dir.mkdir()
    (class_dir / "x.h5").write_bytes(b"\x00")
    np.save(class_dir / "x.npy", np.zeros((10, 10), dtype=bool))
    with pytest.raises(FileNotFoundError, match=r"--target_sr 500"):
        DASLatentPatchDataset(
            data_dir=str(tmp_path),
            classes=["car"],
            target_sample_rate=500,
            patch_time=2048,
            event_offset_range=(250, 1750),
        )


def test_mixed_output_shape(tiny_dataset_root):
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        return_mixed=True,
    )
    out = ds[0]
    assert len(out) == 3
    mixed, class_idx, alpha = out
    assert mixed.shape == (1, DEFAULT_PATCH_CH, DEFAULT_PATCH_TIME)
    assert mixed.dtype.is_floating_point
    assert isinstance(class_idx, int)
    assert isinstance(alpha, float)
    assert 0.0 <= alpha <= 1.0


def test_alpha_zero_returns_event(tiny_dataset_root):
    """With alpha_range=(0, 0), the mixed tensor equals the event tensor."""
    ds_mix = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        return_mixed=True,
        mix_alpha_range=(0.0, 0.0),
    )
    ds_plain = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        return_mixed=False,
    )
    mixed, _, alpha = ds_mix[0]
    event, _ = ds_plain[0]
    assert alpha == 0.0
    assert torch.allclose(mixed, event, atol=1e-6)


def test_alpha_one_adds_full_noise(tiny_dataset_root):
    """With alpha_range=(1, 1), mixed - event ≈ noise_patch."""
    ds_mix = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        return_mixed=True,
        mix_alpha_range=(1.0, 1.0),
    )
    ds_plain = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        return_mixed=False,
    )
    mixed, _, alpha = ds_mix[0]
    event, _ = ds_plain[0]
    assert alpha == 1.0
    diff = (mixed - event).squeeze(0).numpy()
    # The difference must be a real patch (non-zero) with finite values
    assert np.isfinite(diff).all()
    assert np.abs(diff).max() > 0


def test_return_mixed_requires_regular_class(tiny_dataset_root):
    with pytest.raises(RuntimeError, match="regular"):
        DASLatentPatchDataset(
            data_dir=tiny_dataset_root,
            classes=["car"],  # missing 'regular'
            return_mixed=True,
        )


def test_cache_in_ram_loads_full_arrays(tiny_dataset_root):
    """With cache_in_ram=True, the cached arrays should be plain ndarrays, not memmaps."""
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
        cache_in_ram=True,
    )
    # Force one __getitem__ so a cache entry exists.
    _ = ds[0]
    assert ds._cache, "expected at least one cache entry"
    for path, arr in ds._cache.items():
        assert not isinstance(arr, np.memmap), f"{path} is memmap, expected in-RAM ndarray"


def test_cache_in_ram_default_is_mmap(tiny_dataset_root):
    """Default (cache_in_ram=False) must keep using np.memmap so RAM usage stays low."""
    ds = DASLatentPatchDataset(
        data_dir=tiny_dataset_root,
        classes=["car", "regular"],
    )
    _ = ds[0]
    for path, arr in ds._cache.items():
        assert isinstance(arr, np.memmap), f"{path} not memmap; default should mmap"


def test_mix_alpha_range_validation():
    with pytest.raises(ValueError, match="mix_alpha_range"):
        DASLatentPatchDataset(
            data_dir="/nonexistent",
            mix_alpha_range=(0.5, 0.2),  # hi < lo
        )
    with pytest.raises(ValueError, match="mix_alpha_range"):
        DASLatentPatchDataset(
            data_dir="/nonexistent",
            mix_alpha_range=(-0.1, 0.5),  # lo < 0
        )
    with pytest.raises(ValueError, match="mix_alpha_range"):
        DASLatentPatchDataset(
            data_dir="/nonexistent",
            mix_alpha_range=(0.0, 1.5),  # hi > 1
        )
