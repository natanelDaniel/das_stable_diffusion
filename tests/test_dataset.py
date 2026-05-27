"""
Dataset review tests.

Covers:
  - Loading without error
  - Patch shapes and normalisation
  - HDF5 metadata / sampling rate
  - All 9 classes present
  - 1663-channel edge case (construction / some car files)
  - Class-distribution imbalance report (informational)
"""

import os
import collections
import pytest
import h5py
import numpy as np
import torch
from src.data.das_patch_dataset import DASPatchDataset, CLASSES, N_CLASSES

DATA_DIR = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not os.path.isdir(DATA_DIR), reason="DAS dataset not found"
)


# ---------------------------------------------------------------------------
# Shared fixture — tiny decimation so indexing is fast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_small():
    return DASPatchDataset(
        data_dir=DATA_DIR,
        patch_channels=32,
        patch_time=256,
        # bitmap_shift uses default (2048) — matches DASDataLoader bitmap creation
        decimation={c: 0.002 for c in CLASSES},
    )


# ---------------------------------------------------------------------------
# 1. Basic load
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_dataset_loads_without_error(ds_small):
    assert len(ds_small) > 0, "Dataset is empty"


# ---------------------------------------------------------------------------
# 2. Patch shapes
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_sample_shapes(ds_small):
    patch, class_idx, onehot = ds_small[0]
    assert patch.shape == (1, 32, 256), f"Expected (1,32,256), got {patch.shape}"
    assert onehot.shape == (N_CLASSES,), f"Expected ({N_CLASSES},), got {onehot.shape}"
    assert isinstance(class_idx, int)
    assert onehot.sum().item() == 1.0


# ---------------------------------------------------------------------------
# 3. z-score normalisation
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_patch_is_normalised(ds_small):
    patch, _, _ = ds_small[0]
    assert abs(patch.mean().item()) < 0.5, f"mean={patch.mean():.4f} too far from 0"
    assert 0.5 < patch.std().item() < 2.0, f"std={patch.std():.4f} outside (0.5, 2)"


# ---------------------------------------------------------------------------
# 4. HDF5 metadata — sampling rate must be 20 000 Hz
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_hdf5_sampling_rate():
    """OutputDataRate should be 20 000 Hz across every class."""
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        h5_files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".h5"))
        assert h5_files, f"No HDF5 files in {cls}"
        path = os.path.join(cls_dir, h5_files[0])
        with h5py.File(path, "r") as f:
            odr = f["Acquisition"]["Raw[0]"].attrs["OutputDataRate"]
        assert odr == 20_000.0, f"[{cls}] OutputDataRate={odr}, expected 20000"


@SKIP_IF_NO_DATA
def test_hdf5_raw_data_shape():
    """
    RawData.shape == (n_time, n_fiber_ch).
    n_fiber_ch is 1700 for most classes, 1663 for construction (known outlier).
    """
    expected_fiber = {"construction": 1663}
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        h5_files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".h5"))
        path = os.path.join(cls_dir, h5_files[0])
        with h5py.File(path, "r") as f:
            shape = f["Acquisition"]["Raw[0]"]["RawData"].shape
        n_time, n_fiber = shape
        assert n_time > 0,  f"[{cls}] n_time={n_time}"
        assert n_fiber > 0, f"[{cls}] n_fiber={n_fiber}"
        exp = expected_fiber.get(cls, 1700)
        assert n_fiber == exp, f"[{cls}] n_fiber={n_fiber}, expected {exp}"


@SKIP_IF_NO_DATA
def test_patch_temporal_coverage():
    """
    256 samples @ 20 000 Hz → 12.8 ms per patch.
    Each bitmap window = 2048 samples = 102.4 ms; the 256-sample patch sits
    centred inside that window.
    """
    SR = 20_000
    patch_time = 256
    bitmap_shift = 2048
    duration_ms = patch_time / SR * 1000
    window_ms   = bitmap_shift / SR * 1000
    assert abs(duration_ms - 12.8) < 1e-6,  f"patch duration={duration_ms} ms"
    assert abs(window_ms - 102.4)  < 1e-6,  f"bitmap window={window_ms} ms"


# ---------------------------------------------------------------------------
# 5. All 9 classes present in indexed samples
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_all_classes_present(ds_small):
    found = {CLASSES[s[3]] for s in ds_small.samples}
    missing = set(CLASSES) - found
    assert not missing, f"Classes missing from dataset: {missing}"


# ---------------------------------------------------------------------------
# 6. 1663-channel edge case — patch still comes out (1, 32, 256)
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_narrow_fiber_patch_shape():
    """construction files have only 1663 channels; padding must still give (1,32,256)."""
    ds_c = DASPatchDataset(
        data_dir=DATA_DIR,
        patch_channels=32,
        patch_time=256,
        decimation={"construction": 0.01},
        classes=["construction"],
    )
    assert len(ds_c) > 0
    patch, cls_idx, onehot = ds_c[0]
    assert patch.shape == (1, 32, 256), f"Got {patch.shape}"


# ---------------------------------------------------------------------------
# 7. Class distribution report (printed, not asserted)
# ---------------------------------------------------------------------------

@SKIP_IF_NO_DATA
def test_class_distribution_report(ds_small, capsys):
    """
    Informational print — will NOT fail.
    Checks only that no class has 0 samples.
    """
    counts = collections.Counter(CLASSES[s[3]] for s in ds_small.samples)
    total = sum(counts.values())
    max_c = max(counts.values())
    min_c = min(counts.values())

    print("\n── Class distribution (0.2 % decimation) ──")
    for cls in CLASSES:
        bar = "█" * (counts[cls] * 40 // max_c)
        print(f"  {cls:>14s}: {counts[cls]:5d}  {bar}")
    print(f"\n  Total        : {total}")
    print(f"  Imbalance    : {max_c / max(min_c, 1):.1f}×  (max/min)")

    assert min_c > 0, f"A class has 0 samples: {counts}"
