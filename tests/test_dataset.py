import os
import pytest
import torch
from src.data.das_patch_dataset import DASPatchDataset, N_CLASSES

DATA_DIR = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not os.path.isdir(DATA_DIR), reason="DAS dataset not found"
)


@SKIP_IF_NO_DATA
def test_dataset_loads_without_error():
    ds = DASPatchDataset(
        data_dir=DATA_DIR,
        patch_channels=32,
        patch_time=256,
        decimation={c: 0.005 for c in ["car", "walk", "running", "regular",
                                        "construction", "fence", "longboard",
                                        "manipulation", "openclose"]},
    )
    assert len(ds) > 0, "Dataset is empty"


@SKIP_IF_NO_DATA
def test_sample_shapes():
    ds = DASPatchDataset(
        data_dir=DATA_DIR,
        patch_channels=32,
        patch_time=256,
        decimation={c: 0.002 for c in ["car", "walk", "running", "regular",
                                        "construction", "fence", "longboard",
                                        "manipulation", "openclose"]},
    )
    patch, class_idx, onehot = ds[0]
    assert patch.shape == (1, 32, 256), f"Expected (1,32,256), got {patch.shape}"
    assert onehot.shape == (N_CLASSES,), f"Expected ({N_CLASSES},), got {onehot.shape}"
    assert isinstance(class_idx, int)
    assert onehot.sum().item() == 1.0


@SKIP_IF_NO_DATA
def test_patch_is_normalised():
    ds = DASPatchDataset(
        data_dir=DATA_DIR,
        patch_channels=32,
        patch_time=256,
        decimation={c: 0.002 for c in ["car", "walk", "running", "regular",
                                        "construction", "fence", "longboard",
                                        "manipulation", "openclose"]},
    )
    patch, _, _ = ds[0]
    # z-score: mean ≈ 0, std ≈ 1
    assert abs(patch.mean().item()) < 0.5
    assert 0.5 < patch.std().item() < 2.0
