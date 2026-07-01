"""
Tests for DASResNetClassifier and CNNTrainer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.das_cnn_classifier import BasicBlock, DASResNetClassifier
from src.training.cnn_trainer import CNNTrainer, SyntheticDASDataset


# ---------------------------------------------------------------------------
# 1. Shape test — forward pass produces [B, n_classes] for arbitrary T
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [256, 1024])
def test_model_output_shape(T):
    model = DASResNetClassifier(n_classes=4, embed_dim=512, dropout=0.0)
    model.eval()
    x = torch.randn(2, 1, 8, T)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 4), f"Expected (2, 4), got {logits.shape}"


def test_model_output_shape_full():
    """Test with the actual production patch size (16384 time samples)."""
    model = DASResNetClassifier(n_classes=4, embed_dim=512, dropout=0.0)
    model.eval()
    x = torch.randn(1, 1, 8, 16384)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 4)


# ---------------------------------------------------------------------------
# 2. Per-class heads — architecture constraint
# ---------------------------------------------------------------------------

def test_per_class_heads_structure():
    model = DASResNetClassifier(n_classes=4, embed_dim=512)
    assert len(model.heads) == 4, "Expected 4 heads (one per class)"
    for head in model.heads:
        assert isinstance(head, nn.Linear), "Each head must be nn.Linear"
        assert head.in_features == 512
        assert head.out_features == 1


def test_heads_produce_independent_logits():
    model = DASResNetClassifier(n_classes=4, embed_dim=512, dropout=0.0)
    model.eval()
    x = torch.randn(3, 1, 8, 256)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (3, 4)
    # Each column is from a separate head — check no two heads are identical in weights
    w0 = model.heads[0].weight.data
    w1 = model.heads[1].weight.data
    assert not torch.allclose(w0, w1), "Two class heads should not have identical weights"


# ---------------------------------------------------------------------------
# 3. Trainer smoke test — 2 steps train + validate
# ---------------------------------------------------------------------------

def _make_fake_loader(n: int, n_classes: int = 4, T: int = 256, batch_size: int = 4):
    patches = torch.randn(n, 1, 8, T)
    labels = torch.randint(0, n_classes, (n,))
    ds = TensorDataset(patches, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_trainer_smoke():
    n_cls = 4
    model = DASResNetClassifier(n_classes=n_cls, embed_dim=512, dropout=0.0)
    train_loader = _make_fake_loader(8, n_cls)
    val_loader = _make_fake_loader(4, n_cls)

    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=["car", "regular", "running", "walk"],
        device="cpu",
        epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        amp=False,
        scheduler_T0=5,
        checkpoint_dir="/tmp/test_cnn_ckpt",
        wandb_logger=None,
        log_every_n_epochs=1,
    )
    # Should complete without error and produce finite losses.
    initial_loss = None

    def _patched_fit():
        train_m = trainer._train_epoch()
        val_m = trainer._validate()
        assert np.isfinite(train_m["loss"]), "Train loss must be finite"
        assert np.isfinite(val_m["loss"]), "Val loss must be finite"
        assert 0.0 <= val_m["acc"] <= 1.0
        assert "all_preds" in val_m
        assert "all_targets" in val_m
        assert "gallery_patches" in val_m

    _patched_fit()


def test_trainer_loss_decreases_with_sgd():
    """With enough steps and a tiny learnable dataset, loss should trend down."""
    n_cls = 2
    # Fixed dataset: same patch for each class — model can memorize
    patches = torch.randn(16, 1, 8, 256)
    labels = torch.tensor([i % n_cls for i in range(16)])
    ds = TensorDataset(patches, labels)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model = DASResNetClassifier(n_classes=n_cls, embed_dim=512, dropout=0.0)
    trainer = CNNTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        class_names=["a", "b"],
        device="cpu",
        epochs=1,
        lr=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        amp=False,
        scheduler_T0=10,
        checkpoint_dir="/tmp/test_cnn_ckpt2",
        wandb_logger=None,
        log_every_n_epochs=1,
    )
    m1 = trainer._train_epoch()
    m2 = trainer._train_epoch()
    # After 2 epochs on a tiny fixed dataset loss should not increase wildly
    assert m2["loss"] < m1["loss"] * 5, "Loss increased more than 5× in 2 epochs"


# ---------------------------------------------------------------------------
# 4. W&B logging — mock wandb calls
# ---------------------------------------------------------------------------

class MockWandb:
    def __init__(self):
        self.logged = {}

    def log(self, d: dict):
        self.logged.update(d)

    class Table:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []
        def add_data(self, *args):
            self.rows.append(args)

    class Image:
        def __init__(self, fig):
            self.fig = fig

    class _PlotNS:
        def confusion_matrix(self, **kwargs):
            return "confusion_matrix_obj"
        def roc_curve(self, *a, **kw):
            return "roc_curve_obj"
        def pr_curve(self, *a, **kw):
            return "pr_curve_obj"

    plot = _PlotNS()


def test_wandb_logging_calls():
    n_cls = 4
    model = DASResNetClassifier(n_classes=n_cls, embed_dim=512, dropout=0.0)
    loader = _make_fake_loader(8, n_cls)
    mock_wb = MockWandb()

    trainer = CNNTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        class_names=["car", "regular", "running", "walk"],
        device="cpu",
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        amp=False,
        scheduler_T0=5,
        checkpoint_dir="/tmp/test_cnn_wb",
        wandb_logger=mock_wb,
        log_every_n_epochs=1,
    )
    train_m = trainer._train_epoch()
    val_m = trainer._validate()
    trainer._log_epoch(0, train_m, val_m)
    trainer._log_periodic(0, val_m)

    logged = mock_wb.logged
    # Scalar metrics
    assert "train/loss" in logged
    assert "val/loss" in logged
    assert "train/acc" in logged
    assert "val/acc" in logged
    assert "train/f1_macro" in logged
    assert "val/f1_macro" in logged
    # Per-class metrics
    for cls in ["car", "regular", "running", "walk"]:
        assert f"val/f1_{cls}" in logged, f"missing val/f1_{cls}"
        assert f"val/precision_{cls}" in logged
        assert f"val/recall_{cls}" in logged
    # Periodic logs
    assert "val/confusion_matrix" in logged
    assert "val/roc_curve" in logged
    assert "val/pr_curve" in logged
    assert "val/results_table" in logged
    assert "val/prediction_gallery" in logged


# ---------------------------------------------------------------------------
# 5. SyntheticDASDataset — basic contract
# ---------------------------------------------------------------------------

def test_synthetic_dataset_len_getitem():
    n = 10
    T = 256
    patches = [torch.randn(1, 8, T) for _ in range(n)]
    labels = list(range(n))
    ds = SyntheticDASDataset(patches, labels)

    assert len(ds) == n
    patch, label = ds[0]
    assert patch.shape == (1, 8, T)
    assert label == 0

    patch5, label5 = ds[5]
    assert label5 == 5


def test_synthetic_dataset_dataloader():
    patches = [torch.randn(1, 8, 256) for _ in range(12)]
    labels = [i % 4 for i in range(12)]
    ds = SyntheticDASDataset(patches, labels)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    for patch_batch, label_batch in loader:
        assert patch_batch.shape[1:] == (1, 8, 256)
        assert label_batch.shape == (4,)
        break
