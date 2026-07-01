"""Tests for src/evaluation/plotting.py and the W&B image-logging hooks."""

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.plotting import (  # noqa: E402
    SPEC_DEFAULT_FREQ_MAX_HZ,
    compute_spectrogram_db,
    plot_input_vs_recon,
    plot_multi_waterfall,
    plot_patch_panel,
    plot_per_channel_spectrograms,
    save_figure,
)


def test_compute_spectrogram_db_shape():
    sig = np.random.randn(16_384).astype(np.float32)
    f, t, S = compute_spectrogram_db(sig, fs=1000, freq_max=200)
    assert f.ndim == 1 and t.ndim == 1 and S.ndim == 2
    assert S.shape == (f.size, t.size)
    assert f.max() <= SPEC_DEFAULT_FREQ_MAX_HZ
    # nperseg=1024 noverlap=1008 -> hop=16 -> roughly (16384 - 1024) / 16 + 1 ≈ 961 cols
    assert 900 < t.size < 1100


def test_compute_spectrogram_requires_1d():
    with pytest.raises(ValueError):
        compute_spectrogram_db(np.zeros((2, 100)), fs=1000)


def test_plot_patch_panel_returns_figure():
    patch = np.random.randn(1, 8, 4096).astype(np.float32)
    fig = plot_patch_panel(patch, title="hello")
    assert isinstance(fig, plt.Figure)
    # Two axes: waterfall + spectrogram
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_patch_panel_accepts_2d_input():
    patch = np.random.randn(8, 4096).astype(np.float32)
    fig = plot_patch_panel(patch)
    plt.close(fig)


def test_plot_input_vs_recon_grid():
    x = np.random.randn(1, 8, 4096).astype(np.float32)
    y = x + 0.1 * np.random.randn(*x.shape).astype(np.float32)
    fig = plot_input_vs_recon(x, y, title="ep1")
    # 2x2 grid -> 4 axes
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_input_vs_recon_rejects_shape_mismatch():
    x = np.random.randn(1, 8, 4096).astype(np.float32)
    y = np.random.randn(1, 8, 2048).astype(np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        plot_input_vs_recon(x, y)


def test_plot_multi_waterfall_returns_n_axes():
    patches = [np.random.randn(8, 2048).astype(np.float32) for _ in range(4)]
    titles = ["real", "gen1", "gen2", "gen3"]
    fig = plot_multi_waterfall(patches, titles)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_per_channel_spectrograms_grid_shape():
    """4 rows (1 real + 3 gen) x 8 channel cols = 32 axes."""
    patches = [np.random.randn(8, 2048).astype(np.float32) for _ in range(4)]
    titles = ["real", "gen1", "gen2", "gen3"]
    fig = plot_per_channel_spectrograms(patches, titles)
    assert len(fig.axes) == 4 * 8
    plt.close(fig)


def test_save_figure_writes_file(tmp_path):
    fig = plot_patch_panel(np.random.randn(8, 1024).astype(np.float32), title="t")
    out = tmp_path / "wf.png"
    save_figure(fig, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_vae_trainer_logs_recon_image_when_wandb_present(monkeypatch):
    """Hooked into validate(): with wandb_logger set, _log_recon_image must call wandb.log with an Image."""
    import torch
    from src.models.das_vae_v2 import DASVAEv2
    from src.training.vae_trainer import DASVAEv2Trainer

    model = DASVAEv2(
        in_channels=1,
        encoder_channels=(4, 8, 8, 8),
        latent_channels=2,
        temporal_strides=(4, 4, 4),
    )

    class FakeImage:
        def __init__(self, fig):
            self.fig = fig

    class FakeWandb:
        def __init__(self):
            self.calls = []
            self.Image = FakeImage

        def log(self, payload):
            self.calls.append(payload)

    wandb = FakeWandb()
    trainer = DASVAEv2Trainer(
        model=model,
        train_loader=None,
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        wandb_logger=wandb,
    )
    # Pretend validate() cached a patch
    trainer._fixed_val_patch = torch.randn(1, 1, 8, 1024)
    trainer._log_recon_image(epoch=0)
    image_calls = [c for c in wandb.calls if "val/recon" in c]
    assert image_calls, "expected wandb.log to be called with val/recon"
    img = image_calls[0]["val/recon"]
    assert isinstance(img, FakeImage)
