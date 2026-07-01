"""Tests for DASVAEv2: forward shapes, gradient flow, and loss components."""

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.das_vae_v2 import (  # noqa: E402
    DASVAEv2,
    multi_scale_stft_loss,
    vae_loss,
)


# Use a small temporal extent for fast tests: 1024 samples -> latent W = 1024/64 = 16
PATCH_CH = 8
PATCH_T = 1024
LATENT_C = 4
LATENT_T = PATCH_T // 64  # 3 stride-4 blocks


@pytest.fixture
def model():
    return DASVAEv2(
        in_channels=1,
        encoder_channels=(32, 64, 64, 64),
        latent_channels=LATENT_C,
        temporal_strides=(4, 4, 4),
        kernel_time=7,
        kernel_space=3,
    )


def test_forward_shapes(model):
    x = torch.randn(2, 1, PATCH_CH, PATCH_T)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape == (2, LATENT_C, PATCH_CH, LATENT_T)
    assert logvar.shape == (2, LATENT_C, PATCH_CH, LATENT_T)


def test_encode_decode_roundtrip(model):
    x = torch.randn(2, 1, PATCH_CH, PATCH_T)
    mu, _logvar = model.encode(x)
    x_hat = model.decode(mu)
    assert x_hat.shape == x.shape


def test_reparameterize_is_stochastic(model):
    mu = torch.zeros(2, LATENT_C, PATCH_CH, LATENT_T)
    logvar = torch.zeros(2, LATENT_C, PATCH_CH, LATENT_T)  # std=1
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)


def test_gradient_flow(model):
    x = torch.randn(2, 1, PATCH_CH, PATCH_T)
    x_hat, mu, logvar = model(x)
    loss, _ = vae_loss(x, x_hat, mu, logvar, beta=1e-3)
    loss.backward()
    # Every learnable parameter should have a non-None grad
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_stft_loss_zero_when_identical():
    x = torch.randn(2, 1, PATCH_CH, PATCH_T)
    loss = multi_scale_stft_loss(x, x.clone(), n_ffts=(256, 1024))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_stft_loss_positive_when_different():
    x = torch.randn(2, 1, PATCH_CH, PATCH_T)
    y = torch.randn(2, 1, PATCH_CH, PATCH_T)
    loss = multi_scale_stft_loss(x, y, n_ffts=(256, 1024))
    assert loss.item() > 0.0


def test_stft_loss_skips_window_larger_than_signal():
    x = torch.randn(2, 1, PATCH_CH, 128)
    # n_fft=256 should be skipped (T=128 < 256); n_fft=64 should run
    loss = multi_scale_stft_loss(x, x.clone(), n_ffts=(64, 256))
    assert torch.isfinite(loss).item()


def test_vae_loss_components_decrease_with_better_recon():
    x = torch.randn(2, 1, PATCH_CH, PATCH_T)
    near = x + 0.01 * torch.randn_like(x)
    far = x + 1.0 * torch.randn_like(x)
    mu = torch.zeros(2, LATENT_C, PATCH_CH, LATENT_T)
    logvar = torch.zeros(2, LATENT_C, PATCH_CH, LATENT_T)
    loss_near, _ = vae_loss(x, near, mu, logvar)
    loss_far, _ = vae_loss(x, far, mu, logvar)
    assert loss_near < loss_far


def test_full_target_geometry_runs():
    """Smoke-test the full [1, 8, 16384] -> [4, 8, 256] geometry with a small width.

    Using small encoder channels to keep the test fast.
    """
    model = DASVAEv2(
        in_channels=1,
        encoder_channels=(8, 16, 16, 16),
        latent_channels=4,
        temporal_strides=(4, 4, 4),
        kernel_time=7,
        kernel_space=3,
    )
    x = torch.randn(1, 1, 8, 16384)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == (1, 1, 8, 16384)
    assert mu.shape == (1, 4, 8, 256)
