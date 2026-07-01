"""End-to-end sampling tests for class + alpha conditioning."""

import os
import sys

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.generate_diffusion_samples import (  # noqa: E402
    parse_class_alpha,
    sample,
    waterfall_png,
)
from src.data.das_latent_patch_dataset import CLASSES  # noqa: E402
from src.evaluation.diffusion_eval import (  # noqa: E402
    alpha_sweep_curve,
    is_monotonic_decreasing,
    recoverability,
)
from src.models.das_diffusion_unet import DASDiffusionUNet  # noqa: E402
from src.models.das_vae_v2 import DASVAEv2  # noqa: E402
from src.training.diffusion_trainer import unflatten_latent  # noqa: E402


@pytest.fixture
def tiny_diffusion():
    return DASDiffusionUNet(
        latent_channels=2,
        spatial_h=8,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        num_classes=len(CLASSES),
        cond_dim=32,
        time_dim=16,
    )


@pytest.fixture
def tiny_vae():
    return DASVAEv2(
        in_channels=1,
        encoder_channels=(4, 8, 8, 8),
        latent_channels=2,
        temporal_strides=(4, 4, 4),
        kernel_time=7,
        kernel_space=3,
    )


@pytest.fixture
def scheduler():
    return DDIMScheduler(
        num_train_timesteps=50,
        prediction_type="v_prediction",
        beta_schedule="squaredcos_cap_v2",
        rescale_betas_zero_snr=True,
    )


def test_parse_class_alpha_accepts_valid():
    idx, alpha = parse_class_alpha("running", 0.3)
    assert idx == CLASSES.index("running")
    assert alpha == 0.3


def test_parse_class_alpha_rejects_unknown_class():
    with pytest.raises(ValueError, match="Unknown class"):
        parse_class_alpha("biking", 0.5)


def test_parse_class_alpha_rejects_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        parse_class_alpha("running", -0.1)
    with pytest.raises(ValueError, match="alpha"):
        parse_class_alpha("running", 1.5)


def test_sample_shape_one_pass(tiny_diffusion, scheduler):
    """alpha_cfg_scale=1.0 -> 2 forward passes per step. Shape preserved."""
    x = sample(
        diffusion=tiny_diffusion,
        class_idx=3,
        alpha=0.3,
        cfg_scale=2.0,
        alpha_cfg_scale=1.0,
        n_samples=2,
        steps=4,
        latent_w=16,
        scheduler=scheduler,
        device="cpu",
    )
    assert x.shape == (2, 2 * 8, 16)


def test_sample_shape_two_axis(tiny_diffusion, scheduler):
    """alpha_cfg_scale != 1.0 -> 4 forward passes per step. Shape preserved."""
    x = sample(
        diffusion=tiny_diffusion,
        class_idx=3,
        alpha=0.3,
        cfg_scale=2.0,
        alpha_cfg_scale=2.0,
        n_samples=2,
        steps=4,
        latent_w=16,
        scheduler=scheduler,
        device="cpu",
    )
    assert x.shape == (2, 2 * 8, 16)


def test_one_pass_and_two_axis_differ_when_alpha_cfg_nonidentity(tiny_diffusion, scheduler):
    """alpha_cfg=1 (2 passes) and alpha_cfg=3 (4 passes) must produce different output."""
    torch.manual_seed(0)
    x1 = sample(
        diffusion=tiny_diffusion, class_idx=3, alpha=0.3,
        cfg_scale=2.0, alpha_cfg_scale=1.0,
        n_samples=1, steps=4, latent_w=16, scheduler=scheduler, device="cpu",
    )
    torch.manual_seed(0)
    x2 = sample(
        diffusion=tiny_diffusion, class_idx=3, alpha=0.3,
        cfg_scale=2.0, alpha_cfg_scale=3.0,
        n_samples=1, steps=4, latent_w=16, scheduler=scheduler, device="cpu",
    )
    assert not torch.allclose(x1, x2)


def test_end_to_end_decode(tiny_diffusion, tiny_vae, scheduler):
    """Latent -> VAE decode produces patches of the right geometry."""
    x_flat = sample(
        diffusion=tiny_diffusion,
        class_idx=CLASSES.index("running"),
        alpha=0.4,
        cfg_scale=2.0,
        alpha_cfg_scale=1.0,
        n_samples=1,
        steps=3,
        latent_w=16,  # patch_time = 16 * 64 = 1024
        scheduler=scheduler,
        device="cpu",
    )
    latent = unflatten_latent(x_flat, tiny_diffusion.latent_channels, tiny_diffusion.spatial_h)
    with torch.no_grad():
        patch = tiny_vae.decode(latent)
    assert patch.shape == (1, 1, 8, 1024)
    assert torch.isfinite(patch).all()


def test_waterfall_png_creates_file(tmp_path):
    # 4096 samples is enough for the STFT (nperseg=1024) to produce multiple frames.
    patch = np.random.randn(1, 8, 4096).astype(np.float32)
    out = tmp_path / "wf.png"
    waterfall_png(patch, str(out), title="test")
    assert out.exists()
    assert out.stat().st_size > 0


def test_is_monotonic_decreasing():
    assert is_monotonic_decreasing([0.9, 0.7, 0.5, 0.2])
    assert is_monotonic_decreasing([0.9, 0.85, 0.4])
    assert is_monotonic_decreasing([0.9, 0.92, 0.5], tol=0.05)
    assert not is_monotonic_decreasing([0.5, 0.9, 0.3])


def test_alpha_sweep_curve_calls_per_alpha():
    """alpha_sweep_curve must call the generator once per alpha with the right args."""
    calls = []

    def fake_generator(class_name, alpha, n):
        calls.append((class_name, alpha, n))
        return np.random.randn(n, 1, 8, 16384).astype(np.float32)

    def fake_classifier(patches):
        n = patches.shape[0]
        return np.ones((n, len(CLASSES))) / len(CLASSES)

    curve = alpha_sweep_curve(
        fake_generator,
        fake_classifier,
        classes=CLASSES,
        event_class="running",
        alphas=(0.0, 0.5, 1.0),
        n_per=2,
    )
    assert set(curve.keys()) == {0.0, 0.5, 1.0}
    assert len(calls) == 3
    classes_called = {c for c, _, _ in calls}
    alphas_called = sorted(a for _, a, _ in calls)
    assert classes_called == {"running"}
    assert alphas_called == [0.0, 0.5, 1.0]


def test_alpha_sweep_curve_rejects_unknown_class():
    def fake_generator(class_name, alpha, n):
        return np.random.randn(n, 1, 8, 16384).astype(np.float32)

    def fake_classifier(patches):
        n = patches.shape[0]
        return np.ones((n, len(CLASSES))) / len(CLASSES)

    with pytest.raises(ValueError, match="event_class"):
        alpha_sweep_curve(
            fake_generator, fake_classifier,
            classes=CLASSES, event_class="biking",
            alphas=(0.0, 1.0), n_per=2,
        )


def test_recoverability_calls_with_alpha_zero():
    """recoverability must request alpha=0 (clean events) for each class."""
    calls = []

    def fake_generator(class_name, alpha, n):
        calls.append((class_name, alpha, n))
        return np.random.randn(n, 1, 8, 16384).astype(np.float32)

    def fake_classifier(patches):
        n = patches.shape[0]
        out = np.zeros((n, len(CLASSES)))
        out[:, 0] = 1.0  # always predict class 0 (car)
        return out

    scores = recoverability(fake_generator, fake_classifier, CLASSES, n_per_class=2)
    assert set(scores.keys()) == set(CLASSES)
    assert scores["car"] == 1.0
    for c in CLASSES[1:]:
        assert scores[c] == 0.0
    for _, alpha, _ in calls:
        assert alpha == 0.0
