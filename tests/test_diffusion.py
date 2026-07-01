"""Tests for DASDiffusionUNet and DASDiffusionTrainer (class + alpha conditioning)."""

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.das_diffusion_unet import DASDiffusionUNet, sinusoidal_embedding  # noqa: E402
from src.training.diffusion_trainer import (  # noqa: E402
    DASDiffusionTrainer,
    EMA,
    flatten_latent,
    unflatten_latent,
)


LATENT_C = 4
SPATIAL_H = 8
LATENT_W = 64
NUM_CLASSES = 9
COND_DIM = 64


@pytest.fixture
def small_model():
    return DASDiffusionUNet(
        latent_channels=LATENT_C,
        spatial_h=SPATIAL_H,
        base_channels=16,
        channel_mults=(1, 2, 2, 4),
        num_res_blocks=1,
        num_classes=NUM_CLASSES,
        cond_dim=COND_DIM,
        time_dim=32,
    )


def test_dilated_resblocks_preserve_length():
    """ResBlocks with dilation > 1 must still output the same temporal length as input."""
    model = DASDiffusionUNet(
        latent_channels=1,
        spatial_h=SPATIAL_H,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=2,
        num_classes=NUM_CLASSES,
        cond_dim=32,
        time_dim=16,
        dilations_per_level=[[1, 2], [1, 4]],
    )
    B = 2
    W = 256
    x = torch.randn(B, SPATIAL_H, W)
    t = torch.zeros(B, dtype=torch.long)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([0.0, 0.5])
    out = model(x, t, cls, alpha)
    assert out.shape == (B, SPATIAL_H, W)


def test_attention_layers_active_when_enabled():
    """With use_attention_per_level[-1]=True, the deepest level must have a SelfAttention1D."""
    from src.models.das_diffusion_unet import SelfAttention1D
    model = DASDiffusionUNet(
        latent_channels=1,
        spatial_h=SPATIAL_H,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        num_classes=NUM_CLASSES,
        cond_dim=32,
        time_dim=16,
        use_attention_per_level=[False, True],
        attention_in_mid=True,
        attention_heads=2,
    )
    # Level 0 has no attention (Identity); level 1 has SelfAttention1D.
    assert not isinstance(model.down_attns[0], SelfAttention1D)
    assert isinstance(model.down_attns[1], SelfAttention1D)
    assert isinstance(model.mid_attn, SelfAttention1D)
    # Up path: stages iterate in reverse — first up stage = deepest level.
    assert isinstance(model.up_attns[0], SelfAttention1D)
    assert not isinstance(model.up_attns[1], SelfAttention1D)


def test_attention_changes_output():
    """Attention vs no-attention at the same shape must produce different outputs."""
    torch.manual_seed(0)
    model_no = DASDiffusionUNet(
        latent_channels=1, spatial_h=SPATIAL_H, base_channels=16,
        channel_mults=(1, 2), num_res_blocks=1, num_classes=NUM_CLASSES,
        cond_dim=32, time_dim=16,
        use_attention_per_level=[False, False],
    )
    torch.manual_seed(0)
    model_with = DASDiffusionUNet(
        latent_channels=1, spatial_h=SPATIAL_H, base_channels=16,
        channel_mults=(1, 2), num_res_blocks=1, num_classes=NUM_CLASSES,
        cond_dim=32, time_dim=16,
        use_attention_per_level=[False, True],
        attention_heads=2,
    )
    x = torch.randn(1, SPATIAL_H, 256)
    t = torch.zeros(1, dtype=torch.long)
    cls = torch.tensor([0])
    alpha = torch.tensor([0.0])
    out_no = model_no(x, t, cls, alpha)
    out_with = model_with(x, t, cls, alpha)
    # Different architectures must produce different outputs.
    assert not torch.allclose(out_no, out_with)


def test_attention_heads_must_divide_channels():
    from src.models.das_diffusion_unet import SelfAttention1D
    with pytest.raises(ValueError, match="divisible"):
        SelfAttention1D(channels=10, num_heads=4)


def test_sinusoidal_embedding_shape():
    t = torch.tensor([0, 100, 999, 500])
    emb = sinusoidal_embedding(t, dim=64)
    assert emb.shape == (4, 64)


def test_forward_shape_preserved(small_model):
    B = 2
    x = torch.randn(B, LATENT_C * SPATIAL_H, LATENT_W)
    t = torch.randint(0, 1000, (B,))
    cls = torch.tensor([0, 7])
    alpha = torch.tensor([0.3, 0.8])
    out = small_model(x, t, cls, alpha)
    assert out.shape == x.shape


def test_null_class_index_accepted(small_model):
    """class_idx == num_classes routes through the learned NULL class embedding."""
    B = 2
    x = torch.randn(B, LATENT_C * SPATIAL_H, LATENT_W)
    t = torch.zeros(B, dtype=torch.long)
    cls = torch.tensor([NUM_CLASSES, NUM_CLASSES])
    alpha = torch.tensor([0.5, 0.5])
    out = small_model(x, t, cls, alpha)
    assert out.shape == x.shape


def test_null_alpha_path(small_model):
    """alpha == -1 sentinel routes through the learned null_alpha parameter."""
    B = 2
    x = torch.randn(B, LATENT_C * SPATIAL_H, LATENT_W)
    t = torch.zeros(B, dtype=torch.long)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([-1.0, -1.0])
    out = small_model(x, t, cls, alpha)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_alpha_zero_vs_one_produces_different_pred(small_model):
    """Same x, t, class but different alpha must produce different outputs."""
    small_model.eval()
    B = 2
    x = torch.randn(B, LATENT_C * SPATIAL_H, LATENT_W)
    t = torch.zeros(B, dtype=torch.long)
    cls = torch.tensor([3, 3])
    out_a = small_model(x, t, cls, torch.tensor([0.0, 0.0]))
    out_b = small_model(x, t, cls, torch.tensor([1.0, 1.0]))
    assert not torch.allclose(out_a, out_b)


def test_alpha_null_vs_zero_differs(small_model):
    """null_alpha (alpha=-1) must produce a different cond than alpha=0."""
    small_model.eval()
    B = 1
    x = torch.randn(B, LATENT_C * SPATIAL_H, LATENT_W)
    t = torch.zeros(B, dtype=torch.long)
    cls = torch.tensor([2])
    out_zero = small_model(x, t, cls, torch.tensor([0.0]))
    out_null = small_model(x, t, cls, torch.tensor([-1.0]))
    assert not torch.allclose(out_zero, out_null)


def test_gradient_flow(small_model):
    B = 2
    x = torch.randn(B, LATENT_C * SPATIAL_H, LATENT_W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    cls = torch.tensor([1, 4])
    alpha = torch.tensor([0.2, 0.9])
    out = small_model(x, t, cls, alpha)
    out.sum().backward()
    for name, p in small_model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_flatten_unflatten_roundtrip():
    latent = torch.randn(3, LATENT_C, SPATIAL_H, LATENT_W)
    flat = flatten_latent(latent)
    assert flat.shape == (3, LATENT_C * SPATIAL_H, LATENT_W)
    back = unflatten_latent(flat, LATENT_C, SPATIAL_H)
    assert torch.equal(latent, back)


def test_ema_update():
    m = torch.nn.Linear(4, 4)
    ema = EMA(m, decay=0.5)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(10.0)
    ema.update(m)
    target = 0.5 * w0 + 0.5 * (w0 + 10.0)
    assert torch.allclose(ema.shadow["weight"], target, atol=1e-6)


def _toy_loader():
    class _Toy:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0
    return _Toy()


def test_cfg_dropout_fires(small_model):
    """With cfg_dropout=1.0 every sample's class is replaced with NULL — loss must remain finite."""
    trainer = DASDiffusionTrainer(
        model=small_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        cfg_dropout=1.0,
        alpha_cfg_dropout=1.0,
        num_train_timesteps=10,
    )
    latents = torch.randn(2, LATENT_C, SPATIAL_H, LATENT_W)
    cls = torch.tensor([0, 7])
    alpha = torch.tensor([0.5, 0.5])
    loss = trainer._diffusion_loss(
        latents.to(trainer.device),
        cls.to(trainer.device).long(),
        alpha.to(trainer.device).float(),
    )
    assert torch.isfinite(loss)


def test_diffusion_loss_finite(small_model):
    trainer = DASDiffusionTrainer(
        model=small_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        num_train_timesteps=10,
    )
    latents = torch.randn(2, LATENT_C, SPATIAL_H, LATENT_W)
    cls = torch.tensor([2, 5])
    alpha = torch.tensor([0.1, 0.9])
    loss = trainer._diffusion_loss(
        latents.to(trainer.device),
        cls.to(trainer.device).long(),
        alpha.to(trainer.device).float(),
    )
    assert torch.isfinite(loss) and loss.item() > 0


def test_trainer_encode_requires_vae(small_model):
    """Calling _encode_to_latent without a VAE must raise a clear error."""
    trainer = DASDiffusionTrainer(
        model=small_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
    )
    with pytest.raises(RuntimeError, match="VAE"):
        trainer._encode_to_latent(torch.randn(1, 1, 8, 1024))


def test_train_script_initializes_wandb_when_enabled(monkeypatch):
    """train_das_diffusion.init_wandb must call wandb.init with the configured project."""
    import sys as _sys
    import types as _types

    calls = {}

    fake_wandb = _types.ModuleType("wandb")

    def fake_init(project, name, config, resume):
        calls["project"] = project
        calls["name"] = name
        calls["config"] = config
        calls["resume"] = resume
        return None

    fake_wandb.init = fake_init
    monkeypatch.setitem(_sys.modules, "wandb", fake_wandb)

    from scripts.train_das_diffusion import init_wandb  # imported lazily so the patch applies

    cfg = {
        "training": {
            "diffusion": {
                "wandb_project": "das-stable-diffusion",
                "wandb_run_name": "test-run-xyz",
            }
        }
    }
    result = init_wandb(cfg, "diffusion")
    assert result is fake_wandb
    assert calls["project"] == "das-stable-diffusion"
    assert calls["name"] == "test-run-xyz"
    assert calls["resume"] == "allow"


def test_train_script_skips_wandb_when_project_missing(monkeypatch):
    """init_wandb returns None when no wandb_project is configured."""
    from scripts.train_das_diffusion import init_wandb
    cfg = {"training": {"diffusion": {}}}
    assert init_wandb(cfg, "diffusion") is None


def test_trainer_pixel_mode_uses_patch_as_latent(small_model):
    """With vae=None the trainer must use the input patch directly as the diffusion target."""
    trainer = DASDiffusionTrainer(
        model=small_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        num_train_timesteps=10,
    )
    # In pixel mode, _to_diffusion_input must be the identity (no VAE encode).
    patch = torch.randn(2, 1, SPATIAL_H, LATENT_W)
    out = trainer._to_diffusion_input(patch)
    assert torch.equal(out, patch)


def test_trainer_latent_mode_routes_through_vae():
    """With vae set, _to_diffusion_input must call _encode_to_latent (scaled mu)."""
    from src.models.das_vae_v2 import DASVAEv2

    vae = DASVAEv2(
        in_channels=1,
        encoder_channels=(4, 8, 8, 8),
        latent_channels=2,
        temporal_strides=(4, 4, 4),
    )
    diffusion = DASDiffusionUNet(
        latent_channels=2,
        spatial_h=SPATIAL_H,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        num_classes=NUM_CLASSES,
        cond_dim=16,
        time_dim=16,
    )
    trainer = DASDiffusionTrainer(
        model=diffusion,
        vae=vae,
        scaling_factor=3.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
    )
    patch = torch.randn(1, 1, SPATIAL_H, 1024)
    out = trainer._to_diffusion_input(patch)
    # Should be VAE mu * scaling_factor, with the latent geometry [B, 2, 8, 1024//64=16]
    assert out.shape == (1, 2, SPATIAL_H, 16)
    # Verify scaling: should equal vae.encode(patch)[0] * 3.0
    with torch.no_grad():
        mu, _ = vae.encode(patch)
        expected = mu * 3.0
    assert torch.allclose(out, expected, atol=1e-5)


def test_log_real_vs_generated_in_pixel_mode():
    """_log_real_vs_generated should generate one image per cached class via wandb.Image.

    Pixel-mode usage requires latent_channels=1 so the diffusion output IS the patch.
    """
    pixel_model = DASDiffusionUNet(
        latent_channels=1,
        spatial_h=SPATIAL_H,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        num_classes=NUM_CLASSES,
        cond_dim=16,
        time_dim=16,
    )
    # Use enough time samples for the STFT plotter (nperseg=1024) to not warn.
    PATCH_T = 2048

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
    trainer = DASDiffusionTrainer(
        model=pixel_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        num_train_timesteps=10,
        wandb_logger=wandb,
        sample_log_steps=2,
        patch_time=PATCH_T,
    )
    trainer._real_patches_by_class = {
        0: (torch.randn(1, 1, SPATIAL_H, PATCH_T), 0.2),
        7: (torch.randn(1, 1, SPATIAL_H, PATCH_T), 0.8),
    }
    trainer._log_real_vs_generated(epoch=3)
    payloads_with_images = [c for c in wandb.calls
                            if any(k.startswith("real_vs_gen/") for k in c)]
    assert payloads_with_images, "expected wandb.log call with real_vs_gen images"
    payload = payloads_with_images[0]
    # New panel keys: waterfalls + per-channel spec grid, per cached class.
    assert "real_vs_gen/cls0_waterfalls" in payload
    assert "real_vs_gen/cls0_spec_grid" in payload
    assert "real_vs_gen/cls7_waterfalls" in payload
    assert "real_vs_gen/cls7_spec_grid" in payload
    assert isinstance(payload["real_vs_gen/cls0_waterfalls"], FakeImage)
    assert isinstance(payload["real_vs_gen/cls0_spec_grid"], FakeImage)


def test_stft_aux_loss_active_in_pixel_mode():
    """With lambda_stft>0 and vae=None, the auxiliary STFT term must contribute (>0 mse part stash)."""
    pixel_model = DASDiffusionUNet(
        latent_channels=1,
        spatial_h=SPATIAL_H,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        num_classes=NUM_CLASSES,
        cond_dim=16,
        time_dim=16,
    )
    PATCH_T = 4096  # long enough for n_fft=1024
    trainer = DASDiffusionTrainer(
        model=pixel_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        num_train_timesteps=10,
        lambda_stft=0.5,
        stft_n_ffts=(1024,),
        patch_time=PATCH_T,
    )
    # Latent shape in pixel mode: [B, latent_channels=1, spatial_h=8, W=patch_time]
    fake_patch = torch.randn(2, 1, SPATIAL_H, PATCH_T)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([0.2, 0.7])
    loss = trainer._diffusion_loss(fake_patch, cls.long(), alpha.float())
    assert torch.isfinite(loss)
    # Components stashed on the trainer:
    assert trainer._last_loss_parts["mse"] > 0
    assert trainer._last_loss_parts["stft"] > 0


def test_min_snr_weighting_changes_mse():
    """Min-SNR weighting must produce a different MSE than uniform weighting."""
    pixel_model = DASDiffusionUNet(
        latent_channels=1, spatial_h=SPATIAL_H, base_channels=8,
        channel_mults=(1, 2), num_res_blocks=1, num_classes=NUM_CLASSES,
        cond_dim=16, time_dim=16,
    )
    PATCH_T = 2048
    trainer_uniform = DASDiffusionTrainer(
        model=pixel_model, vae=None, scaling_factor=1.0,
        train_loader=_toy_loader(), val_loader=None, device="cpu",
        epochs=1, amp=False, num_train_timesteps=10,
        min_snr_gamma=None, patch_time=PATCH_T,
    )
    trainer_snr = DASDiffusionTrainer(
        model=pixel_model, vae=None, scaling_factor=1.0,
        train_loader=_toy_loader(), val_loader=None, device="cpu",
        epochs=1, amp=False, num_train_timesteps=10,
        min_snr_gamma=5.0, patch_time=PATCH_T,
    )
    fake_patch = torch.randn(4, 1, SPATIAL_H, PATCH_T)
    cls = torch.tensor([0, 1, 2, 3])
    alpha = torch.tensor([0.0, 0.3, 0.6, 1.0])
    torch.manual_seed(0)
    loss_u = trainer_uniform._diffusion_loss(fake_patch, cls.long(), alpha.float())
    torch.manual_seed(0)
    loss_s = trainer_snr._diffusion_loss(fake_patch, cls.long(), alpha.float())
    assert torch.isfinite(loss_u) and torch.isfinite(loss_s)
    # The weighted version must produce a different scalar.
    assert not torch.isclose(loss_u, loss_s, atol=1e-6)


def test_band_stft_loss_active_in_pixel_mode():
    """lambda_band_stft > 0 must produce a positive 'band_stft' part stash in pixel mode."""
    pixel_model = DASDiffusionUNet(
        latent_channels=1, spatial_h=SPATIAL_H, base_channels=8,
        channel_mults=(1, 2), num_res_blocks=1, num_classes=NUM_CLASSES,
        cond_dim=16, time_dim=16,
    )
    PATCH_T = 2048
    trainer = DASDiffusionTrainer(
        model=pixel_model, vae=None, scaling_factor=1.0,
        train_loader=_toy_loader(), val_loader=None, device="cpu",
        epochs=1, amp=False, num_train_timesteps=10,
        lambda_band_stft=1.0, band_stft_freq_max=50.0,
        band_stft_n_ffts=(1024,),
        lambda_stft=0.0, lambda_deriv=0.0,
        patch_time=PATCH_T, sample_rate=500,
    )
    fake_patch = torch.randn(2, 1, SPATIAL_H, PATCH_T)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([0.3, 0.7])
    loss = trainer._diffusion_loss(fake_patch, cls.long(), alpha.float())
    assert torch.isfinite(loss)
    assert trainer._last_loss_parts["band_stft"] > 0
    assert trainer._last_loss_parts["stft"] == 0
    assert trainer._last_loss_parts["deriv"] == 0


def test_band_stft_keeps_only_low_freq_bins():
    """band_limited_stft_loss should produce different values for different freq cutoffs."""
    from src.models.das_vae_v2 import band_limited_stft_loss
    torch.manual_seed(0)
    x = torch.randn(2, 1, 8, 2048)
    x_hat = x + 0.1 * torch.randn_like(x)
    loss_low = band_limited_stft_loss(x, x_hat, fs=500, freq_max_hz=30.0, n_ffts=(1024,))
    loss_wide = band_limited_stft_loss(x, x_hat, fs=500, freq_max_hz=200.0, n_ffts=(1024,))
    # With noise added uniformly across frequencies, restricting to <30 Hz vs <200 Hz must
    # yield different mean magnitudes.
    assert not torch.isclose(loss_low, loss_wide, atol=1e-6)


def test_deriv_loss_active_in_pixel_mode():
    """lambda_deriv > 0 must contribute a positive 'deriv' part stash."""
    pixel_model = DASDiffusionUNet(
        latent_channels=1, spatial_h=SPATIAL_H, base_channels=8,
        channel_mults=(1, 2), num_res_blocks=1, num_classes=NUM_CLASSES,
        cond_dim=16, time_dim=16,
    )
    PATCH_T = 2048
    trainer = DASDiffusionTrainer(
        model=pixel_model, vae=None, scaling_factor=1.0,
        train_loader=_toy_loader(), val_loader=None, device="cpu",
        epochs=1, amp=False, num_train_timesteps=10,
        lambda_deriv=1.0, lambda_stft=0.0,
        patch_time=PATCH_T,
    )
    fake_patch = torch.randn(2, 1, SPATIAL_H, PATCH_T)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([0.2, 0.7])
    loss = trainer._diffusion_loss(fake_patch, cls.long(), alpha.float())
    assert torch.isfinite(loss)
    assert trainer._last_loss_parts["deriv"] > 0
    # And STFT term stays zero
    assert trainer._last_loss_parts["stft"] == 0


def test_stft_aux_loss_skipped_in_latent_mode():
    """In latent mode (vae set) the STFT term must be 0 — STFT on latent codes is meaningless."""
    from src.models.das_vae_v2 import DASVAEv2

    vae = DASVAEv2(
        in_channels=1,
        encoder_channels=(4, 8, 8, 8),
        latent_channels=2,
        temporal_strides=(4, 4, 4),
    )
    diffusion = DASDiffusionUNet(
        latent_channels=2,
        spatial_h=SPATIAL_H,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        num_classes=NUM_CLASSES,
        cond_dim=16,
        time_dim=16,
    )
    trainer = DASDiffusionTrainer(
        model=diffusion,
        vae=vae,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        num_train_timesteps=10,
        lambda_stft=0.5,
        stft_n_ffts=(1024,),
    )
    fake_latent = torch.randn(2, 2, SPATIAL_H, 64)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([0.2, 0.7])
    loss = trainer._diffusion_loss(fake_latent, cls.long(), alpha.float())
    assert torch.isfinite(loss)
    # STFT must be skipped in latent mode.
    assert trainer._last_loss_parts["stft"] == 0


def test_pixel_diffusion_loss_finite_no_vae(small_model):
    """End-to-end: pixel-space diffusion loss is finite with vae=None."""
    trainer = DASDiffusionTrainer(
        model=small_model,
        vae=None,
        scaling_factor=1.0,
        train_loader=_toy_loader(),
        val_loader=None,
        device="cpu",
        epochs=1,
        amp=False,
        num_train_timesteps=10,
    )
    # Build a "latent" that is actually the raw patch in pixel mode:
    # shape [B, latent_channels=4, spatial_h=8, W=64] — model dims are reused.
    fake_patch_as_latent = torch.randn(2, LATENT_C, SPATIAL_H, LATENT_W)
    cls = torch.tensor([0, 1])
    alpha = torch.tensor([0.0, 0.5])
    loss = trainer._diffusion_loss(
        fake_patch_as_latent.to(trainer.device),
        cls.to(trainer.device).long(),
        alpha.to(trainer.device).float(),
    )
    assert torch.isfinite(loss) and loss.item() > 0
