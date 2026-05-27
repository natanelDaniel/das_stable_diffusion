import torch
from src.training.losses import beta_vae_loss, kl_divergence, linear_beta_schedule


def test_kl_zero_for_standard_normal():
    """KL(N(0,I) || N(0,I)) should be 0."""
    mu = torch.zeros(8, 64)
    logvar = torch.zeros(8, 64)
    kl = kl_divergence(mu, logvar)
    assert abs(kl.item()) < 1e-5, f"Expected ~0, got {kl.item()}"


def test_kl_positive_for_non_standard():
    """KL should increase when mu or logvar deviate from 0."""
    mu = torch.ones(8, 64) * 2.0
    logvar = torch.ones(8, 64) * 1.0
    kl = kl_divergence(mu, logvar)
    assert kl.item() > 0


def test_loss_dict_keys():
    x = torch.randn(4, 1, 32, 256)
    x_hat = torch.randn(4, 1, 32, 256)
    mu = torch.zeros(4, 64)
    logvar = torch.zeros(4, 64)
    out = beta_vae_loss(x, x_hat, mu, logvar, beta=4.0)
    assert set(out.keys()) == {"loss", "recon", "kl"}


def test_loss_is_differentiable():
    x = torch.randn(4, 1, 32, 256)
    x_hat = torch.randn(4, 1, 32, 256, requires_grad=True)
    mu = torch.randn(4, 64, requires_grad=True)
    logvar = torch.randn(4, 64, requires_grad=True)
    out = beta_vae_loss(x, x_hat, mu, logvar, beta=4.0)
    out["loss"].backward()
    assert x_hat.grad is not None
    assert mu.grad is not None


def test_beta_schedule_warmup():
    assert linear_beta_schedule(0, 100, 4.0) == 0.0
    assert linear_beta_schedule(50, 100, 4.0) == 2.0
    assert linear_beta_schedule(100, 100, 4.0) == 4.0
    assert linear_beta_schedule(200, 100, 4.0) == 4.0  # clamps at target
