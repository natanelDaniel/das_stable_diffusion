"""
β-VAE loss.

L = reconstruction_loss + beta * kl_loss
"""

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(N(mu, sigma^2) || N(0, I)).
    Returns mean KL over the batch.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE over all elements, averaged over batch."""
    return F.mse_loss(x_hat, x, reduction="mean")


def beta_vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> dict:
    """
    Compute β-VAE loss components.

    Returns:
        dict with keys: 'loss', 'recon', 'kl'
    """
    recon = reconstruction_loss(x, x_hat)
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return {"loss": total, "recon": recon, "kl": kl}


def linear_beta_schedule(
    step: int, warmup_steps: int, target_beta: float
) -> float:
    """Linearly anneal beta from 0 to target_beta over warmup_steps."""
    if warmup_steps == 0:
        return target_beta
    return min(target_beta, target_beta * step / warmup_steps)
