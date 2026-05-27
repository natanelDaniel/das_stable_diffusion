"""
Conditional Variational Autoencoder (CVAE).
"""

import torch
import torch.nn as nn
from src.models.encoder import CVAEEncoder
from src.models.decoder import CVAEDecoder


class CVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        patch_channels: int = 32,
        patch_time: int = 256,
        enc_channels: tuple = (32, 64, 128),
        n_classes: int = 9,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.encoder = CVAEEncoder(
            in_channels=1,
            patch_channels=patch_channels,
            patch_time=patch_time,
            enc_channels=enc_channels,
            latent_dim=latent_dim,
            n_classes=n_classes,
        )
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim,
            patch_channels=patch_channels,
            patch_time=patch_time,
            enc_channels=enc_channels,
            n_classes=n_classes,
        )

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """z = mu + sigma * eps,  eps ~ N(0, I)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # deterministic at eval time

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ):
        """
        Args:
            x: [B, 1, H, W]  — input patch
            c: [B, n_classes] — one-hot class condition
        Returns:
            x_hat: [B, 1, H, W], mu: [B, latent_dim], logvar: [B, latent_dim]
        """
        mu, logvar = self.encoder(x, c)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    @torch.no_grad()
    def generate(
        self, c: torch.Tensor, n: int = 1, device: str = "cuda"
    ) -> torch.Tensor:
        """
        Sample n patches conditioned on class label c.

        Saves and restores training mode so this can be called mid-training
        (e.g. for logging sample images) without corrupting the training state.

        Args:
            c: [1, n_classes] or [n, n_classes] — condition(s)
            n: number of samples (ignored if c has batch dim > 1)
        Returns:
            samples: [n, 1, H, W]
        """
        was_training = self.training
        self.eval()
        try:
            if c.dim() == 1:
                c = c.unsqueeze(0).expand(n, -1)
            elif c.size(0) == 1 and n > 1:
                c = c.expand(n, -1)
            c = c.to(device)
            z = torch.randn(c.size(0), self.latent_dim, device=device)
            return self.decoder(z, c)
        finally:
            if was_training:
                self.train()
