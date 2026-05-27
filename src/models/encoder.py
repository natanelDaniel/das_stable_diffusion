"""
CVAE Encoder.

Input:  x [B, 1, H=32, W=256], c [B, n_classes]
Output: mu [B, latent_dim], logvar [B, latent_dim]

Encoding path: Conv2D × 3 (stride=2) → Flatten → FC(+ condition) → (mu, logvar)
"""

import torch
import torch.nn as nn
from typing import Tuple


class CVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_channels: int = 32,
        patch_time: int = 256,
        enc_channels: Tuple[int, ...] = (32, 64, 128),
        latent_dim: int = 64,
        n_classes: int = 9,
    ):
        super().__init__()
        self.patch_channels = patch_channels
        self.patch_time = patch_time

        # ---- Convolutional backbone ----
        layers = []
        ch_in = in_channels
        for ch_out in enc_channels:
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch_in = ch_out
        self.conv_backbone = nn.Sequential(*layers)

        # Compute flattened size after all strided convolutions
        n_strides = len(enc_channels)
        H_out = patch_channels // (2 ** n_strides)
        W_out = patch_time // (2 ** n_strides)
        self.flat_dim = enc_channels[-1] * H_out * W_out

        # ---- Fully-connected projection ----
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim + n_classes, 512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, H, W]  — input patch
            c: [B, n_classes] — one-hot condition
        Returns:
            mu, logvar: each [B, latent_dim]
        """
        h = self.conv_backbone(x)          # [B, 128, H//8, W//8]
        h = h.flatten(start_dim=1)         # [B, flat_dim]
        h = torch.cat([h, c], dim=1)       # [B, flat_dim + n_classes]
        h = self.fc(h)                     # [B, 512]
        return self.fc_mu(h), self.fc_logvar(h)
