"""
CVAE Decoder.

Input:  z [B, latent_dim], c [B, n_classes]
Output: x_hat [B, 1, H=32, W=256]

Decoding path: FC(z + c) → Reshape → ConvTranspose2D × 3 → x_hat
"""

import torch
import torch.nn as nn
from typing import Tuple


class CVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        patch_channels: int = 32,
        patch_time: int = 256,
        enc_channels: Tuple[int, ...] = (32, 64, 128),
        n_classes: int = 9,
    ):
        super().__init__()
        n_strides = len(enc_channels)
        H_base = patch_channels // (2 ** n_strides)  # e.g. 32//8 = 4
        W_base = patch_time // (2 ** n_strides)       # e.g. 256//8 = 32
        ch_base = enc_channels[-1]                     # e.g. 128
        self.flat_dim = ch_base * H_base * W_base      # e.g. 128*4*32 = 16384
        self.ch_base = ch_base
        self.H_base = H_base
        self.W_base = W_base

        # ---- Fully-connected projection ----
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.flat_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Transposed convolutional backbone (mirror of encoder) ----
        dec_channels = list(reversed(enc_channels))  # [128, 64, 32]
        layers = []
        ch_in = dec_channels[0]
        for ch_out in dec_channels[1:]:
            layers += [
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            ch_in = ch_out
        # Final layer to output 1 channel
        layers += [
            nn.ConvTranspose2d(ch_in, 1, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
        ]
        self.deconv_backbone = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
            c: [B, n_classes]
        Returns:
            x_hat: [B, 1, H, W]
        """
        h = torch.cat([z, c], dim=1)                       # [B, latent_dim + n_classes]
        h = self.fc(h)                                     # [B, flat_dim]
        h = h.view(-1, self.ch_base, self.H_base, self.W_base)  # [B, 128, 4, 32]
        return self.deconv_backbone(h)                     # [B, 1, 32, 256]
