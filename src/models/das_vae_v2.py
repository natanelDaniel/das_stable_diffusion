"""
DASVAEv2 — unconditional VAE for [1, 8, 16384] DAS patches @ 1 kHz.

Stage-1 component of the latent-diffusion pipeline. Compresses input
[B, 1, 8, 16384] -> latent [B, 4, 8, 256] (64x temporal compression, H=8 preserved).

Design choices (see plan):
  * H=8 preserved throughout (only 8 input fibers; spatial down loses too much).
  * 3 down-blocks of temporal stride 4 -> 4**3 = 64x temporal compression.
  * Rectangular kernels (3 spatial x 7 temporal) tilted toward the time axis.
  * GroupNorm + SiLU to match diffusers UNet conventions.
  * Decoder uses nearest-upsample + conv (no ConvTranspose) to avoid checkerboard.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    # GroupNorm needs num_channels divisible by num_groups.
    g = num_groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class ResBlock(nn.Module):
    """Two-conv ResNet block, optionally with stride on the time axis."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_space: int = 3,
        kernel_time: int = 7,
        temporal_stride: int = 1,
    ):
        super().__init__()
        pad_s = kernel_space // 2
        pad_t = kernel_time // 2
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_space, kernel_time),
            stride=(1, temporal_stride),
            padding=(pad_s, pad_t),
        )
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=(kernel_space, kernel_time),
            stride=(1, 1),
            padding=(pad_s, pad_t),
        )
        # Skip connection: 1x1 conv if shape changes (channels or stride).
        if in_ch != out_ch or temporal_stride != 1:
            self.skip = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=1,
                stride=(1, temporal_stride),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DASVAEv2Encoder(nn.Module):
    """Encode [B, 1, 8, T] -> (mu, logvar) each [B, latent_channels, 8, T // prod(strides)]."""

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: Sequence[int] = (64, 128, 128, 128),
        latent_channels: int = 4,
        temporal_strides: Sequence[int] = (4, 4, 4),
        kernel_time: int = 7,
        kernel_space: int = 3,
    ):
        super().__init__()
        assert len(encoder_channels) == len(temporal_strides) + 1, (
            "encoder_channels must be one longer than temporal_strides"
        )
        self.in_conv = nn.Conv2d(
            in_channels,
            encoder_channels[0],
            kernel_size=(kernel_space, kernel_time),
            padding=(kernel_space // 2, kernel_time // 2),
        )
        blocks = []
        for i, stride in enumerate(temporal_strides):
            blocks.append(ResBlock(
                encoder_channels[i],
                encoder_channels[i + 1],
                kernel_space=kernel_space,
                kernel_time=kernel_time,
                temporal_stride=stride,
            ))
        self.down_blocks = nn.ModuleList(blocks)
        self.norm_out = _gn(encoder_channels[-1])
        # 1x1 projection to 2 * latent_channels (mu and logvar interleaved as halves).
        self.proj = nn.Conv2d(encoder_channels[-1], 2 * latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_conv(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.proj(F.silu(self.norm_out(h)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class DASVAEv2Decoder(nn.Module):
    """Decode latent [B, latent_channels, 8, T_lat] -> [B, 1, 8, T_lat * prod(strides)]."""

    def __init__(
        self,
        out_channels: int = 1,
        decoder_channels: Sequence[int] = (128, 128, 128, 64),
        latent_channels: int = 4,
        temporal_strides: Sequence[int] = (4, 4, 4),
        kernel_time: int = 7,
        kernel_space: int = 3,
    ):
        super().__init__()
        assert len(decoder_channels) == len(temporal_strides) + 1
        self.in_conv = nn.Conv2d(latent_channels, decoder_channels[0], kernel_size=1)
        self.up_strides = list(temporal_strides)

        # Up-blocks pair a nearest-upsample with a ResBlock at stride 1.
        blocks = []
        for i in range(len(temporal_strides)):
            blocks.append(ResBlock(
                decoder_channels[i],
                decoder_channels[i + 1],
                kernel_space=kernel_space,
                kernel_time=kernel_time,
                temporal_stride=1,
            ))
        self.up_blocks = nn.ModuleList(blocks)
        self.norm_out = _gn(decoder_channels[-1])
        self.out_conv = nn.Conv2d(
            decoder_channels[-1],
            out_channels,
            kernel_size=(kernel_space, kernel_time),
            padding=(kernel_space // 2, kernel_time // 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        for block, stride in zip(self.up_blocks, self.up_strides):
            if stride != 1:
                h = F.interpolate(h, scale_factor=(1, float(stride)), mode="nearest")
            h = block(h)
        h = self.out_conv(F.silu(self.norm_out(h)))
        return h


class DASVAEv2(nn.Module):
    """Wrapper exposing encode / reparameterize / decode / forward."""

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: Sequence[int] = (64, 128, 128, 128),
        latent_channels: int = 4,
        temporal_strides: Sequence[int] = (4, 4, 4),
        kernel_time: int = 7,
        kernel_space: int = 3,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_strides = tuple(temporal_strides)
        self.encoder = DASVAEv2Encoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
            temporal_strides=temporal_strides,
            kernel_time=kernel_time,
            kernel_space=kernel_space,
        )
        decoder_channels = tuple(reversed(encoder_channels))
        self.decoder = DASVAEv2Decoder(
            out_channels=in_channels,
            decoder_channels=decoder_channels,
            latent_channels=latent_channels,
            temporal_strides=tuple(reversed(temporal_strides)),
            kernel_time=kernel_time,
            kernel_space=kernel_space,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def multi_scale_stft_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    n_ffts: Sequence[int] = (256, 1024, 4096),
    hop_ratio: int = 4,
) -> torch.Tensor:
    """L1 loss on magnitude STFT, averaged over multiple window sizes.

    x, x_hat: [B, 1, C, T]. STFT runs per-channel; the C axis is folded into the batch.
    """
    if x.shape != x_hat.shape:
        raise ValueError(f"x and x_hat shape mismatch: {x.shape} vs {x_hat.shape}")
    B, _, C, T = x.shape
    x_flat = x.reshape(B * C, T)
    x_hat_flat = x_hat.reshape(B * C, T)
    loss = x.new_zeros(())
    n_used = 0
    for n_fft in n_ffts:
        if T < n_fft:
            continue
        hop = max(1, n_fft // hop_ratio)
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        spec_x = torch.stft(
            x_flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True,
        ).abs()
        spec_xh = torch.stft(
            x_hat_flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True,
        ).abs()
        loss = loss + (spec_x - spec_xh).abs().mean()
        n_used += 1
    return loss / max(1, n_used)


def band_limited_stft_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    fs: int,
    freq_max_hz: float,
    n_ffts: Sequence[int] = (1024, 2048),
    hop_ratio: int = 4,
) -> torch.Tensor:
    """L1 magnitude STFT restricted to freq bins below `freq_max_hz`.

    Most DAS event energy is concentrated below ~50 Hz; a full-band STFT loss spends
    a lot of effort matching empty high-frequency bins. This variant keeps only the
    bins below freq_max_hz so the model is rewarded for getting the signal band right.

    x, x_hat: [B, 1, C, T].
    """
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {x_hat.shape}")
    B, _, C, T = x.shape
    x_flat = x.reshape(B * C, T)
    x_hat_flat = x_hat.reshape(B * C, T)
    loss = x.new_zeros(())
    n_used = 0
    for n_fft in n_ffts:
        if T < n_fft:
            continue
        hop = max(1, n_fft // hop_ratio)
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        spec_x = torch.stft(
            x_flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True,
        ).abs()
        spec_xh = torch.stft(
            x_hat_flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True,
        ).abs()
        # Slice along freq axis to bins covering [0, freq_max_hz].
        freq_per_bin = fs / n_fft
        n_keep = max(1, min(spec_x.shape[1], int(round(freq_max_hz / freq_per_bin)) + 1))
        loss = loss + (spec_x[:, :n_keep] - spec_xh[:, :n_keep]).abs().mean()
        n_used += 1
    return loss / max(1, n_used)


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
    lambda_stft: float = 1.0,
    stft_n_ffts: Sequence[int] = (256, 1024, 4096),
) -> Tuple[torch.Tensor, dict]:
    mse = F.mse_loss(x_hat, x)
    # Skip STFT entirely when its weight is 0 — saves substantial FLOPs and memory.
    if lambda_stft > 0:
        stft = multi_scale_stft_loss(x, x_hat, n_ffts=stft_n_ffts)
    else:
        stft = mse.new_zeros(())
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = mse + lambda_stft * stft + beta * kl
    parts = {
        "mse": mse.detach(),
        "stft": stft.detach(),
        "kl": kl.detach(),
        "total": total.detach(),
    }
    return total, parts
