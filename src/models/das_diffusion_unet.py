"""
DASDiffusionUNet — 1D U-Net for class + alpha-conditional latent diffusion.

Input/output: [B, latent_channels * spatial_h, W]
The VAE latent [B, 4, 8, 256] is flattened along H -> [B, 32, 256] before this model.
Conditioning is via FiLM (scale+shift from time + class + alpha embedding) at every ResBlock.

Inputs to forward(x, t, class_idx, alpha):
  * class_idx in [0, num_classes-1] is real; index == num_classes is the learned NULL
    class embedding used for classifier-free guidance.
  * alpha is a [B] float tensor. Values in [0, 1] are real noise levels; the sentinel
    value alpha < 0 (e.g. -1) routes through the learned null_alpha embedding (CFG drop).
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding for diffusion timesteps. t: [B]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = num_groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class ResBlock1D(nn.Module):
    """Two-conv 1D ResBlock with FiLM scale+shift conditioning.

    Optional dilated convolutions: with dilation>1, effective receptive field grows
    without extra parameters or downsampling — WaveNet-style.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        kernel: int = 5,
        dilation: int = 1,
    ):
        super().__init__()
        # Padding preserves length under dilation: pad = (k-1)*d / 2 for odd k
        pad = (kernel - 1) * dilation // 2
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the temporal dimension.

    Used at low-resolution levels of the UNet to give the model GLOBAL receptive
    field on the temporal axis (any time bin can attend to any other). This is
    what stable-diffusion-style models need to capture events that span many
    seconds of audio/signal.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_ch = channels // num_heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, W]
        B, C, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        # Reshape for multi-head
        q = q.view(B, self.num_heads, self.head_ch, W)
        k = k.view(B, self.num_heads, self.head_ch, W)
        v = v.view(B, self.num_heads, self.head_ch, W)
        scale = self.head_ch ** -0.5
        # Attention scores: [B, heads, W_q, W_k]
        attn = torch.einsum("bhcq,bhck->bhqk", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bhck->bhcq", attn, v)
        out = out.reshape(B, C, W)
        return x + self.proj(out)


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DASDiffusionUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        spatial_h: int = 8,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        num_classes: int = 9,
        cond_dim: int = 256,
        time_dim: int = 128,
        kernel: int = 5,
        dilations_per_level: Optional[Sequence[Sequence[int]]] = None,
        use_attention_per_level: Optional[Sequence[bool]] = None,
        attention_in_mid: bool = False,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_h = spatial_h
        self.num_classes = num_classes
        self.null_idx = num_classes
        self.cond_dim = cond_dim
        self.time_dim = time_dim

        in_ch = latent_channels * spatial_h
        chs = [base_channels * m for m in channel_mults]
        N = len(chs)
        self._chs = chs

        # Time + class + alpha -> cond
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)
        nn.init.normal_(self.class_emb.weight, std=0.02)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Learned null token for CFG dropout of the alpha conditioning.
        self.null_alpha = nn.Parameter(torch.zeros(cond_dim))
        nn.init.normal_(self.null_alpha, std=0.02)

        # Default dilations: all 1 (backward compatible)
        if dilations_per_level is None:
            dilations_per_level = [[1] * num_res_blocks for _ in range(N)]
        if len(dilations_per_level) != N:
            raise ValueError(
                f"dilations_per_level has {len(dilations_per_level)} entries, expected {N}"
            )
        for lvl in dilations_per_level:
            if len(lvl) != num_res_blocks:
                raise ValueError(
                    f"each dilations_per_level entry must have {num_res_blocks} dilations"
                )
        self.dilations_per_level = [list(d) for d in dilations_per_level]

        # Default attention: off at every level (backward compatible)
        if use_attention_per_level is None:
            use_attention_per_level = [False] * N
        if len(use_attention_per_level) != N:
            raise ValueError(
                f"use_attention_per_level has {len(use_attention_per_level)} entries, expected {N}"
            )
        self.use_attention_per_level = list(use_attention_per_level)
        self.attention_heads = int(attention_heads)
        self.attention_in_mid = bool(attention_in_mid)

        # Input projection
        self.in_conv = nn.Conv1d(in_ch, chs[0], 3, padding=1)

        # Down stages
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = chs[0]
        for i in range(N):
            stage = nn.ModuleList()
            for k in range(num_res_blocks):
                in_block = prev if k == 0 else chs[i]
                stage.append(ResBlock1D(
                    in_block, chs[i], cond_dim,
                    kernel=kernel,
                    dilation=self.dilations_per_level[i][k],
                ))
            self.down_blocks.append(stage)
            if self.use_attention_per_level[i]:
                self.down_attns.append(SelfAttention1D(chs[i], num_heads=self.attention_heads))
            else:
                self.down_attns.append(nn.Identity())
            self.downsamples.append(Downsample1D(chs[i]) if i < N - 1 else nn.Identity())
            prev = chs[i]

        # Bottleneck (deepest level; receives the most compressed signal).
        self.mid1 = ResBlock1D(chs[-1], chs[-1], cond_dim, kernel=kernel,
                               dilation=self.dilations_per_level[-1][-1])
        if self.attention_in_mid:
            self.mid_attn = SelfAttention1D(chs[-1], num_heads=self.attention_heads)
        else:
            self.mid_attn = nn.Identity()
        self.mid2 = ResBlock1D(chs[-1], chs[-1], cond_dim, kernel=kernel,
                               dilation=self.dilations_per_level[-1][-1])

        # Up stages. For up level i_up=0..N-1, i_down = N-1-i_up.
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev = chs[-1]
        for i_up in range(N):
            i_down = N - 1 - i_up
            stage = nn.ModuleList()
            for k in range(num_res_blocks):
                in_block = (prev + chs[i_down]) if k == 0 else chs[i_down]
                stage.append(ResBlock1D(
                    in_block, chs[i_down], cond_dim,
                    kernel=kernel,
                    dilation=self.dilations_per_level[i_down][k],
                ))
            self.up_blocks.append(stage)
            if self.use_attention_per_level[i_down]:
                self.up_attns.append(SelfAttention1D(chs[i_down], num_heads=self.attention_heads))
            else:
                self.up_attns.append(nn.Identity())
            self.upsamples.append(Upsample1D(chs[i_down]) if i_down > 0 else nn.Identity())
            prev = chs[i_down]

        # Output
        self.norm_out = _gn(chs[0])
        self.out_conv = nn.Conv1d(chs[0], in_ch, 3, padding=1)

    def _build_alpha_embedding(self, alpha: torch.Tensor) -> torch.Tensor:
        """alpha: [B] float. alpha < 0 sentinel -> null_alpha (CFG drop).

        Real alphas in [0, 1] are scaled to [0, 1000] so the sinusoidal embedding
        sees a numeric range comparable to diffusion timesteps and the frequencies
        actually resolve the input.
        """
        # Sentinel mask
        null_mask = (alpha < 0).unsqueeze(-1)  # [B, 1]
        alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
        a_emb = sinusoidal_embedding(alpha_clamped * 1000.0, self.time_dim)
        a_emb = self.alpha_mlp(a_emb)
        null_batch = self.null_alpha.unsqueeze(0).expand_as(a_emb)
        return torch.where(null_mask, null_batch, a_emb)

    def _build_cond(
        self,
        t: torch.Tensor,
        class_idx: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(class_idx)
        a_emb = self._build_alpha_embedding(alpha)
        return t_emb + c_emb + a_emb

    def _forward_with_cond(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        skips = []
        for stage, attn, downsample in zip(self.down_blocks, self.down_attns, self.downsamples):
            for block in stage:
                h = block(h, cond)
            h = attn(h)
            skips.append(h)
            h = downsample(h)
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        for stage, attn, upsample in zip(self.up_blocks, self.up_attns, self.upsamples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h, cond)
            h = attn(h)
            h = upsample(h)
        h = F.silu(self.norm_out(h))
        return self.out_conv(h)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_idx: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._build_cond(t, class_idx, alpha)
        return self._forward_with_cond(x, cond)
