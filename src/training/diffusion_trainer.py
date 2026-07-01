"""
DASDiffusionTrainer — Stage-3 trainer for the class + alpha-conditional latent diffusion model.

Each batch yields (mixed_patch [B, 1, 8, 16384], class_idx [B], alpha [B]) from
DASLatentPatchDataset(return_mixed=True). The trainer encodes the mixed patch through
the frozen VAE encoder on the fly (gradients do NOT flow into the VAE), scales latents
by 1/sigma_latent, and runs the diffusion forward conditioned on (class_idx, alpha).

Schedule: cosine (squaredcos_cap_v2) with rescale_betas_zero_snr=True; prediction_type
defaults to v_prediction. CFG dropout is applied independently to class and alpha so
guidance can be applied to either dimension at sampling time.

EMA of weights is maintained alongside the trainable model. AMP optional.
"""

import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import DDIMScheduler

from src.evaluation.plotting import (
    plot_input_vs_recon,
    plot_multi_waterfall,
    plot_patch_panel,
    plot_per_channel_spectrograms,
)
from src.models.das_diffusion_unet import DASDiffusionUNet
from src.models.das_vae_v2 import DASVAEv2, band_limited_stft_loss, multi_scale_stft_loss


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


def flatten_latent(latent: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, C*H, W]"""
    B, C, H, W = latent.shape
    return latent.reshape(B, C * H, W)


def unflatten_latent(x: torch.Tensor, latent_channels: int, spatial_h: int) -> torch.Tensor:
    """[B, C*H, W] -> [B, C, H, W]"""
    B, CH, W = x.shape
    assert CH == latent_channels * spatial_h
    return x.reshape(B, latent_channels, spatial_h, W)


class DASDiffusionTrainer:
    def __init__(
        self,
        model: DASDiffusionUNet,
        vae: Optional[DASVAEv2],
        scaling_factor: float,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: str = "cuda",
        epochs: int = 200,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        amp: bool = True,
        cfg_dropout: float = 0.1,
        alpha_cfg_dropout: float = 0.1,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v_prediction",
        beta_schedule: str = "squaredcos_cap_v2",
        ema_decay: float = 0.9999,
        checkpoint_dir: str = "checkpoints/das_diffusion/",
        wandb_logger=None,
        log_freq: int = 100,
        sample_every_n_epochs: int = 5,
        sample_log_class_idx: int = 0,        # which class to render
        sample_log_alphas: tuple = (0.0, 0.5, 1.0),
        sample_log_cfg_scale: float = 5.0,
        sample_log_steps: int = 50,
        patch_time: int = 16_384,             # used only for pixel-mode sample logging
        lambda_stft: float = 0.0,             # weight of STFT-on-x0_pred auxiliary loss
        stft_n_ffts: tuple = (1024, 2048),    # window sizes for multi-scale STFT
        sample_log_n_gens: int = 3,           # generations per class in real_vs_gen panel
        lambda_deriv: float = 0.0,            # weight of |d(real)/dt - d(pred)/dt| transient loss
        min_snr_gamma: Optional[float] = None,  # min-SNR weighting clamp; None = off
        lambda_band_stft: float = 0.0,        # weight of band-limited STFT loss (low-freq focus)
        band_stft_freq_max: float = 50.0,     # upper cutoff for band-limited STFT (Hz)
        band_stft_n_ffts: tuple = (1024, 2048),
        sample_rate: int = 500,               # data sample rate (needed for freq->bin mapping)
    ):
        # Lets cuDNN pick the fastest conv kernel for each input shape on first iter.
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self.model = model.to(device)
        # VAE is frozen — eval mode, no grad. May be None for unit tests that
        # call _diffusion_loss directly with pre-encoded latents.
        self.vae = vae.to(device) if vae is not None else None
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)
        self.scaling_factor = float(scaling_factor)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.amp = amp and device.startswith("cuda")
        self.cfg_dropout = cfg_dropout
        self.alpha_cfg_dropout = alpha_cfg_dropout
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.wandb_logger = wandb_logger
        self.log_freq = log_freq
        self.null_idx = model.null_idx
        self.latent_channels = model.latent_channels
        self.spatial_h = model.spatial_h
        self.prediction_type = prediction_type

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler_lr = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.scaler = GradScaler(enabled=self.amp)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            beta_schedule=beta_schedule,
            rescale_betas_zero_snr=True,
        )

        self.ema = EMA(self.model, decay=ema_decay)
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Sample-logging configuration
        self.sample_every_n_epochs = int(sample_every_n_epochs)
        self.sample_log_class_idx = int(sample_log_class_idx)
        self.sample_log_alphas = tuple(sample_log_alphas)
        self.sample_log_cfg_scale = float(sample_log_cfg_scale)
        self.sample_log_steps = int(sample_log_steps)
        self.patch_time = int(patch_time)
        self.lambda_stft = float(lambda_stft)
        self.stft_n_ffts = tuple(stft_n_ffts)
        self.sample_log_n_gens = int(sample_log_n_gens)
        self.lambda_deriv = float(lambda_deriv)
        self.min_snr_gamma = None if min_snr_gamma in (None, 0) else float(min_snr_gamma)
        self.lambda_band_stft = float(lambda_band_stft)
        self.band_stft_freq_max = float(band_stft_freq_max)
        self.band_stft_n_ffts = tuple(band_stft_n_ffts)
        self.sample_rate = int(sample_rate)
        # Cache: class_idx -> (real_patch [1,1,H,W], alpha_used_for_real). Populated on
        # first call to validate(). Same patches used every epoch so the real-vs-generated
        # comparison is visually consistent across training.
        self._real_patches_by_class: Dict[int, Tuple[torch.Tensor, float]] = {}
        # DDIM scheduler mirrors the training schedule but with DDIM sampling logic.
        self._sampler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            beta_schedule=beta_schedule,
            rescale_betas_zero_snr=True,
        )

    @torch.no_grad()
    def _encode_to_latent(self, patches: torch.Tensor) -> torch.Tensor:
        """Pass mixed patches through the frozen VAE and scale by 1/sigma_latent."""
        if self.vae is None:
            raise RuntimeError(
                "Trainer has no VAE; either pass one or call _diffusion_loss with "
                "pre-encoded latents directly."
            )
        mu, _ = self.vae.encode(patches)
        return mu * self.scaling_factor

    def _diffusion_loss(
        self,
        latents: torch.Tensor,
        class_idx: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """latents:   [B, latent_ch, H, W] (already scaled by 1/sigma_latent)
        class_idx: [B] long in [0, num_classes-1]
        alpha:     [B] float in [0, 1]
        """
        B = latents.size(0)
        x0 = flatten_latent(latents)  # [B, C*H, W]

        # Independent CFG dropout on class and alpha. Different mask per dim so the
        # model learns the marginal scores for both axes separately.
        if self.cfg_dropout > 0:
            cls_mask = torch.rand(B, device=self.device) < self.cfg_dropout
            class_idx = class_idx.clone()
            class_idx[cls_mask] = self.null_idx
        if self.alpha_cfg_dropout > 0:
            alpha_mask = torch.rand(B, device=self.device) < self.alpha_cfg_dropout
            alpha = alpha.clone()
            alpha[alpha_mask] = -1.0  # sentinel -> null_alpha

        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()
        noise = torch.randn_like(x0)
        noisy = self.noise_scheduler.add_noise(x0, noise, t)

        if self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x0, noise, t)
        elif self.prediction_type == "epsilon":
            target = noise
        else:
            raise NotImplementedError(f"prediction_type={self.prediction_type}")

        pred = self.model(noisy, t, class_idx, alpha)

        # ----- 1. MSE on v (with optional min-SNR weighting) -----
        sq = (pred - target).pow(2)  # [B, C*H, W]
        if self.min_snr_gamma is not None:
            # SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            ab = alphas_cumprod[t]
            snr = ab / (1.0 - ab).clamp(min=1e-8)
            # v-prediction: w = min(snr, gamma) / (snr + 1)
            # epsilon:      w = min(snr, gamma) / snr
            if self.prediction_type == "v_prediction":
                w = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1.0)
            else:
                w = torch.clamp(snr, max=self.min_snr_gamma) / snr.clamp(min=1e-8)
            # Per-sample mean over channels/time, then weighted batch mean.
            loss_mse = (sq.mean(dim=[1, 2]) * w.to(sq.dtype)).mean()
        else:
            loss_mse = sq.mean()

        # ----- 2/3/4. STFT (full-band) + STFT (band-limited) + derivative on x0_pred -----
        # All physically meaningful only in pixel-space (vae=None) where x0 is the signal.
        stft_term = pred.new_zeros(())
        band_stft_term = pred.new_zeros(())
        deriv_term = pred.new_zeros(())
        need_x0 = (
            (self.lambda_stft > 0 or self.lambda_deriv > 0 or self.lambda_band_stft > 0)
            and self.vae is None
        )
        if need_x0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            ab = alphas_cumprod[t].view(-1, 1, 1)
            sqrt_ab = ab.sqrt()
            sqrt_one_minus = (1.0 - ab).sqrt()
            if self.prediction_type == "v_prediction":
                x0_pred = sqrt_ab * noisy - sqrt_one_minus * pred
            else:
                x0_pred = (noisy - sqrt_one_minus * pred) / sqrt_ab.clamp(min=1e-8)

            x0_4d = x0.view(B, 1, self.spatial_h, -1)
            x0_pred_4d = x0_pred.view(B, 1, self.spatial_h, -1)

            if self.lambda_stft > 0:
                stft_term = multi_scale_stft_loss(x0_4d, x0_pred_4d, n_ffts=self.stft_n_ffts)

            if self.lambda_band_stft > 0:
                band_stft_term = band_limited_stft_loss(
                    x0_4d, x0_pred_4d,
                    fs=self.sample_rate,
                    freq_max_hz=self.band_stft_freq_max,
                    n_ffts=self.band_stft_n_ffts,
                )

            if self.lambda_deriv > 0:
                d_real = x0[..., 1:] - x0[..., :-1]
                d_pred = x0_pred[..., 1:] - x0_pred[..., :-1]
                deriv_term = (d_real - d_pred).abs().mean()

        total = (
            loss_mse
            + self.lambda_stft * stft_term
            + self.lambda_band_stft * band_stft_term
            + self.lambda_deriv * deriv_term
        )
        self._last_loss_parts = {
            "mse": float(loss_mse.detach()),
            "stft": float(stft_term.detach()),
            "band_stft": float(band_stft_term.detach()),
            "deriv": float(deriv_term.detach()),
        }
        return total

    def _batch_to_device(self, batch):
        """Accept (mixed_patch, class_idx, alpha) and move to device with correct dtypes."""
        mixed_patch, class_idx, alpha = batch
        mixed_patch = mixed_patch.to(self.device, non_blocking=True)
        class_idx = class_idx.to(self.device, non_blocking=True).long()
        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha)
        alpha = alpha.to(self.device, non_blocking=True).float()
        return mixed_patch, class_idx, alpha

    def _to_diffusion_input(self, mixed_patch: torch.Tensor) -> torch.Tensor:
        """Latent-space (vae set): encode through frozen VAE and scale.
        Pixel-space (vae=None): use the patch itself as the diffusion target."""
        if self.vae is not None:
            return self._encode_to_latent(mixed_patch)
        return mixed_patch

    def train_epoch(self, epoch: int):
        self.model.train()
        for batch in tqdm(self.train_loader, desc=f"train epoch {epoch}"):
            mixed_patch, class_idx, alpha = self._batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                latents = self._to_diffusion_input(mixed_patch)
                loss = self._diffusion_loss(latents, class_idx, alpha)

            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)

            self.global_step += 1
            if self.wandb_logger and self.global_step % self.log_freq == 0:
                payload = {
                    "train/loss": float(loss),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/alpha_mean": float(alpha.mean()),
                    "train/alpha_std": float(alpha.std()),
                    "step": self.global_step,
                }
                parts = getattr(self, "_last_loss_parts", None)
                if parts is not None:
                    payload["train/loss_mse"] = parts["mse"]
                    payload["train/loss_stft"] = parts["stft"]
                    payload["train/loss_deriv"] = parts.get("deriv", 0.0)
                    payload["train/loss_band_stft"] = parts.get("band_stft", 0.0)
                self.wandb_logger.log(payload)
        self.scheduler_lr.step()

    @torch.no_grad()
    def validate(self, epoch: int) -> Optional[float]:
        if self.val_loader is None:
            return None
        self.model.eval()
        total = 0.0
        n = 0
        # Per-class running loss for breakdown reporting.
        per_class = defaultdict(lambda: [0.0, 0])  # ci -> [sum, count]
        for batch in tqdm(self.val_loader, desc=f"val epoch {epoch}"):
            mixed_patch, class_idx, alpha = self._batch_to_device(batch)

            # Cache one real (mixed) patch per class for the real-vs-generated comparison.
            self._cache_real_patches(mixed_patch, class_idx, alpha)

            with autocast(enabled=self.amp):
                latents = self._to_diffusion_input(mixed_patch)
                loss = self._diffusion_loss(latents, class_idx, alpha)
            B = mixed_patch.size(0)
            total += float(loss) * B
            n += B

            # Per-class accumulation: compute one extra cheap pass with the same batch
            # split by class. The batch-mean loss is a single scalar so we need per-sample.
            # Skip if all samples in this batch share the same class.
            uniq = torch.unique(class_idx).tolist()
            if len(uniq) > 1:
                for ci in uniq:
                    mask = (class_idx == ci)
                    sub_latents = latents[mask]
                    sub_alpha = alpha[mask]
                    sub_cls = class_idx[mask]
                    with autocast(enabled=self.amp):
                        sub_loss = self._diffusion_loss(sub_latents, sub_cls, sub_alpha)
                    per_class[int(ci)][0] += float(sub_loss) * int(mask.sum())
                    per_class[int(ci)][1] += int(mask.sum())
            else:
                ci = int(uniq[0])
                per_class[ci][0] += float(loss) * B
                per_class[ci][1] += B
        avg = total / max(1, n)

        if self.wandb_logger:
            log_payload = {"val/loss": avg, "epoch": epoch}
            for ci, (s, c) in per_class.items():
                if c > 0:
                    log_payload[f"val/loss_class_{ci}"] = s / c
            self.wandb_logger.log(log_payload)
        return avg

    @torch.no_grad()
    def _cache_real_patches(self, mixed_patch, class_idx, alpha):
        """Populate self._real_patches_by_class lazily — one real patch per class."""
        for i in range(mixed_patch.size(0)):
            ci = int(class_idx[i])
            if ci not in self._real_patches_by_class:
                self._real_patches_by_class[ci] = (
                    mixed_patch[i:i + 1].clone(),
                    float(alpha[i]),
                )

    def save_checkpoint(self, epoch: int, tag: str = "best") -> str:
        path = os.path.join(self.checkpoint_dir, f"diffusion_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "noise_scheduler_config": self.noise_scheduler.config,
        }, path)
        return path

    @torch.no_grad()
    def _sample_with_ema(self, class_idx: int, alpha: float, latent_w: int) -> torch.Tensor:
        """Sample one latent with EMA weights at a given (class, alpha). Returns flattened latent."""
        # Snapshot live weights and load EMA into the model.
        live_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            self.ema.copy_to(self.model)
            self.model.eval()
            in_ch = self.model.latent_channels * self.model.spatial_h
            cls = torch.tensor([class_idx], dtype=torch.long, device=self.device)
            cls_null = torch.tensor([self.null_idx], dtype=torch.long, device=self.device)
            a = torch.tensor([alpha], dtype=torch.float32, device=self.device)
            x = torch.randn(1, in_ch, latent_w, device=self.device)
            self._sampler.set_timesteps(self.sample_log_steps, device=self.device)
            for t in self._sampler.timesteps:
                t_b = t.expand(1).long().to(self.device)
                eps_uc = self.model(x, t_b, cls_null, a)
                eps_cc = self.model(x, t_b, cls, a)
                pred = eps_uc + self.sample_log_cfg_scale * (eps_cc - eps_uc)
                x = self._sampler.step(pred, t, x).prev_sample
            return x
        finally:
            self.model.load_state_dict(live_state)
            self.model.train()

    @torch.no_grad()
    def _log_samples_image(self, epoch: int):
        """Generate one sample per alpha (under EMA) and log a waterfall+spec panel each.

        Works in both latent mode (decode via VAE) and pixel mode (sample is the patch).
        """
        if self.wandb_logger is None:
            return
        if self.vae is not None:
            probe = torch.zeros(1, 1, self.spatial_h, self.patch_time, device=self.device)
            mu, _ = self.vae.encode(probe)
            latent_w = mu.shape[-1]
        else:
            # Pixel-space: the diffusion output IS the patch. W matches patch_time.
            latent_w = self.patch_time

        images = {}
        for a in self.sample_log_alphas:
            x_flat = self._sample_with_ema(self.sample_log_class_idx, float(a), latent_w)
            patch = self._decode_to_patch(x_flat).float().cpu().numpy()[0]  # [1, H, T]
            fig = plot_patch_panel(patch, title=f"epoch {epoch}  cls={self.sample_log_class_idx}  α={a:.2f}")
            try:
                images[f"samples/cls{self.sample_log_class_idx}_a{a:.2f}"] = self.wandb_logger.Image(fig)
            finally:
                import matplotlib.pyplot as plt
                plt.close(fig)
        images["epoch"] = epoch
        self.wandb_logger.log(images)

    @torch.no_grad()
    def _decode_to_patch(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Turn a flattened diffusion output [B, C*H, W] into a viewable patch [B, 1, H, T]."""
        latent = unflatten_latent(x_flat, self.latent_channels, self.spatial_h)
        if self.vae is not None:
            latent = latent / self.scaling_factor if self.scaling_factor != 0 else latent
            return self.vae.decode(latent)
        return latent  # pixel-space: already [B, 1, H, W=patch_time]

    @torch.no_grad()
    def _log_real_vs_generated(self, epoch: int):
        """Log richer per-class comparison panels: N varied generations + per-channel specs.

        Per class:
          real_vs_gen/cls{ci}_waterfalls  — N+1 waterfalls stacked (1 real, N generated)
          real_vs_gen/cls{ci}_spec_grid   — (N+1) rows x C channels grid of spectrograms
        All generations use the same (class, alpha) but different random init -> showcase
        the variety the model produces.
        """
        if self.wandb_logger is None or not self._real_patches_by_class:
            return
        if self.vae is not None:
            probe = torch.zeros(1, 1, self.spatial_h, self.patch_time, device=self.device)
            mu, _ = self.vae.encode(probe)
            latent_w = mu.shape[-1]
            fs = 1000  # latent is conceptually 1 kHz patches even if compressed
        else:
            latent_w = self.patch_time
            fs = 1000

        n_gens = max(1, self.sample_log_n_gens)
        images = {}
        import matplotlib.pyplot as plt

        for ci, (real_patch, real_alpha) in self._real_patches_by_class.items():
            # Generate N independent samples at the same (class, alpha) - different RNG
            gens = []
            for _ in range(n_gens):
                x_flat = self._sample_with_ema(ci, real_alpha, latent_w)
                gen_patch = self._decode_to_patch(x_flat)
                gens.append(gen_patch.float().cpu().numpy()[0])  # [1, H, T] or [H, T]

            real_np = real_patch.float().cpu().numpy()[0]   # [1, H, T]
            titles = [f"REAL  cls={ci}  α={real_alpha:.2f}"] + [
                f"GEN {i + 1}" for i in range(n_gens)
            ]

            # Waterfalls stacked
            try:
                fig_wf = plot_multi_waterfall([real_np] + gens, titles, fs=fs)
                images[f"real_vs_gen/cls{ci}_waterfalls"] = self.wandb_logger.Image(fig_wf)
            finally:
                plt.close(fig_wf)

            # Per-channel spectrograms grid
            try:
                row_titles = ["real"] + [f"gen{i + 1}" for i in range(n_gens)]
                fig_sg = plot_per_channel_spectrograms(
                    [real_np] + gens, row_titles, fs=fs,
                )
                images[f"real_vs_gen/cls{ci}_spec_grid"] = self.wandb_logger.Image(fig_sg)
            finally:
                plt.close(fig_sg)

        images["epoch"] = epoch
        self.wandb_logger.log(images)

    def fit(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, tag="best")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, tag=f"epoch{epoch + 1}")
            if (
                self.sample_every_n_epochs > 0
                and (epoch + 1) % self.sample_every_n_epochs == 0
                and self.wandb_logger is not None
            ):
                self._log_samples_image(epoch)
                self._log_real_vs_generated(epoch)
