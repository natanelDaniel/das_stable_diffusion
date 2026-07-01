"""
DASVAEv2Trainer — Stage-1 trainer for the unconditional VAE used by latent diffusion.

Loss: MSE + lambda_stft * multi-scale STFT + beta * KL (beta = 1e-3, fixed; no warmup).
Uses AdamW + cosine LR schedule, grad-clip 1.0, optional AMP, optional wandb.

After training, call `compute_latent_scaling_factor()` to derive `1 / sigma_latent`,
which the diffusion stage needs to multiply latents into approximately unit variance.
"""

import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.plotting import plot_input_vs_recon
from src.models.das_vae_v2 import DASVAEv2, vae_loss


class DASVAEv2Trainer:
    def __init__(
        self,
        model: DASVAEv2,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: str = "cuda",
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        beta: float = 1e-3,
        lambda_stft: float = 1.0,
        stft_n_ffts: Sequence[int] = (256, 1024, 4096),
        grad_clip: float = 1.0,
        amp: bool = True,
        checkpoint_dir: str = "checkpoints/das_vae_v2/",
        wandb_logger: Optional[object] = None,
        log_freq: int = 50,
    ):
        # Lets cuDNN pick the fastest conv kernel for each input shape on first iter.
        # After the warmup pass, subsequent iters use the cached fast kernel.
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            # channels_last layout lets Tensor Cores run conv kernels at full fp16
            # throughput; ~20-30% speedup on conv-heavy networks.
            self.use_channels_last = True
        else:
            self.use_channels_last = False

        self.model = model.to(device)
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.beta = beta
        self.lambda_stft = lambda_stft
        self.stft_n_ffts = tuple(stft_n_ffts)
        self.grad_clip = grad_clip
        self.amp = amp and device.startswith("cuda")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.wandb_logger = wandb_logger
        self.log_freq = log_freq

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.scaler = GradScaler(enabled=self.amp)

        self.global_step = 0
        self.best_val_loss = float("inf")
        # Cached fixed validation patch for reproducible recon plots.
        self._fixed_val_patch: Optional[torch.Tensor] = None

    def _forward_loss(self, patch: torch.Tensor):
        if self.use_channels_last:
            patch = patch.to(memory_format=torch.channels_last)
        with autocast(enabled=self.amp):
            x_hat, mu, logvar = self.model(patch)
            loss, parts = vae_loss(
                patch, x_hat, mu, logvar,
                beta=self.beta,
                lambda_stft=self.lambda_stft,
                stft_n_ffts=self.stft_n_ffts,
            )
        return loss, parts

    def train_epoch(self, epoch: int):
        self.model.train()
        for batch in tqdm(self.train_loader, desc=f"train epoch {epoch}"):
            patch, _ = batch
            patch = patch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss, parts = self._forward_loss(patch)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            if self.wandb_logger is not None and self.global_step % self.log_freq == 0:
                self.wandb_logger.log({
                    "train/total": float(parts["total"]),
                    "train/mse": float(parts["mse"]),
                    "train/stft": float(parts["stft"]),
                    "train/kl": float(parts["kl"]),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.global_step,
                })
        self.scheduler.step()

    @torch.no_grad()
    def validate(self, epoch: int) -> Optional[float]:
        if self.val_loader is None:
            return None
        self.model.eval()
        total = 0.0
        n = 0
        for batch in tqdm(self.val_loader, desc=f"val epoch {epoch}"):
            patch, _ = batch
            patch = patch.to(self.device, non_blocking=True)
            if self._fixed_val_patch is None:
                # Cache a single sample from the first val batch — same across epochs.
                self._fixed_val_patch = patch[0:1].clone()
            loss, _ = self._forward_loss(patch)
            total += float(loss) * patch.size(0)
            n += patch.size(0)
        avg = total / max(1, n)
        if self.wandb_logger is not None:
            self.wandb_logger.log({"val/total": avg, "epoch": epoch})
            self._log_recon_image(epoch)
        return avg

    @torch.no_grad()
    def _log_recon_image(self, epoch: int):
        """Log a 2x2 input/recon waterfall + spectrogram panel to W&B."""
        if self.wandb_logger is None or self._fixed_val_patch is None:
            return
        x = self._fixed_val_patch
        x_hat, _, _ = self.model(x)
        fig = plot_input_vs_recon(
            x.float().cpu().numpy()[0],     # [1, C, T] -> [C, T]
            x_hat.float().cpu().numpy()[0],
            title=f"epoch {epoch}",
        )
        try:
            self.wandb_logger.log({
                "val/recon": self.wandb_logger.Image(fig),
                "epoch": epoch,
            })
        finally:
            import matplotlib.pyplot as plt  # local import to keep top minimal
            plt.close(fig)

    def save_checkpoint(self, epoch: int, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"vae_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        return path

    def fit(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, tag="best")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, tag=f"epoch{epoch + 1}")


@torch.no_grad()
def compute_latent_scaling_factor(
    model: DASVAEv2,
    loader: DataLoader,
    device: str = "cuda",
    max_batches: int = 64,
) -> float:
    """Encode a subset of the training set, return 1 / std(latent) for use as the
    SD-style latent scaling factor. Diffusion training multiplies latents by this."""
    model.eval()
    total_sum = 0.0
    total_sq = 0.0
    total_n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        patch, _ = batch
        patch = patch.to(device)
        mu, _ = model.encode(patch)
        # Use the deterministic encoder mean (not a sample) for stable statistics.
        x = mu.detach().float().reshape(-1).cpu().numpy()
        total_sum += float(x.sum())
        total_sq += float((x * x).sum())
        total_n += x.size
    mean = total_sum / total_n
    var = total_sq / total_n - mean * mean
    sigma = max(1e-8, float(var) ** 0.5)
    return 1.0 / sigma
