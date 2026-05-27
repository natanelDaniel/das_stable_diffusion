"""
CVAE Trainer with WandB logging and checkpoint saving.

Usage:
    trainer = CVAETrainer(model, train_loader, val_loader, config)
    trainer.train()
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.training.losses import beta_vae_loss, linear_beta_schedule


class CVAETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config["training"]
        model_cfg = config["model"]

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["epochs"]
        )
        self.epochs = train_cfg["epochs"]
        self.target_beta = model_cfg["beta"]
        self.beta_warmup_epochs = model_cfg["beta_warmup_epochs"]
        self.checkpoint_dir = train_cfg["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")

    def _step(self, batch, beta: float):
        patch, _, onehot = batch
        patch = patch.to(self.device)
        onehot = onehot.to(self.device)
        x_hat, mu, logvar = self.model(patch, onehot)
        return beta_vae_loss(patch, x_hat, mu, logvar, beta)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        warmup_steps = self.beta_warmup_epochs * len(self.train_loader)
        step = epoch * len(self.train_loader)

        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)

        for batch in pbar:
            beta = linear_beta_schedule(step, warmup_steps, self.target_beta)
            self.optimizer.zero_grad()
            losses = self._step(batch, beta)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            step += 1
            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}", beta=f"{beta:.3f}")

        n = len(self.train_loader)
        return {f"train/{k}": v / n for k, v in running.items()}

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
            losses = self._step(batch, beta=self.target_beta)
            for k in running:
                running[k] += losses[k].item()
        n = len(self.val_loader)
        return {f"val/{k}": v / n for k, v in running.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"cvae_{tag}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )

    def train(self):
        wandb.init(
            project=self.config["training"]["wandb_project"],
            config=self.config,
            name=f"cvae-beta{self.target_beta}-latent{self.config['model']['latent_dim']}",
        )
        wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            self.scheduler.step()

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            wandb.log(metrics)

            val_loss = val_metrics["val/loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")
                print(f"  ✓ New best val_loss={val_loss:.4f} → saved checkpoint")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, tag=f"epoch{epoch+1}")
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"train_loss={train_metrics['train/loss']:.4f} | "
                      f"val_loss={val_loss:.4f}")

        wandb.finish()
        print(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")
