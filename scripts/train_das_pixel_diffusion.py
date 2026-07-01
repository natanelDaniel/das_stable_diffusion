"""
Pixel-space stable diffusion for DAS patches.

Same DDPM + class + alpha conditioning as the latent-diffusion path, but the UNet
runs directly on the [1, 8, 16384] patches — no VAE encode/decode. Much simpler
pipeline (no Stage-1 to train) at the cost of more compute per step (W=16384
instead of W=256 at the latent bottleneck).

Trade-offs:
  + No VAE to train; no scaling_factor to set; no normalization mismatch failure mode.
  + One config, one run.
  - Each iter does ~64x more time-domain work than the latent path.
  - Sampling is also slower (DDIM denoise loop runs at W=16384).

Usage:
    python scripts/train_das_pixel_diffusion.py --config configs/pixel_diffusion_config.yaml
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.splits import recording_level_split  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import DASLatentPatchDataset  # noqa: E402
from src.models.das_diffusion_unet import DASDiffusionUNet  # noqa: E402
from src.training.diffusion_trainer import DASDiffusionTrainer  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(cfg: dict, stage: str = "pixel_diffusion"):
    stage_cfg = cfg["training"][stage]
    project = stage_cfg.get("wandb_project")
    if not project:
        return None
    try:
        import wandb  # noqa: WPS433
    except ImportError:
        print("WARNING: wandb not installed; logging disabled.")
        return None
    run_name = stage_cfg.get("wandb_run_name") or f"{stage}-{datetime.now():%Y%m%d-%H%M%S}"
    wandb.init(project=project, name=run_name, config=cfg, resume="allow")
    return wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pixel_diffusion_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    diff_cfg = cfg["model"]["pixel_diffusion"]
    train_cfg = cfg["training"]["pixel_diffusion"]

    set_seed(data_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    normalize = (data_cfg["normalize"]["mean"], data_cfg["normalize"]["std"])
    dataset = DASLatentPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        event_offset_range=tuple(data_cfg["event_offset_range"]),
        decimation=data_cfg["decimation"],
        classes=data_cfg["classes"],
        normalize=normalize,
        seed=data_cfg["seed"],
        return_mixed=True,
        mix_alpha_range=tuple(data_cfg.get("mix_alpha_range", [0.0, 1.0])),
        cache_in_ram=data_cfg.get("cache_in_ram", False),
        target_sample_rate=data_cfg.get("target_sample_rate", 1000),
    )
    print(f"Total samples: {len(dataset)}  noise pool: {len(dataset._noise_samples)}")

    train_ds, val_ds, _ = recording_level_split(
        dataset,
        val_frac=data_cfg["val_split"],
        test_frac=data_cfg["test_split"],
        seed=data_cfg["seed"],
    )

    train_labels = np.array(
        [int(dataset.samples[train_ds.indices[i]][3]) for i in range(len(train_ds))]
    )
    class_counts = np.bincount(train_labels, minlength=len(data_cfg["classes"]))
    weights = 1.0 / np.clip(class_counts[train_labels], 1, None)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    nw = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], sampler=sampler,
        num_workers=nw, pin_memory=(device == "cuda"),
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=(device == "cuda"),
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    # Pixel-space: latent_channels=1, spatial_h=patch_channels, W=patch_time.
    # The UNet's flatten step turns [B, 1, 8, 16384] into [B, 8, 16384] (input channels = 8).
    dilations = diff_cfg.get("dilations_per_level")
    use_attn = diff_cfg.get("use_attention_per_level")
    model = DASDiffusionUNet(
        latent_channels=1,
        spatial_h=data_cfg["patch_channels"],
        base_channels=diff_cfg["base_channels"],
        channel_mults=tuple(diff_cfg["channel_mults"]),
        num_res_blocks=diff_cfg["num_res_blocks"],
        num_classes=diff_cfg["num_classes"],
        cond_dim=diff_cfg["cond_dim"],
        kernel=diff_cfg.get("kernel", 5),
        dilations_per_level=[list(d) for d in dilations] if dilations else None,
        use_attention_per_level=list(use_attn) if use_attn else None,
        attention_in_mid=diff_cfg.get("attention_in_mid", False),
        attention_heads=diff_cfg.get("attention_heads", 4),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if args.dry_run:
        batch = next(iter(train_loader))
        mixed_patch, class_idx, alpha = batch
        model = model.to(device)
        x = mixed_patch.to(device)
        from src.training.diffusion_trainer import flatten_latent
        x_flat = flatten_latent(x)
        t = torch.zeros(x_flat.size(0), device=device, dtype=torch.long)
        cls = class_idx.to(device).long()
        a = torch.as_tensor(alpha, device=device, dtype=torch.float32)
        pred = model(x_flat, t, cls, a)
        print(f"Dry run OK")
        print(f"  patch in   : {tuple(x.shape)}")
        print(f"  flattened  : {tuple(x_flat.shape)}")
        print(f"  pred out   : {tuple(pred.shape)}")
        return

    wandb_logger = None if args.no_wandb else init_wandb(cfg, "pixel_diffusion")

    trainer = DASDiffusionTrainer(
        model=model,
        vae=None,                        # pixel-space: no VAE
        scaling_factor=1.0,              # ignored when vae=None
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip=train_cfg["grad_clip"],
        amp=train_cfg["amp"],
        cfg_dropout=diff_cfg["cfg_dropout"],
        alpha_cfg_dropout=diff_cfg.get("alpha_cfg_dropout", diff_cfg["cfg_dropout"]),
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        prediction_type=diff_cfg["prediction_type"],
        beta_schedule=diff_cfg["beta_schedule"],
        ema_decay=diff_cfg["ema_decay"],
        checkpoint_dir=train_cfg["checkpoint_dir"],
        wandb_logger=wandb_logger,
        log_freq=train_cfg.get("wandb_log_freq", 100),
        sample_every_n_epochs=train_cfg.get("wandb_sample_every_n_epochs", 5),
        patch_time=data_cfg["patch_time"],
        lambda_stft=diff_cfg.get("lambda_stft", 0.0),
        stft_n_ffts=tuple(diff_cfg.get("stft_n_ffts", (1024, 2048))),
        sample_log_n_gens=train_cfg.get("wandb_sample_n_gens", 3),
        lambda_deriv=diff_cfg.get("lambda_deriv", 0.0),
        min_snr_gamma=diff_cfg.get("min_snr_gamma", None),
        lambda_band_stft=diff_cfg.get("lambda_band_stft", 0.0),
        band_stft_freq_max=diff_cfg.get("band_stft_freq_max", 50.0),
        band_stft_n_ffts=tuple(diff_cfg.get("band_stft_n_ffts", (1024, 2048))),
        sample_rate=int(data_cfg.get("target_sample_rate", 500)),
    )
    trainer.fit()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
