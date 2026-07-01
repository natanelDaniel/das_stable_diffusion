"""
Stage-1 latent-diffusion training entry point: trains the unconditional DASVAEv2.

Usage:
    python scripts/train_das_vae.py --config configs/latent_diffusion_config.yaml
    python scripts/train_das_vae.py --config configs/latent_diffusion_config.yaml --dry-run

After training completes, computes the SD-style latent scaling factor (1/sigma_latent)
and prints it. Paste the value into the config under `model.vae.scaling_factor`
before launching Stage 3 diffusion training.
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import DASLatentPatchDataset  # noqa: E402
from src.models.das_vae_v2 import DASVAEv2  # noqa: E402
from src.training.vae_trainer import DASVAEv2Trainer, compute_latent_scaling_factor  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def init_wandb(cfg: dict, stage: str = "vae"):
    """Initialize W&B for a stage; returns the wandb module or None if disabled."""
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
    parser.add_argument("--config", default="configs/latent_diffusion_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build dataset, run a single forward/backward, exit")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]["vae"]
    train_cfg = cfg["training"]["vae"]

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
        cache_in_ram=data_cfg.get("cache_in_ram", False),
        target_sample_rate=data_cfg.get("target_sample_rate", 1000),
    )
    print(f"Total samples: {len(dataset)}")

    n_test = int(len(dataset) * data_cfg["test_split"])
    n_val = int(len(dataset) * data_cfg["val_split"])
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, _test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(data_cfg["seed"]),
    )
    print(f"Split  train:{n_train}  val:{n_val}  test:{n_test}")

    nw = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=(device == "cuda"),
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=(device == "cuda"),
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    model = DASVAEv2(
        in_channels=1,
        encoder_channels=tuple(model_cfg["encoder_channels"]),
        latent_channels=model_cfg["latent_channels"],
        temporal_strides=tuple(model_cfg["temporal_strides"]),
        kernel_time=model_cfg["kernel_time"],
        kernel_space=model_cfg["kernel_space"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if args.dry_run:
        patch, _ = next(iter(train_loader))
        model = model.to(device)
        x = patch.to(device)
        x_hat, mu, logvar = model(x)
        print(f"Dry run OK")
        print(f"  input  : {tuple(x.shape)}")
        print(f"  latent : {tuple(mu.shape)}")
        print(f"  output : {tuple(x_hat.shape)}")
        return

    wandb_logger = None if args.no_wandb else init_wandb(cfg, "vae")

    trainer = DASVAEv2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        beta=model_cfg["beta"],
        lambda_stft=model_cfg["lambda_stft"],
        stft_n_ffts=model_cfg["stft_n_ffts"],
        grad_clip=train_cfg["grad_clip"],
        amp=train_cfg["amp"],
        checkpoint_dir=train_cfg["checkpoint_dir"],
        wandb_logger=wandb_logger,
        log_freq=train_cfg.get("wandb_log_freq", 50),
    )
    trainer.fit()

    scaling = compute_latent_scaling_factor(model, train_loader, device=device, max_batches=64)
    print("\n=== Stage-1 done ===")
    print(f"Suggested model.vae.scaling_factor for {args.config}: {scaling:.6f}")
    print("Paste this into the config before training Stage 3 diffusion.")

    if wandb_logger is not None:
        wandb_logger.log({"latent_scaling_factor": scaling})
        wandb_logger.finish()


if __name__ == "__main__":
    main()
