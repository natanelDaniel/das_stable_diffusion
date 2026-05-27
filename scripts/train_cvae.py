"""
CVAE Training Entry Point.

Usage:
    python scripts/train_cvae.py --config configs/cvae_config.yaml
    python scripts/train_cvae.py --config configs/cvae_config.yaml --dry-run
"""

import argparse
import os
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.das_patch_dataset import DASPatchDataset
from src.models.cvae import CVAE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cvae_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load 1 batch then exit (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Dataset ----
    data_cfg = cfg["data"]
    dataset = DASPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        bitmap_shift=data_cfg["bitmap_shift"],
        decimation=data_cfg["decimation"],
        classes=data_cfg["classes"],
        seed=cfg["training"]["seed"],
    )
    print(f"Total samples: {len(dataset)}")

    # Train / val / test split
    n_test = int(len(dataset) * data_cfg["test_split"])
    n_val = int(len(dataset) * data_cfg["val_split"])
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"]),
    )
    print(f"Split — train:{n_train}  val:{n_val}  test:{n_test}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        patch, cls_idx, onehot = batch
        print(f"Dry run OK — patch: {patch.shape}, onehot: {onehot.shape}")
        return

    # ---- Model ----
    from src.training.trainer import CVAETrainer  # noqa: deferred to avoid tqdm/wandb at dry-run
    model_cfg = cfg["model"]
    model = CVAE(
        latent_dim=model_cfg["latent_dim"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        enc_channels=tuple(model_cfg["encoder_channels"]),
        n_classes=model_cfg["num_classes"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Train ----
    trainer = CVAETrainer(model, train_loader, val_loader, cfg, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
