"""
Train DASResNetClassifier on DAS event patches.

Two runs (controlled experiment):
  baseline  — real DAS data only
  synthetic — real data + synthetic patches from the pixel-space diffusion model

Usage:
    python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run baseline
    python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run synthetic
    python scripts/train_das_cnn.py --config configs/cnn_classifier_config.yaml --run synthetic --no-wandb
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm
from typing import List
from torch.utils.data import ConcatDataset, DataLoader, Subset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import DASLatentPatchDataset  # noqa: E402
from src.data.splits import recording_level_split                      # noqa: E402
from src.models.das_cnn_classifier import DASResNetClassifier          # noqa: E402
from src.training.cnn_trainer import CNNTrainer, SyntheticDASDataset, BackgroundMixupDataset  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(cfg: dict, run_mode: str):
    train_cfg = cfg["training"]
    project = train_cfg.get("wandb_project")
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed — logging disabled.")
        return None
    run_name = f"cnn_{run_mode}_{datetime.now():%Y%m%d-%H%M%S}"
    wandb.init(project=project, name=run_name, config=cfg, resume="allow")
    return wandb


def generate_synthetic_data(cfg: dict, device: str) -> SyntheticDASDataset:
    """Sample patches from the pixel-space diffusion model and return a Dataset.

    If a cache file already exists (synthetic.cache_path in config, or the default
    path derived from the checkpoint name), it is loaded directly — no generation.
    After generation the cache is saved so future runs skip generation entirely.
    """
    syn_cfg = cfg["synthetic"]
    data_cfg = cfg["data"]

    # Determine cache path.
    default_cache = os.path.join(
        os.path.dirname(syn_cfg["diffusion_ckpt"]),
        f"synthetic_cache_n{syn_cfg['n_per_alpha']}_a{len(syn_cfg['alpha_values'])}.pt",
    )
    cache_path = syn_cfg.get("cache_path", default_cache)

    if os.path.exists(cache_path):
        print(f"Loading synthetic dataset from cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return SyntheticDASDataset(data["patches"], data["labels"])

    from diffusers import DDIMScheduler

    from scripts.generate_pixel_samples import load_pixel_diffusion
    from scripts.generate_diffusion_samples import sample
    from src.training.diffusion_trainer import unflatten_latent

    # Load the diffusion model once — read architecture from its own config file.
    pixel_diff_config_path = syn_cfg.get(
        "pixel_diffusion_config", "configs/pixel_diffusion_config.yaml"
    )
    with open(pixel_diff_config_path) as _f:
        diff_pixel_cfg = yaml.safe_load(_f)

    ckpt_path = syn_cfg["diffusion_ckpt"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {ckpt_path}\n"
            f"Train the pixel diffusion model first with scripts/train_das_pixel_diffusion.py"
        )

    print(f"Loading pixel diffusion checkpoint: {ckpt_path}")
    diffusion = load_pixel_diffusion(
        diff_pixel_cfg, ckpt_path, device, use_ema=syn_cfg.get("use_ema", True)
    )

    # Rebuild scheduler to match how the model was trained.
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        prediction_type="v_prediction",
        beta_schedule="squaredcos_cap_v2",
        rescale_betas_zero_snr=True,
    )

    classes = data_cfg["classes"]
    n_per_alpha = int(syn_cfg["n_per_alpha"])
    alpha_values = list(syn_cfg["alpha_values"])
    cfg_scale = float(syn_cfg.get("cfg_scale", 7.5))
    alpha_cfg_scale = float(syn_cfg.get("alpha_cfg_scale", 1.0))
    steps = int(syn_cfg.get("steps", 50))
    patch_time = int(data_cfg["patch_time"])

    all_patches: list = []
    all_labels: list = []

    gen_batch_size = int(syn_cfg.get("gen_batch_size", 50))
    total_combos = len(classes) * len(alpha_values)
    n_batches = (n_per_alpha + gen_batch_size - 1) // gen_batch_size
    total_steps = total_combos * n_batches
    print(f"Generating synthetic patches: {len(classes)} classes × {len(alpha_values)} alpha × {n_per_alpha} samples")
    print(f"  batch_size={gen_batch_size}, {n_batches} batches/combo, {total_steps} total batches")
    with tqdm(total=total_steps, desc="generating", unit="batch") as pbar:
        for cls_idx, cls_name in enumerate(classes):
            for alpha in alpha_values:
                collected = 0
                while collected < n_per_alpha:
                    bs = min(gen_batch_size, n_per_alpha - collected)
                    pbar.set_postfix(cls=cls_name, alpha=f"{alpha:.3f}", n=f"{collected}/{n_per_alpha}")
                    x_flat = sample(
                        diffusion=diffusion,
                        class_idx=cls_idx,
                        alpha=float(alpha),
                        cfg_scale=cfg_scale,
                        alpha_cfg_scale=alpha_cfg_scale,
                        n_samples=bs,
                        steps=steps,
                        latent_w=patch_time,
                        scheduler=scheduler,
                        device=device,
                    )
                    # unflatten → [B, 1, C, T] in normalized space
                    patches_t = unflatten_latent(x_flat, 1, data_cfg["patch_channels"]).float().cpu()
                    for p in patches_t:
                        all_patches.append(p)
                        all_labels.append(cls_idx)
                    collected += bs
                    pbar.update(1)

    print(f"Synthetic dataset: {len(all_patches)} patches")
    print(f"Saving synthetic cache: {cache_path}")
    torch.save({"patches": all_patches, "labels": all_labels}, cache_path)
    return SyntheticDASDataset(all_patches, all_labels)



def generate_background_patches(cfg: dict, device: str) -> List[torch.Tensor]:
    """Generate background (regular/noise class) patches for mixup augmentation.

    Only the noise class is sampled across all configured alpha values.
    Returns a list of [1, C, T] float32 tensors in normalized space.
    Caches to disk so subsequent runs skip generation entirely.
    """
    syn_cfg = cfg["synthetic"]
    data_cfg = cfg["data"]

    classes = data_cfg["classes"]
    noise_cls_name = data_cfg.get("noise_class", "regular")
    noise_cls_idx = classes.index(noise_cls_name)

    alpha_values = list(syn_cfg["alpha_values"])
    n_per_alpha = int(syn_cfg["n_per_alpha"])

    default_cache = os.path.join(
        os.path.dirname(syn_cfg["diffusion_ckpt"]),
        f"bg_cache_n{n_per_alpha}_a{len(alpha_values)}.pt",
    )
    cache_path = syn_cfg.get("bg_cache_path", default_cache)

    if os.path.exists(cache_path):
        print(f"Loading background cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        bg = data["patches"]
        print(f"  {len(bg)} background patches loaded")
        return bg

    from diffusers import DDIMScheduler
    from scripts.generate_pixel_samples import load_pixel_diffusion
    from scripts.generate_diffusion_samples import sample
    from src.training.diffusion_trainer import unflatten_latent

    pixel_diff_config_path = syn_cfg.get(
        "pixel_diffusion_config", "configs/pixel_diffusion_config.yaml"
    )
    with open(pixel_diff_config_path) as _f:
        diff_pixel_cfg = yaml.safe_load(_f)

    ckpt_path = syn_cfg["diffusion_ckpt"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    print(f"Loading pixel diffusion checkpoint: {ckpt_path}")
    diffusion = load_pixel_diffusion(
        diff_pixel_cfg, ckpt_path, device, use_ema=syn_cfg.get("use_ema", True)
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        prediction_type="v_prediction",
        beta_schedule="squaredcos_cap_v2",
        rescale_betas_zero_snr=True,
    )

    gen_batch_size = int(syn_cfg.get("gen_batch_size", 50))
    n_batches = (n_per_alpha + gen_batch_size - 1) // gen_batch_size
    total_steps = len(alpha_values) * n_batches
    patch_time = int(data_cfg["patch_time"])

    print(
        f"Generating background patches: class='{noise_cls_name}' × "
        f"{len(alpha_values)} alpha × {n_per_alpha} samples"
    )

    all_patches: List[torch.Tensor] = []
    cfg_scale = float(syn_cfg.get("cfg_scale", 7.5))
    alpha_cfg_scale = float(syn_cfg.get("alpha_cfg_scale", 1.0))
    steps = int(syn_cfg.get("steps", 50))

    with tqdm(total=total_steps, desc="bg generation", unit="batch") as pbar:
        for alpha in alpha_values:
            collected = 0
            while collected < n_per_alpha:
                bs = min(gen_batch_size, n_per_alpha - collected)
                pbar.set_postfix(alpha=f"{alpha:.3f}", n=f"{collected}/{n_per_alpha}")
                x_flat = sample(
                    diffusion=diffusion,
                    class_idx=noise_cls_idx,
                    alpha=float(alpha),
                    cfg_scale=cfg_scale,
                    alpha_cfg_scale=alpha_cfg_scale,
                    n_samples=bs,
                    steps=steps,
                    latent_w=patch_time,
                    scheduler=scheduler,
                    device=device,
                )
                patches_t = unflatten_latent(x_flat, 1, data_cfg["patch_channels"]).float().cpu()
                for p in patches_t:
                    all_patches.append(p)
                collected += bs
                pbar.update(1)

    print(f"Background dataset: {len(all_patches)} patches")
    print(f"Saving background cache: {cache_path}")
    torch.save({"patches": all_patches}, cache_path)
    return all_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn_classifier_config.yaml")
    parser.add_argument("--run",
                        choices=["baseline", "synthetic", "synth_ratio_0.5", "synth_ratio_1.0"],
                        required=True,
                        help="baseline=real only; synthetic=bg mixup; synth_ratio_X=real+X*real synthetic events")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(data_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Run mode: {args.run}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    normalize = (data_cfg["normalize"]["mean"], data_cfg["normalize"]["std"])
    full_dataset = DASLatentPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        event_offset_range=tuple(data_cfg["event_offset_range"]),
        decimation=data_cfg["decimation"],
        classes=data_cfg["classes"],
        normalize=normalize,
        seed=data_cfg["seed"],
        return_mixed=False,       # classifier: no noise mixing
        cache_in_ram=data_cfg.get("cache_in_ram", False),
        target_sample_rate=data_cfg.get("target_sample_rate", 500),
    )
    print(f"Real dataset: {len(full_dataset)} samples")

    train_real, val_ds, _ = recording_level_split(
        full_dataset,
        val_frac=data_cfg["val_split"],
        test_frac=data_cfg["test_split"],
        seed=data_cfg["seed"],
    )

    # Class weights from training set (inverse frequency, like diffusion trainer).
    train_labels = np.array(
        [int(full_dataset.samples[train_real.indices[i]][3]) for i in range(len(train_real))]
    )
    n_cls = len(data_cfg["classes"])
    class_counts = np.bincount(train_labels, minlength=n_cls).astype(np.float32)
    class_weights = torch.tensor(1.0 / np.clip(class_counts, 1, None))
    class_weights /= class_weights.sum()    # normalize so weights sum to 1
    print(f"Class weights: {dict(zip(data_cfg['classes'], class_weights.numpy().round(4)))}")

    # Balanced sampler for real data.
    sample_weights = class_weights[train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(train_real),
        replacement=True,
    )

    # ------------------------------------------------------------------ #
    # Background mixup (run == "synthetic")
    # ------------------------------------------------------------------ #
    if args.run == "synthetic":
        bg_patches = generate_background_patches(cfg, device)
        mix_alpha = tuple(cfg["synthetic"].get("mix_alpha_range", [0.0, 0.5]))
        train_ds = BackgroundMixupDataset(
            train_real, bg_patches,
            alpha_range=mix_alpha,
            seed=data_cfg["seed"],
        )
        print(
            f"BackgroundMixup: {len(bg_patches)} bg patches, "
            f"mix alpha ~ Uniform{mix_alpha}"
        )
        # Keep WeightedRandomSampler — class distribution unchanged.
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg["batch_size"], sampler=sampler,
            num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"),
        )
    elif args.run in ("synth_ratio_0.5", "synth_ratio_1.0"):
        # Both modes share the same cache — run synth_ratio_1.0 first to generate it.
        # synth_ratio_1.0 uses the full pool; synth_ratio_0.5 uses the first half per class.
        synthetic_ds = generate_synthetic_data(cfg, device)
        syn_labels_arr = np.array(synthetic_ds.labels)

        if args.run == "synth_ratio_1.0":
            syn_indices = list(range(len(synthetic_ds)))
        else:  # synth_ratio_0.5 — first half of each class's patches
            syn_indices = []
            for cls in range(n_cls):
                cls_idx = np.where(syn_labels_arr == cls)[0]
                syn_indices.extend(cls_idx[: len(cls_idx) // 2].tolist())

        syn_subset = Subset(synthetic_ds, syn_indices)
        train_ds = ConcatDataset([train_real, syn_subset])

        # Class weights over the combined dataset
        syn_labels = syn_labels_arr[syn_indices]
        combined_labels = np.concatenate([train_labels, syn_labels])
        combined_counts = np.bincount(combined_labels, minlength=n_cls).astype(np.float32)
        combined_weights = torch.tensor(1.0 / np.clip(combined_counts, 1, None))
        combined_weights /= combined_weights.sum()
        class_weights = combined_weights

        sample_weights_combined = combined_weights[combined_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights_combined.double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg["batch_size"], sampler=sampler,
            num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"),
        )
        print(f"Ratio-synthetic ({args.run}): {len(train_real)} real + {len(syn_indices)} synthetic = {len(train_ds)} total")
        print(f"Class weights (combined): {dict(zip(data_cfg['classes'], class_weights.numpy().round(4)))}")

    else:
        train_loader = DataLoader(
            train_real, batch_size=train_cfg["batch_size"], sampler=sampler,
            num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"),
        )

    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = DASResNetClassifier(
        n_classes=model_cfg["n_classes"],
        embed_dim=model_cfg["embed_dim"],
        dropout=model_cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    wandb_logger = None if args.no_wandb else init_wandb(cfg, args.run)

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=data_cfg["classes"],
        device=device,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip=train_cfg["grad_clip"],
        amp=train_cfg["amp"],
        scheduler_T0=train_cfg["scheduler_T0"],
        checkpoint_dir=os.path.join(train_cfg["checkpoint_dir"], args.run),
        wandb_logger=wandb_logger,
        log_every_n_epochs=train_cfg["log_every_n_epochs"],
        class_weights=class_weights,
        run_name=args.run,
    )
    noise_std = float(train_cfg.get("background_noise_std", 0.0))
    if noise_std > 0.0:
        trainer.add_background_noise(noise_std)
        print(f"Background noise augmentation: std={noise_std} (normalized space)")
    trainer.fit()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
