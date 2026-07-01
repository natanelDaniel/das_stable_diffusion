"""
Train DASResNetClassifier with full-class synthetic augmentation at varying ratios.

Generates synthetic patches for ALL 4 classes via the pixel diffusion model,
then combines real training data with synthetic at two ratios:
  --ratio 0.5  ->  1 real : 0.5 synthetic
  --ratio 1.0  ->  1 real : 1.0 synthetic

The synthetic pool (20k patches total) is generated once and cached.
For ratio 1.0 the pool is sampled with ~2x replacement on average.

Compare W&B runs against the baseline run (no synthetic) for a scaling analysis.

Usage:
    python scripts/train_das_cnn_synth_ratio.py --ratio 0.5
    python scripts/train_das_cnn_synth_ratio.py --ratio 1.0
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import DASLatentPatchDataset
from src.data.splits import recording_level_split
from src.models.das_cnn_classifier import DASResNetClassifier
from src.training.cnn_trainer import CNNTrainer, SyntheticDASDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(cfg: dict, ratio: float):
    train_cfg = cfg["training"]
    project = train_cfg.get("wandb_project")
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed — logging disabled.")
        return None
    ratio_str = f"{ratio:.1f}".replace(".", "p")
    run_name = f"cnn_synth_ratio_{ratio_str}_{datetime.now():%Y%m%d-%H%M%S}"
    wandb.init(project=project, name=run_name, config=cfg, resume="allow")
    return wandb


def generate_full_synthetic(cfg: dict, device: str) -> SyntheticDASDataset:
    """Generate synthetic patches for ALL classes and cache to disk.

    Pool size: n_per_alpha * len(alpha_values) * len(classes)
    With defaults: 500 * 10 * 4 = 20,000 patches.
    """
    syn_cfg = cfg["synth_ratio"]
    data_cfg = cfg["data"]

    n_per_alpha = int(syn_cfg["n_per_alpha"])
    alpha_values = list(syn_cfg["alpha_values"])
    classes = data_cfg["classes"]

    default_cache = os.path.join(
        os.path.dirname(syn_cfg["diffusion_ckpt"]),
        f"synth_ratio_cache_n{n_per_alpha}_a{len(alpha_values)}.pt",
    )
    cache_path = syn_cfg.get("cache_path", default_cache)

    if os.path.exists(cache_path):
        print(f"Loading synthetic pool from cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        ds = SyntheticDASDataset(data["patches"], data["labels"])
        print(f"  {len(ds)} patches loaded ({len(classes)} classes)")
        return ds

    from diffusers import DDIMScheduler
    from tqdm import tqdm

    from scripts.generate_diffusion_samples import sample
    from scripts.generate_pixel_samples import load_pixel_diffusion
    from src.training.diffusion_trainer import unflatten_latent

    pixel_diff_config_path = syn_cfg.get("pixel_diffusion_config", "configs/pixel_diffusion_config.yaml")
    with open(pixel_diff_config_path) as f:
        diff_pixel_cfg = yaml.safe_load(f)

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
    patch_time = int(data_cfg["patch_time"])
    cfg_scale = float(syn_cfg.get("cfg_scale", 7.5))
    alpha_cfg_scale = float(syn_cfg.get("alpha_cfg_scale", 1.0))
    steps = int(syn_cfg.get("steps", 50))
    n_batches = (n_per_alpha + gen_batch_size - 1) // gen_batch_size
    total_steps = len(classes) * len(alpha_values) * n_batches

    print(
        f"Generating: {len(classes)} classes x {len(alpha_values)} alphas x {n_per_alpha} = "
        f"{len(classes) * len(alpha_values) * n_per_alpha} patches total"
    )

    all_patches: list = []
    all_labels: list = []

    with tqdm(total=total_steps, desc="generating", unit="batch") as pbar:
        for cls_idx, cls_name in enumerate(classes):
            for alpha in alpha_values:
                collected = 0
                while collected < n_per_alpha:
                    bs = min(gen_batch_size, n_per_alpha - collected)
                    pbar.set_postfix(cls=cls_name, alpha=f"{alpha:.2f}", n=f"{collected}/{n_per_alpha}")
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
                    patches_t = unflatten_latent(x_flat, 1, data_cfg["patch_channels"]).float().cpu()
                    for p in patches_t:
                        all_patches.append(p)
                        all_labels.append(cls_idx)
                    collected += bs
                    pbar.update(1)

    print(f"Synthetic pool: {len(all_patches)} patches")
    print(f"Saving cache: {cache_path}")
    torch.save({"patches": all_patches, "labels": all_labels}, cache_path)
    return SyntheticDASDataset(all_patches, all_labels)


def sample_stratified(
    synthetic_ds: SyntheticDASDataset,
    n_total: int,
    n_classes: int,
    seed: int,
) -> Subset:
    """Sample n_total patches stratified by class. Uses replacement when pool < needed."""
    rng = np.random.default_rng(seed)
    labels = np.array(synthetic_ds.labels)
    n_per_cls = n_total // n_classes
    selected: list = []

    for cls in range(n_classes):
        cls_indices = np.where(labels == cls)[0]
        replace = len(cls_indices) < n_per_cls
        chosen = rng.choice(cls_indices, size=n_per_cls, replace=replace)
        selected.extend(chosen.tolist())

    # Fill remainder (rounding) uniformly
    remainder = n_total - len(selected)
    if remainder > 0:
        extra = rng.choice(len(synthetic_ds), size=remainder, replace=True)
        selected.extend(extra.tolist())

    return Subset(synthetic_ds, selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn_classifier_config.yaml")
    parser.add_argument(
        "--ratio", type=float, required=True,
        help="Synthetic-to-real ratio: 0.5, 1.0, or 2.0",
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.ratio not in (0.5, 1.0):
        raise ValueError(f"--ratio must be 0.5 or 1.0 (got {args.ratio})")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(data_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Ratio: 1:{args.ratio}")

    # ------------------------------------------------------------------ #
    # Real dataset + split
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
        return_mixed=False,
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

    # ------------------------------------------------------------------ #
    # Synthetic pool -> subsample for this ratio
    # ------------------------------------------------------------------ #
    synthetic_ds = generate_full_synthetic(cfg, device)

    n_synthetic = int(len(train_real) * args.ratio)
    n_cls = len(data_cfg["classes"])
    print(
        f"Real train: {len(train_real)}  |  Synthetic to add: {n_synthetic}  "
        f"|  Pool: {len(synthetic_ds)}  |  Repetition factor: {n_synthetic / max(len(synthetic_ds), 1):.2f}x"
    )

    synth_subset = sample_stratified(synthetic_ds, n_synthetic, n_cls, seed=data_cfg["seed"])

    # ------------------------------------------------------------------ #
    # Combined dataset + WeightedRandomSampler
    # ------------------------------------------------------------------ #
    train_ds = ConcatDataset([train_real, synth_subset])

    real_labels = np.array([
        int(full_dataset.samples[train_real.indices[i]][3]) for i in range(len(train_real))
    ])
    synth_labels = np.array([synthetic_ds.labels[j] for j in synth_subset.indices])
    combined_labels = np.concatenate([real_labels, synth_labels])

    class_counts = np.bincount(combined_labels, minlength=n_cls).astype(np.float32)
    class_weights = torch.tensor(1.0 / np.clip(class_counts, 1, None))
    class_weights /= class_weights.sum()
    print(f"Class weights: {dict(zip(data_cfg['classes'], class_weights.numpy().round(4)))}")

    sample_weights = class_weights[combined_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(train_ds),
        replacement=True,
    )

    nw = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], sampler=sampler,
        num_workers=nw, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=(device == "cuda"),
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

    wandb_logger = None if args.no_wandb else init_wandb(cfg, args.ratio)

    # Ratio-specific checkpoint dir so runs don't overwrite each other
    ratio_str = f"{args.ratio:.1f}".replace(".", "p")
    ckpt_dir = os.path.join(train_cfg["checkpoint_dir"], f"synth_ratio_{ratio_str}")

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
        checkpoint_dir=ckpt_dir,
        wandb_logger=wandb_logger,
        log_every_n_epochs=train_cfg["log_every_n_epochs"],
        class_weights=class_weights,
        run_name=ratio_str,
    )
    noise_std = float(train_cfg.get("background_noise_std", 0.0))
    if noise_std > 0.0:
        trainer.add_background_noise(noise_std)
        print(f"Background noise augmentation: std={noise_std}")
    trainer.fit()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
