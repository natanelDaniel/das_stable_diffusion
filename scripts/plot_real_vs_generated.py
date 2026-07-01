"""
plot_real_vs_generated.py

Presentation-quality comparison figure:
    4 classes x 3 columns [Real | Generated-1 | Generated-2]
    Each cell: waterfall (8-channel DAS) at alpha=0.0 (clean event).

Requires a trained diffusion checkpoint.

Usage:
    python scripts/plot_real_vs_generated.py
    python scripts/plot_real_vs_generated.py --ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.generate_pixel_samples import load_pixel_diffusion  # noqa: E402
from scripts.generate_diffusion_samples import sample  # noqa: E402
from src.data.das_latent_patch_dataset import DASLatentPatchDataset  # noqa: E402
from src.evaluation.plotting import _draw_waterfall  # noqa: E402
from src.training.diffusion_trainer import unflatten_latent  # noqa: E402

ALPHA = 0.0
CFG_SCALE = 7.5

CLASS_LABELS = {
    "car": "Car",
    "regular": "Background",
    "running": "Running",
    "walk": "Walk",
}


def fetch_real_patch(ds: DASLatentPatchDataset, class_name: str) -> np.ndarray:
    target_idx = ds.classes.index(class_name)
    for i in range(len(ds)):
        if ds.samples[i][3] == target_idx:
            patch, _ = ds[i]
            return patch.numpy()  # [1, C, T]
    raise RuntimeError(f"No samples found for class '{class_name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pixel_diffusion_config.yaml")
    parser.add_argument("--ckpt", default="checkpoints/das_pixel_diffusion/diffusion_best.pt")
    parser.add_argument("--out", default="figures/real_vs_generated.png")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--cfg_scale", type=float, default=CFG_SCALE)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    diff_cfg = cfg["model"]["pixel_diffusion"]
    fs = int(data_cfg.get("target_sample_rate", 1000))
    classes = data_cfg["classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    normalize = (data_cfg["normalize"]["mean"], data_cfg["normalize"]["std"])
    ds = DASLatentPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        event_offset_range=tuple(data_cfg["event_offset_range"]),
        decimation=data_cfg["decimation"],
        classes=classes,
        normalize=normalize,
        seed=data_cfg["seed"],
        return_mixed=False,
        cache_in_ram=data_cfg.get("cache_in_ram", False),
        target_sample_rate=fs,
    )
    print(f"Dataset: {len(ds)} samples")

    diffusion = load_pixel_diffusion(cfg, args.ckpt, device, use_ema=True)
    print(f"Loaded EMA weights from {args.ckpt}")

    scheduler = DDIMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        prediction_type=diff_cfg["prediction_type"],
        beta_schedule=diff_cfg["beta_schedule"],
        rescale_betas_zero_snr=True,
    )

    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, 3, figsize=(16, 3.2 * n_classes))
    fig.patch.set_facecolor("#f8f9fa")

    col_headers = ["Real", "Generated  (sample 1)", "Generated  (sample 2)"]
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=13, fontweight="bold", pad=10,
                               color="#1a3a5c")

    for row, class_name in enumerate(classes):
        print(f"  class={class_name} …")

        real = fetch_real_patch(ds, class_name)  # [1, C, T]
        _draw_waterfall(axes[row, 0], real[0], fs=fs)

        class_idx = classes.index(class_name)
        x_flat = sample(
            diffusion=diffusion,
            class_idx=class_idx,
            alpha=float(args.alpha),
            cfg_scale=args.cfg_scale,
            alpha_cfg_scale=1.0,
            n_samples=2,
            steps=args.steps,
            latent_w=data_cfg["patch_time"],
            scheduler=scheduler,
            device=device,
        )
        patches_gen = unflatten_latent(x_flat, 1, diffusion.spatial_h).float().cpu().numpy()
        _draw_waterfall(axes[row, 1], patches_gen[0][0], fs=fs)
        _draw_waterfall(axes[row, 2], patches_gen[1][0], fs=fs)

        label = CLASS_LABELS.get(class_name, class_name.capitalize())
        axes[row, 0].set_ylabel(f"{label}\n\nfiber channel", fontsize=11,
                                fontweight="bold", color="#1a3a5c")
        for col in range(1, 3):
            axes[row, col].set_ylabel("fiber channel", fontsize=9)

    fig.suptitle(
        f"Real vs. Diffusion-Generated DAS Patches  (α = {args.alpha:.2f}, CFG = {args.cfg_scale})",
        fontsize=15, fontweight="bold", y=1.01, color="#1a3a5c",
    )
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
