"""
plot_alpha_sweep.py

Presentation-quality alpha-conditioning figure:
    5 alpha values x 4 classes grid of DAS waterfall plots.

Alpha semantics:
    alpha=0.0  →  pure event signal (conditioned class fully dominates)
    alpha=1.0  →  pure noise mixing  (event structure masked by noise)

The grid shows the model's smooth interpolation from clean event to noise.

Usage:
    python scripts/plot_alpha_sweep.py
    python scripts/plot_alpha_sweep.py --ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt
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
from src.evaluation.plotting import _draw_waterfall  # noqa: E402
from src.training.diffusion_trainer import unflatten_latent  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CFG_SCALE = 7.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pixel_diffusion_config.yaml")
    parser.add_argument("--ckpt", default="checkpoints/das_pixel_diffusion/diffusion_best.pt")
    parser.add_argument("--out", default="figures/alpha_sweep.png")
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

    diffusion = load_pixel_diffusion(cfg, args.ckpt, device, use_ema=True)
    print(f"Loaded EMA weights from {args.ckpt}")

    scheduler = DDIMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        prediction_type=diff_cfg["prediction_type"],
        beta_schedule=diff_cfg["beta_schedule"],
        rescale_betas_zero_snr=True,
    )

    n_alpha = len(ALPHAS)
    n_classes = len(classes)
    fig, axes = plt.subplots(n_alpha, n_classes, figsize=(4.5 * n_classes, 2.8 * n_alpha))
    fig.patch.set_facecolor("#f8f9fa")

    for col, cls in enumerate(classes):
        axes[0, col].set_title(cls.capitalize(), fontsize=13, fontweight="bold",
                               pad=8, color="#1a3a5c")

    total = n_alpha * n_classes
    done = 0
    for row, alpha in enumerate(ALPHAS):
        for col, class_name in enumerate(classes):
            done += 1
            print(f"  [{done}/{total}] alpha={alpha:.2f}  class={class_name}")
            class_idx = classes.index(class_name)
            x_flat = sample(
                diffusion=diffusion,
                class_idx=class_idx,
                alpha=alpha,
                cfg_scale=args.cfg_scale,
                alpha_cfg_scale=1.0,
                n_samples=1,
                steps=args.steps,
                latent_w=data_cfg["patch_time"],
                scheduler=scheduler,
                device=device,
            )
            patch = unflatten_latent(x_flat, 1, diffusion.spatial_h).float().cpu().numpy()[0]
            _draw_waterfall(axes[row, col], patch[0], fs=fs)

            if row < n_alpha - 1:
                axes[row, col].set_xlabel("")
                axes[row, col].set_xticklabels([])

        axes[row, 0].set_ylabel(
            f"α = {alpha:.2f}\n\nfiber channel",
            fontsize=11, fontweight="bold", color="#1a3a5c",
        )
        for col in range(1, n_classes):
            axes[row, col].set_ylabel("fiber channel", fontsize=9)

    fig.suptitle(
        "Alpha Conditioning: Event → Noise Transition  (CFG = 7.5)",
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
