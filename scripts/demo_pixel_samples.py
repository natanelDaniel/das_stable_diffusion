"""
demo_pixel_samples.py

Side-by-side demo: for a set of (class, alpha) inputs, generate samples from the
best EMA checkpoint and pair them with a real validation patch of the same class.
Produces one PNG per (class, alpha) row showing: real | generated_1 | generated_2.

Useful for "what does the model actually produce" sanity checks without W&B.

Usage:
    python scripts/demo_pixel_samples.py \
        --config configs/pixel_diffusion_config.yaml \
        --ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt \
        --out demo_outputs/
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

from scripts.generate_pixel_samples import load_pixel_diffusion, sample  # noqa: E402
from src.data.das_latent_patch_dataset import DASLatentPatchDataset  # noqa: E402
from src.evaluation.plotting import (  # noqa: E402
    SPEC_DEFAULT_FREQ_MAX_HZ,
    _draw_spectrogram,
    _draw_waterfall,
)
from src.training.diffusion_trainer import unflatten_latent  # noqa: E402


# All 4 classes x 5 alpha levels
_CLASSES = ["car", "regular", "running", "walk"]
_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEMOS = [
    (cls, alpha, f"{cls}  a={alpha:.2f}")
    for cls in _CLASSES
    for alpha in _ALPHAS
]

N_SAMPLES_PER_DEMO = 4  # generated samples shown per figure (columns: real + N_SAMPLES)


def draw_all_channels(axes_col, patch_2d, col_title, fs):
    """Fill a column of axes (one row per channel) with 2D STFT spectrograms."""
    C = patch_2d.shape[0]
    for ch in range(C):
        ax = axes_col[ch]
        _draw_spectrogram(ax, patch_2d[ch], fs=fs,
                          freq_max=SPEC_DEFAULT_FREQ_MAX_HZ)
        ax.set_ylabel(f"ch{ch}\nHz", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        if ch == 0:
            ax.set_title(col_title, fontsize=9)
        if ch < C - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])


def fetch_real_patch(ds: DASLatentPatchDataset, class_name: str) -> np.ndarray:
    """Fetch a real (unmixed, class-aligned) patch for the given class name."""
    target_idx = ds.classes.index(class_name)
    for i in range(len(ds)):
        if ds.samples[i][3] == target_idx:
            patch, _ = ds[i]   # ds was constructed with return_mixed=False
            return patch.numpy()  # [1, C, T]
    raise RuntimeError(f"No samples found for class '{class_name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    diff_cfg = cfg["model"]["pixel_diffusion"]
    classes = data_cfg["classes"]
    fs = int(data_cfg.get("target_sample_rate", 1000))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # Dataset for fetching real patches (un-mixed so we see the actual class).
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
    print(f"Dataset: {len(ds)} samples for real comparison.")

    diffusion = load_pixel_diffusion(cfg, args.ckpt, device, use_ema=True)
    print(f"Loaded model from {args.ckpt} (EMA weights).")

    scheduler = DDIMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        prediction_type=diff_cfg["prediction_type"],
        beta_schedule=diff_cfg["beta_schedule"],
        rescale_betas_zero_snr=True,
    )

    os.makedirs(args.out, exist_ok=True)

    for class_name, alpha, label in DEMOS:
        print(f"\n=== {label}  (class={class_name}, alpha={alpha}) ===")
        class_idx = classes.index(class_name)
        x_flat = sample(
            diffusion=diffusion,
            class_idx=class_idx,
            alpha=float(alpha),
            cfg_scale=args.cfg_scale,
            alpha_cfg_scale=1.0,
            n_samples=N_SAMPLES_PER_DEMO,
            steps=args.steps,
            latent_w=data_cfg["patch_time"],
            scheduler=scheduler,
            device=device,
        )
        patches_gen = unflatten_latent(x_flat, 1, diffusion.spatial_h).float().cpu().numpy()
        real = fetch_real_patch(ds, class_name)  # [1, C, T]

        # 8 rows (one per channel) x (1 real + N_SAMPLES_PER_DEMO) columns
        n_cols = 1 + N_SAMPLES_PER_DEMO
        C = real[0].shape[0]
        fig, axes = plt.subplots(C, n_cols, figsize=(n_cols * 4, C * 2.0),
                                 gridspec_kw={"hspace": 0.35, "wspace": 0.3})
        draw_all_channels(axes[:, 0], real[0], f"REAL  class={class_name}", fs)
        for g in range(N_SAMPLES_PER_DEMO):
            draw_all_channels(axes[:, g + 1], patches_gen[g][0],
                              f"GEN {g + 1}  a={alpha}  cfg={args.cfg_scale}", fs)
        fig.suptitle(f"{label}  (input vector: class_idx={class_idx}, a={alpha})",
                     fontsize=14)
        fig.tight_layout()
        out_path = os.path.join(
            args.out,
            f"{class_name}_a{alpha:.2f}.png".replace(" ", "_"),
        )
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"\nDone. {len(DEMOS)} figures in {args.out}")


if __name__ == "__main__":
    main()
