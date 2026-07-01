"""
generate_pixel_samples.py

Sample DAS patches from the pixel-space diffusion model (no VAE).
Mirrors scripts/generate_diffusion_samples.py but skips the VAE decode step.

Usage:
    python scripts/generate_pixel_samples.py \
        --config configs/pixel_diffusion_config.yaml \
        --diffusion_ckpt checkpoints/das_pixel_diffusion/diffusion_best.pt \
        --class running --alpha 0.3 --n 8 --out ./generated_pixel/
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.generate_diffusion_samples import parse_class_alpha, sample, waterfall_png  # noqa: E402
from src.data.das_latent_patch_dataset import CLASSES  # noqa: E402
from src.models.das_diffusion_unet import DASDiffusionUNet  # noqa: E402
from src.training.diffusion_trainer import unflatten_latent  # noqa: E402


def load_pixel_diffusion(cfg: dict, ckpt_path: str, device: str, use_ema: bool) -> DASDiffusionUNet:
    data_cfg = cfg["data"]
    diff_cfg = cfg["model"]["pixel_diffusion"]
    model = DASDiffusionUNet(
        latent_channels=1,
        spatial_h=data_cfg["patch_channels"],
        base_channels=diff_cfg["base_channels"],
        channel_mults=tuple(diff_cfg["channel_mults"]),
        num_res_blocks=diff_cfg["num_res_blocks"],
        num_classes=diff_cfg["num_classes"],
        cond_dim=diff_cfg["cond_dim"],
        kernel=diff_cfg.get("kernel", 5),
        dilations_per_level=diff_cfg.get("dilations_per_level"),
        use_attention_per_level=diff_cfg.get("use_attention_per_level"),
        attention_in_mid=diff_cfg.get("attention_in_mid", False),
        attention_heads=diff_cfg.get("attention_heads", 4),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["ema_state"] if (use_ema and "ema_state" in ckpt) else ckpt["model_state"]
    state = {k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--class", dest="class_name", required=True,
                        help=f"One of: {CLASSES}")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--alpha_cfg_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    diff_cfg = cfg["model"]["pixel_diffusion"]
    classes = data_cfg["classes"]

    class_idx, alpha = parse_class_alpha(args.class_name, args.alpha, classes)
    print(f"class={args.class_name} (idx={class_idx})  alpha={alpha}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    diffusion = load_pixel_diffusion(cfg, args.diffusion_ckpt, device, use_ema=args.use_ema)

    # Pixel-space: W = patch_time directly (no VAE temporal compression).
    latent_w = data_cfg["patch_time"]
    print(f"Patch W: {latent_w}")

    scheduler = DDIMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        prediction_type=diff_cfg["prediction_type"],
        beta_schedule=diff_cfg["beta_schedule"],
        rescale_betas_zero_snr=True,
    )

    x_flat = sample(
        diffusion=diffusion,
        class_idx=class_idx,
        alpha=alpha,
        cfg_scale=args.cfg_scale,
        alpha_cfg_scale=args.alpha_cfg_scale,
        n_samples=args.n,
        steps=args.steps,
        latent_w=latent_w,
        scheduler=scheduler,
        device=device,
    )

    # Pixel-space: x_flat IS the (normalized) patch — just reshape, then inverse normalize.
    patches = unflatten_latent(x_flat, 1, diffusion.spatial_h).float().cpu().numpy()
    mu = float(data_cfg["normalize"]["mean"])
    sigma = float(data_cfg["normalize"]["std"])
    patches = patches * sigma + mu

    os.makedirs(args.out, exist_ok=True)
    title = f"{args.class_name}  alpha={alpha:.2f}  cfg={args.cfg_scale}"
    for i, p in enumerate(patches):
        np.save(os.path.join(args.out, f"sample_{i:03d}.npy"), p.astype(np.float32))
        waterfall_png(p, os.path.join(args.out, f"sample_{i:03d}.png"), title=title)

    print(f"Wrote {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
