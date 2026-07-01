"""
generate_diffusion_samples.py

Sample synthetic DAS patches from the trained latent diffusion model.

Two conditioning axes:
  * --class    one of 9 dataset classes (which event to generate)
  * --alpha    a float in [0, 1] (how much background noise to mix in)

Usage:
    python scripts/generate_diffusion_samples.py \
        --config configs/latent_diffusion_config.yaml \
        --vae_ckpt checkpoints/das_vae_v2/vae_best.pt \
        --diffusion_ckpt checkpoints/das_diffusion/diffusion_best.pt \
        --class running --alpha 0.3 --cfg_scale 7.5 --n 8 --out ./generated/

Outputs (one per sample i):
    {out}/sample_{i:03d}.npy   shape [1, 8, 16384] float32
    {out}/sample_{i:03d}.png   waterfall image

CFG math (2-axis): default --alpha_cfg_scale=1.0 gives the classic single-axis CFG
formula 'eps = eps_uc + cfg_scale * (eps_cc - eps_uc)' (only 2 forward passes per step).
Setting --alpha_cfg_scale != 1.0 enables 4-pass guided sampling that lets the user push
the alpha condition more or less aggressively.
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import CLASSES, NOISE_CLASS  # noqa: E402
from src.evaluation.plotting import plot_patch_panel, save_figure  # noqa: E402
from src.models.das_diffusion_unet import DASDiffusionUNet  # noqa: E402
from src.models.das_vae_v2 import DASVAEv2  # noqa: E402
from src.training.diffusion_trainer import unflatten_latent  # noqa: E402


def parse_class_alpha(class_name: str, alpha: float, classes=CLASSES) -> Tuple[int, float]:
    """Validate (class, alpha) and return (class_idx, alpha)."""
    if class_name not in classes:
        raise ValueError(f"Unknown class '{class_name}'. Valid: {list(classes)}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0, 1], got {alpha}")
    return classes.index(class_name), float(alpha)


def waterfall_png(patch: np.ndarray, path: str, title: str = ""):
    """Save a 2-panel (waterfall + spectrogram) figure for a single patch.

    patch: [1, 8, 16384]. Kept under the historical name so older callers still work.
    """
    fig = plot_patch_panel(patch, title=title)
    save_figure(fig, path)


def load_vae(cfg: dict, ckpt_path: str, device: str) -> Tuple[DASVAEv2, float]:
    vae_cfg = cfg["model"]["vae"]
    model = DASVAEv2(
        in_channels=1,
        encoder_channels=tuple(vae_cfg["encoder_channels"]),
        latent_channels=vae_cfg["latent_channels"],
        temporal_strides=tuple(vae_cfg["temporal_strides"]),
        kernel_time=vae_cfg["kernel_time"],
        kernel_space=vae_cfg["kernel_space"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaling = float(vae_cfg.get("scaling_factor", 1.0))
    return model, scaling


def load_diffusion(cfg: dict, ckpt_path: str, device: str, use_ema: bool) -> DASDiffusionUNet:
    data_cfg = cfg["data"]
    vae_cfg = cfg["model"]["vae"]
    diff_cfg = cfg["model"]["diffusion"]
    model = DASDiffusionUNet(
        latent_channels=vae_cfg["latent_channels"],
        spatial_h=data_cfg["patch_channels"],
        num_classes=diff_cfg["num_classes"],
        cond_dim=diff_cfg["cond_dim"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["ema_state"] if (use_ema and "ema_state" in ckpt) else ckpt["model_state"]
    state = {k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def sample(
    diffusion: DASDiffusionUNet,
    class_idx: int,
    alpha: float,
    cfg_scale: float,
    alpha_cfg_scale: float,
    n_samples: int,
    steps: int,
    latent_w: int,
    scheduler: DDIMScheduler,
    device: str,
) -> torch.Tensor:
    """2-axis classifier-free guidance sampler.

    For alpha_cfg_scale == 1.0, simplifies to the classic single-CFG-axis formula
    with only 2 forward passes per step. Otherwise runs 4 forward passes per step.

    Returns latent in flattened shape [n_samples, latent_ch * spatial_h, latent_w].
    """
    in_ch = diffusion.latent_channels * diffusion.spatial_h
    null_cls = diffusion.null_idx
    one_pass = abs(alpha_cfg_scale - 1.0) < 1e-12

    cls_real = torch.full((n_samples,), class_idx, dtype=torch.long, device=device)
    cls_null = torch.full((n_samples,), null_cls, dtype=torch.long, device=device)
    alpha_real = torch.full((n_samples,), float(alpha), dtype=torch.float32, device=device)
    alpha_null = torch.full((n_samples,), -1.0, dtype=torch.float32, device=device)  # sentinel

    x = torch.randn(n_samples, in_ch, latent_w, device=device)
    scheduler.set_timesteps(steps, device=device)

    for t in tqdm(scheduler.timesteps, desc="ddim", leave=False):
        t_batch = t.expand(n_samples).long().to(device)
        if one_pass:
            # eps_uc = unet(x, t, null_cls, alpha)
            # eps_cc = unet(x, t, class,    alpha)
            # eps    = eps_uc + cfg_scale * (eps_cc - eps_uc)
            eps_uc = diffusion(x, t_batch, cls_null, alpha_real)
            eps_cc = diffusion(x, t_batch, cls_real, alpha_real)
            pred = eps_uc + cfg_scale * (eps_cc - eps_uc)
        else:
            eps_uu = diffusion(x, t_batch, cls_null, alpha_null)
            eps_cu = diffusion(x, t_batch, cls_real, alpha_null)
            eps_uc = diffusion(x, t_batch, cls_null, alpha_real)
            eps_cc = diffusion(x, t_batch, cls_real, alpha_real)
            pred = (
                eps_uu
                + cfg_scale * (eps_cu - eps_uu)
                + alpha_cfg_scale * (eps_uc - eps_uu)
                + cfg_scale * alpha_cfg_scale
                * (eps_cc - eps_cu - eps_uc + eps_uu)
            )
        x = scheduler.step(pred, t, x).prev_sample
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--class", dest="class_name", required=True,
                        help=f"One of: {CLASSES}")
    parser.add_argument("--alpha", type=float, required=True,
                        help=f"Noise mix level in [0, 1]. 0=clean event, 1=event + full noise. "
                             f"For --class {NOISE_CLASS} this knob is partially redundant.")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale for the class condition.")
    parser.add_argument("--alpha_cfg_scale", type=float, default=1.0,
                        help="CFG scale for the alpha condition. 1.0 = no extra guidance (2 fwd passes/step). "
                             "!= 1.0 enables 2-axis CFG (4 fwd passes/step).")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_ema", action="store_true",
                        help="Sample from EMA weights (recommended) instead of last weights")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    vae_cfg = cfg["model"]["vae"]
    diff_cfg = cfg["model"]["diffusion"]
    classes = data_cfg["classes"]

    class_idx, alpha = parse_class_alpha(args.class_name, args.alpha, classes)
    print(f"class={args.class_name} (idx={class_idx})  alpha={alpha}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    vae, scaling = load_vae(cfg, args.vae_ckpt, device)
    diffusion = load_diffusion(cfg, args.diffusion_ckpt, device, use_ema=args.use_ema)

    latent_w = data_cfg["patch_time"]
    for s in vae_cfg["temporal_strides"]:
        latent_w //= s
    print(f"Latent W: {latent_w}")

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

    latents = unflatten_latent(x_flat, diffusion.latent_channels, diffusion.spatial_h)
    latents_unscaled = latents / scaling if scaling != 0 else latents

    decoded = []
    chunk = max(1, args.n // 4)
    for i in range(0, args.n, chunk):
        z = latents_unscaled[i:i + chunk]
        patch = vae.decode(z).float().cpu().numpy()
        decoded.append(patch)
    patches = np.concatenate(decoded, axis=0)

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
