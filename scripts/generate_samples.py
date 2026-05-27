"""
Generate synthetic DAS patches from a trained CVAE checkpoint.

Usage:
    python scripts/generate_samples.py \
        --checkpoint checkpoints/cvae_best.pth \
        --n-per-class 500 \
        --output-dir generated/

Output:
    generated/<class_name>.npy  — shape [N, 1, 32, 256] per class
"""

import argparse
import os
import torch
import numpy as np

from src.data.das_patch_dataset import CLASSES, N_CLASSES
from src.models.cvae import CVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--output-dir", default="generated/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    model = CVAE(
        latent_dim=model_cfg["latent_dim"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        enc_channels=tuple(model_cfg["encoder_channels"]),
        n_classes=model_cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # Generate per class
    classes = data_cfg.get("classes", CLASSES)
    n_cls = len(classes)
    for class_idx, class_name in enumerate(classes):
        c = torch.zeros(1, n_cls)
        c[0, class_idx] = 1.0

        with torch.no_grad():
            samples = model.generate(c, n=args.n_per_class, device=device)

        samples_np = samples.cpu().numpy()  # [N, 1, 32, 256]
        out_path = os.path.join(args.output_dir, f"{class_name}.npy")
        np.save(out_path, samples_np)
        print(f"  Saved {args.n_per_class} samples → {out_path}")

    print("Generation complete.")


if __name__ == "__main__":
    main()
