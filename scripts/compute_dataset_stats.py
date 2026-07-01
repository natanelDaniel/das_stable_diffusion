"""
compute_dataset_stats.py

One-time: sample patches from the un-normalized dataset, compute global (mean, std),
and update configs/latent_diffusion_config.yaml in place.

Without proper global stats the VAE sees raw DAS values (10^3-10^5) and diverges.
This script writes the computed (mean, std) into data.normalize so subsequent
training runs feed in roughly-unit-variance patches.

Usage:
    python scripts/compute_dataset_stats.py --config configs/latent_diffusion_config.yaml
    python scripts/compute_dataset_stats.py --config ... --n_samples 1024
"""

import argparse
import os
import sys

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import (  # noqa: E402
    DASLatentPatchDataset,
    compute_global_stats,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/latent_diffusion_config.yaml")
    parser.add_argument("--n_samples", type=int, default=512,
                        help="Number of patches to sample for stats estimation.")
    parser.add_argument("--no_write", action="store_true",
                        help="Print the computed stats without updating the config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    ds = DASLatentPatchDataset(
        data_dir=data_cfg["data_dir"],
        patch_channels=data_cfg["patch_channels"],
        patch_time=data_cfg["patch_time"],
        event_offset_range=tuple(data_cfg["event_offset_range"]),
        decimation=data_cfg["decimation"],
        classes=data_cfg["classes"],
        normalize=None,
        seed=data_cfg["seed"],
        cache_in_ram=data_cfg.get("cache_in_ram", False),
        target_sample_rate=data_cfg.get("target_sample_rate", 1000),
    )
    print(f"Dataset: {len(ds)} samples. Sampling {args.n_samples} for stats...")

    mean, std = compute_global_stats(ds, n_samples=args.n_samples, seed=data_cfg["seed"])
    print(f"\nGlobal mean: {mean:.6g}")
    print(f"Global std:  {std:.6g}")

    if args.no_write:
        return

    cfg["data"]["normalize"] = {"mean": float(mean), "std": float(std)}
    with open(args.config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\nUpdated {args.config}  data.normalize -> mean={mean:.6g}, std={std:.6g}")


if __name__ == "__main__":
    main()
