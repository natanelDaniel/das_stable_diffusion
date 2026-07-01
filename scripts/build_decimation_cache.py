"""
build_decimation_cache.py

One-time preprocessing: decimate each *.h5 raw DAS recording from 20 kHz to 1 kHz
and cache as *.dec1k.npy next to the h5 file.

Reading a 16,384-sample patch at 1 kHz requires 327,680 raw samples per patch
(~10 MB from h5py per patch). Pre-decimating once gives >100x speedup vs
decimating per __getitem__.

Output: *.dec1k.npy with dtype float32, shape [n_time_dec, n_fiber_ch], where
n_time_dec = ceil(n_time_raw / 20).

Idempotent: existing cache files are skipped unless --force is given.

Usage:
    python scripts/build_decimation_cache.py \
        --data_dir "C:/Users/netanel.daniel/DAS-dataset/DAS-dataset/data"
"""

import argparse
import os
import sys
from glob import glob

import h5py
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.das_latent_patch_dataset import ORIG_SR_HZ, cache_suffix_for  # noqa: E402


def decimate_recording(h5_path: str, decim_factor: int) -> np.ndarray:
    """Read raw [n_time, n_fiber] @ 20 kHz, return decimated [~n_time/factor, n_fiber]."""
    with h5py.File(h5_path, "r") as f:
        raw = f["Acquisition"]["Raw[0]"]["RawData"][:]
    raw = np.asarray(raw, dtype=np.float32)
    dec = resample_poly(raw, up=1, down=decim_factor, axis=0, padtype="line")
    return dec.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True,
                        help="Root containing class subdirs with *.h5")
    parser.add_argument("--target_sr", type=int, default=1000,
                        help="Target sample rate in Hz. Must divide 20000.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing caches")
    args = parser.parse_args()

    if ORIG_SR_HZ % args.target_sr != 0:
        raise SystemExit(f"--target_sr {args.target_sr} must divide ORIG_SR_HZ ({ORIG_SR_HZ}).")
    decim_factor = ORIG_SR_HZ // args.target_sr
    cache_suffix = cache_suffix_for(args.target_sr)

    h5_paths = sorted(glob(os.path.join(args.data_dir, "*", "*.h5")))
    if not h5_paths:
        raise SystemExit(f"No *.h5 files found under {args.data_dir}/<class>/")

    print(f"Found {len(h5_paths)} recordings under {args.data_dir}")
    print(f"Decimating {ORIG_SR_HZ} Hz -> {args.target_sr} Hz (factor {decim_factor})")
    print(f"Output suffix: {cache_suffix}")
    for h5_path in tqdm(h5_paths, desc="decimating"):
        out_path = h5_path[:-3] + cache_suffix
        if os.path.exists(out_path) and not args.force:
            tqdm.write(f"skip (exists): {os.path.basename(out_path)}")
            continue
        try:
            dec = decimate_recording(h5_path, decim_factor)
        except Exception as e:
            tqdm.write(f"FAIL {h5_path}: {e}")
            continue
        np.save(out_path, dec)
        tqdm.write(f"wrote {os.path.basename(out_path)} shape={dec.shape} dtype={dec.dtype}")


if __name__ == "__main__":
    main()
