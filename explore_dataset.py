import h5py
import numpy as np
import os
import json
from glob import glob

data_dir = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"

print("=" * 70)
print("DAS DATASET EXPLORATION")
print("=" * 70)

summary = []

for cls in sorted(os.listdir(data_dir)):
    cls_dir = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    h5_files = sorted(glob(os.path.join(cls_dir, "*.h5")))
    if not h5_files:
        continue

    h5_path = h5_files[0]
    npy_path = h5_path[:-3] + ".npy"
    json_path = h5_path[:-3] + ".json"

    print(f"\n{'='*60}")
    print(f"CLASS: {cls}")
    print(f"File: {os.path.basename(h5_path)}")

    with h5py.File(h5_path, "r") as f:
        # Walk the full tree
        def visit_item(name, obj):
            indent = "  " * (name.count("/") + 1)
            attrs = {k: v for k, v in obj.attrs.items()}
            if hasattr(obj, "shape"):
                print(f"{indent}{name}: shape={obj.shape} dtype={obj.dtype} attrs={attrs}")
            else:
                print(f"{indent}{name}/  attrs={attrs}")

        print("HDF5 Tree:")
        f.visititems(visit_item)
        print(f"  Root attrs: {dict(f.attrs)}")

        # Get RawData
        raw = f["Acquisition"]["Raw[0]"]["RawData"]
        shape = raw.shape
        dtype = raw.dtype
        print(f"\nRawData shape: {shape}  dtype: {dtype}")

        # Sample values
        slice_t = min(100, shape[0])
        slice_c = min(10, shape[1])
        sample = np.array(raw[:slice_t, :slice_c], dtype=np.float32)
        print(f"Value range (first {slice_t}t x {slice_c}ch): min={sample.min():.4f}  max={sample.max():.4f}  mean={sample.mean():.4f}")

        # All attrs across hierarchy
        rate = None
        for path in ["", "Acquisition", "Acquisition/Raw[0]", "Acquisition/Raw[0]/RawData"]:
            try:
                obj = f[path] if path else f
                for k, v in obj.attrs.items():
                    print(f"  Attr [{path}] {k} = {v}")
                    if any(x in k.lower() for x in ["rate", "freq", "sampl", "hz", "fs", "dt"]):
                        rate = (k, v)
            except Exception:
                pass

        # Check for other datasets that might hold metadata
        for key in ["Acquisition/Raw[0]/RawDataTime", "Acquisition/Raw[0]/RawDataUnit"]:
            try:
                ds = f[key]
                print(f"  Dataset {key}: shape={ds.shape} dtype={ds.dtype} val={ds[0] if ds.shape else ds[()]}")
            except Exception:
                pass

    # Bitmap
    if os.path.exists(npy_path):
        bmp = np.load(npy_path)
        print(f"\nBitmap shape: {bmp.shape}  nonzero events: {int(bmp.sum())}")
    else:
        print("No bitmap found")
        bmp = None

    # JSON
    if os.path.exists(json_path):
        with open(json_path) as jf:
            meta = json.load(jf)
        print(f"JSON keys: {list(meta.keys())}")
        if "shape" in meta:
            print(f"  JSON shape: {meta['shape']}")
        for k, v in meta.items():
            if k != "curve0":
                print(f"  JSON {k}: {v}")
    else:
        print("No JSON found")

    summary.append({
        "class": cls,
        "file": os.path.basename(h5_path),
        "raw_shape": shape,
        "dtype": str(dtype),
        "n_files": len(h5_files),
        "bitmap_shape": tuple(bmp.shape) if bmp is not None else None,
        "n_events": int(bmp.sum()) if bmp is not None else 0,
    })

print("\n\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Class':<15} {'Files':>5} {'RawData shape':>20} {'dtype':>8} {'Bitmap':>15} {'Events':>8}")
print("-" * 75)
for s in summary:
    raw_str = f"{s['raw_shape']}"
    bmp_str = f"{s['bitmap_shape']}" if s["bitmap_shape"] else "N/A"
    print(f"{s['class']:<15} {s['n_files']:>5} {raw_str:>20} {s['dtype']:>8} {bmp_str:>15} {s['n_events']:>8}")
