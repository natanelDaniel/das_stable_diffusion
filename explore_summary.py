import h5py
import numpy as np
import os
import json
from glob import glob

data_dir = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"

print("=" * 80)
print("DAS DATASET KEY METADATA SUMMARY")
print("=" * 80)

rows = []

for cls in sorted(os.listdir(data_dir)):
    cls_dir = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    h5_files = sorted(glob(os.path.join(cls_dir, "*.h5")))
    if not h5_files:
        continue

    h5_path = h5_files[0]
    npy_path = h5_path[:-3] + ".npy"

    with h5py.File(h5_path, "r") as f:
        raw = f["Acquisition"]["Raw[0]"]["RawData"]
        raw_shape = raw.shape
        raw_dtype = raw.dtype

        # Collect all attributes across the hierarchy
        acq_attrs = dict(f["Acquisition"].attrs) if "Acquisition" in f else {}
        raw0_attrs = dict(f["Acquisition"]["Raw[0]"].attrs) if "Raw[0]" in f["Acquisition"] else {}
        custom_attrs = dict(f["Acquisition"]["Custom"].attrs) if "Custom" in f["Acquisition"] else {}

        # Key metadata
        pulse_rate = acq_attrs.get("PulseRate", "N/A")   # Hz
        max_freq = acq_attrs.get("MaximumFrequency", "N/A")
        n_loci = acq_attrs.get("NumberOfLoci", "N/A")
        spatial_interval = acq_attrs.get("SpatialSamplingInterval", "N/A")
        spatial_unit = acq_attrs.get("SpatialSamplingIntervalUnit", b"?").decode() if isinstance(acq_attrs.get("SpatialSamplingIntervalUnit"), bytes) else "?"
        data_width = custom_attrs.get("Data Width (Bits)", "N/A")
        decimation = custom_attrs.get("Decimation Factor", "N/A")

        # RawDataTime if it exists
        rawtime_shape = "N/A"
        rawtime_sample = "N/A"
        if "RawDataTime" in f["Acquisition"]["Raw[0]"]:
            rt = f["Acquisition"]["Raw[0]"]["RawDataTime"]
            rawtime_shape = rt.shape
            rawtime_sample = str(rt[:3])

    bmp_shape = "N/A"
    n_events = 0
    if os.path.exists(npy_path):
        bmp = np.load(npy_path)
        bmp_shape = bmp.shape
        n_events = int(bmp.sum())

    print(f"\n{'='*60}")
    print(f"CLASS: {cls}  ({len(h5_files)} files)")
    print(f"  File:              {os.path.basename(h5_path)}")
    print(f"  RawData shape:     {raw_shape}  (n_time x n_channels)")
    print(f"  dtype:             {raw_dtype}")
    print(f"  PulseRate (Hz):    {pulse_rate}")
    print(f"  MaxFrequency:      {max_freq} Hz")
    print(f"  NumberOfLoci:      {n_loci}   (fiber channels)")
    print(f"  SpatialInterval:   {spatial_interval} {spatial_unit}")
    print(f"  Data Width:        {data_width} bits")
    print(f"  Decimation Factor: {decimation}")
    print(f"  Bitmap shape:      {bmp_shape}  nonzero={n_events}")
    print(f"  RawDataTime:       {rawtime_shape}  sample={rawtime_sample}")

    rows.append({
        "cls": cls,
        "files": len(h5_files),
        "raw_shape": raw_shape,
        "dtype": str(raw_dtype),
        "pulse_hz": pulse_rate,
        "max_freq": max_freq,
        "n_loci": n_loci,
        "dx_m": f"{spatial_interval:.3f}" if isinstance(spatial_interval, float) else spatial_interval,
        "bits": data_width,
        "bitmap": bmp_shape,
        "events": n_events,
    })

print("\n\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"{'Class':<14} {'Files':>5} {'RawData (n_time x n_ch)':>26} {'dtype':>7} {'PulseHz':>9} {'MaxFreq':>8} {'dx(m)':>7} {'Bitmap':>18} {'Events':>8}")
print("-" * 100)
for r in rows:
    shape_str = f"{r['raw_shape']}"
    bmp_str = f"{r['bitmap']}" if r['bitmap'] != 'N/A' else "N/A"
    print(f"{r['cls']:<14} {r['files']:>5} {shape_str:>26} {r['dtype']:>7} {str(r['pulse_hz']):>9} {str(r['max_freq']):>8} {str(r['dx_m']):>7} {bmp_str:>18} {r['events']:>8}")

# Important derived insight
print("\n\n" + "=" * 80)
print("KEY INSIGHTS FOR CVAE DESIGN")
print("=" * 80)
for r in rows:
    if isinstance(r['pulse_hz'], float) and isinstance(r['raw_shape'], tuple):
        n_time, n_ch = r['raw_shape']
        duration_s = n_time / r['pulse_hz']
        print(f"  {r['cls']:<14}: {n_time} time samples / {r['pulse_hz']:.0f} Hz = {duration_s:.2f} s recording")
