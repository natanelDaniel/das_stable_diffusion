"""
DAS Dataset Review — generates a multi-panel figure saved to
  outputs/dataset_review.png

Panels:
  1. Class distribution bar chart (raw event counts)
  2. Class distribution after 0.2 % decimation
  3. One sample patch per class — 2D heatmap [channels × time]
  4. Mean amplitude trace per class (channel-averaged waveform)
  5. Mean power spectrum per class (FFT, channel-averaged)
  6. HDF5 recording durations per class
"""

import os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.das_patch_dataset import DASPatchDataset, CLASSES, N_CLASSES

DATA_DIR   = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 20_000   # Hz

# ── colour palette (one per class) ──────────────────────────────────────────
PALETTE = plt.cm.tab10.colors

# ============================================================================
# 1. Collect raw event counts from bitmaps
# ============================================================================
print("Reading bitmap event counts …")
raw_counts = {}
durations   = {}
for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    # event counts
    npys = sorted(f for f in os.listdir(cls_dir) if f.endswith(".npy"))
    total = sum(int(np.load(os.path.join(cls_dir, n)).sum()) for n in npys)
    raw_counts[cls] = total
    # recording durations (seconds)
    h5s  = sorted(f for f in os.listdir(cls_dir) if f.endswith(".h5"))
    dur  = 0.0
    for h5 in h5s:
        with h5py.File(os.path.join(cls_dir, h5), "r") as f:
            n_time = f["Acquisition"]["Raw[0]"]["RawData"].shape[0]
        dur += n_time / SR
    durations[cls] = dur

# ============================================================================
# 2. Build DASPatchDataset (very small decimation for speed)
# ============================================================================
print("Indexing dataset …")
ds = DASPatchDataset(
    data_dir=DATA_DIR,
    patch_channels=32,
    patch_time=256,
    # bitmap_shift uses default (2048) — matches DASDataLoader bitmap creation
    decimation={c: 0.002 for c in CLASSES},
)
print(f"  -> {len(ds)} samples after decimation")

dec_counts = collections.Counter(CLASSES[s[3]] for s in ds.samples)

# ============================================================================
# 3. Gather one patch + waveform + spectrum per class
# ============================================================================
print("Loading representative patches …")
class_patches  = {}   # cls → np.array [32, 256]
class_waveform = {}   # cls → np.array [256]  (mean over channels)
class_spectrum = {}   # cls → np.array [freq_bins]

for cls in CLASSES:
    # find first sample of this class
    for h5_path, win_idx, ch_idx, cls_idx in ds.samples:
        if CLASSES[cls_idx] == cls:
            patch_tensor, _, _ = ds[ds.samples.index((h5_path, win_idx, ch_idx, cls_idx))]
            patch = patch_tensor.squeeze(0).numpy()  # [32, 256]
            class_patches[cls]  = patch
            class_waveform[cls] = patch.mean(axis=0)  # mean over channels
            # Power spectrum (single-sided)
            N = patch.shape[1]
            fft_vals = np.fft.rfft(class_waveform[cls])
            psd = (np.abs(fft_vals) ** 2) / N
            freqs = np.fft.rfftfreq(N, d=1.0 / SR)
            class_spectrum[cls] = (freqs, psd)
            break

# ============================================================================
# 4. Build figure
# ============================================================================
print("Generating plots …")
fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor("#0e1117")

gs_main = gridspec.GridSpec(
    4, 1, figure=fig,
    height_ratios=[1, 1, 2.2, 1.6],
    hspace=0.45, left=0.06, right=0.97, top=0.95, bottom=0.04,
)

TITLE_COLOR  = "#e0e0e0"
TICK_COLOR   = "#aaaaaa"
GRID_COLOR   = "#2a2a2a"
BG_COLOR     = "#161b22"

def style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TICK_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)
    ax.title.set_color(TITLE_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


# ── Panel 1: Raw event counts ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs_main[0])
style_ax(ax1)
vals = [raw_counts[c] for c in CLASSES]
bars = ax1.bar(CLASSES, vals, color=PALETTE[:N_CLASSES], edgecolor="#222", linewidth=0.5)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
             f"{v:,}", ha="center", va="bottom", fontsize=7.5, color=TITLE_COLOR)
ax1.set_title("Raw event counts (bitmap sum)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Events", fontsize=9)
ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", rotation=30)

# ── Panel 2: Decimated sample counts ────────────────────────────────────────
ax2 = fig.add_subplot(gs_main[1])
style_ax(ax2)
vals2 = [dec_counts[c] for c in CLASSES]
bars2 = ax2.bar(CLASSES, vals2, color=PALETTE[:N_CLASSES], edgecolor="#222", linewidth=0.5)
for bar, v in zip(bars2, vals2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals2)*0.01,
             str(v), ha="center", va="bottom", fontsize=7.5, color=TITLE_COLOR)
ax2.set_title("Dataset samples after 0.2 % decimation  (patch_time=256, shift=128)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Samples", fontsize=9)
ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", rotation=30)

# ── Panel 3: Sample patches — one per class ──────────────────────────────────
gs_patches = gridspec.GridSpecFromSubplotSpec(
    2, 5, subplot_spec=gs_main[2], hspace=0.55, wspace=0.3,
)
t_axis  = np.arange(256) / SR * 1000   # ms
ch_axis = np.arange(32)

for i, cls in enumerate(CLASSES):
    row, col = divmod(i, 5)
    ax = fig.add_subplot(gs_patches[row, col])
    style_ax(ax)
    patch = class_patches.get(cls)
    if patch is not None:
        vmax = np.percentile(np.abs(patch), 98)
        im = ax.imshow(
            patch,
            aspect="auto",
            origin="lower",
            extent=[t_axis[0], t_axis[-1], ch_axis[0], ch_axis[-1]],
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="z-score").ax.tick_params(labelsize=6, colors=TICK_COLOR)
    ax.set_title(cls, fontsize=9, fontweight="bold", color=PALETTE[i])
    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel("Channel", fontsize=7)

# ── Panel 4: Mean waveform + spectrum ────────────────────────────────────────
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_main[3], hspace=0.4, wspace=0.32,
)

# Waveform
ax_wave = fig.add_subplot(gs_bottom[0])
style_ax(ax_wave)
for i, cls in enumerate(CLASSES):
    wf = class_waveform.get(cls)
    if wf is not None:
        ax_wave.plot(t_axis, wf, color=PALETTE[i], linewidth=0.9, alpha=0.85, label=cls)
ax_wave.set_title("Mean channel amplitude per class  (z-scored patch)", fontsize=10, fontweight="bold")
ax_wave.set_xlabel("Time (ms)", fontsize=9)
ax_wave.set_ylabel("Amplitude (z-score)", fontsize=9)
ax_wave.legend(fontsize=7, ncol=3, facecolor="#1a1a2e", labelcolor=TITLE_COLOR,
               edgecolor="#333", framealpha=0.9)
ax_wave.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
ax_wave.set_axisbelow(True)

# Spectrum
ax_spec = fig.add_subplot(gs_bottom[1])
style_ax(ax_spec)
for i, cls in enumerate(CLASSES):
    spec = class_spectrum.get(cls)
    if spec is not None:
        freqs, psd = spec
        mask = freqs <= 2000   # show up to 2 kHz
        ax_spec.semilogy(freqs[mask] / 1000, psd[mask] + 1e-12,
                         color=PALETTE[i], linewidth=0.9, alpha=0.85, label=cls)
ax_spec.set_title("Power spectrum — mean channel  (up to 2 kHz)", fontsize=10, fontweight="bold")
ax_spec.set_xlabel("Frequency (kHz)", fontsize=9)
ax_spec.set_ylabel("Power (log)", fontsize=9)
ax_spec.legend(fontsize=7, ncol=3, facecolor="#1a1a2e", labelcolor=TITLE_COLOR,
               edgecolor="#333", framealpha=0.9)
ax_spec.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
ax_spec.set_axisbelow(True)

# ── Main title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "DAS Dataset Review  |  SR=20 kHz  |  1700 channels (1663 for construction)  |  "
    "patch 32ch × 256t (12.8 ms)  |  spatial res ~1.02 m/ch",
    fontsize=11, color=TITLE_COLOR, fontweight="bold", y=0.975,
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "dataset_review.png")
fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved -> {out_path}")
