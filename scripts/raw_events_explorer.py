"""
Raw Events Explorer
===================
Loads a single DAS recording (car class) and produces a 5-panel figure that
explains exactly how raw events are stored and what they look like.

File format recap
-----------------
  *.h5   : RawData  shape=[n_time, n_fiber_ch]  int16  @ 20 000 Hz
  *.npy  : bitmap   shape=[n_windows, n_fiber_ch]  bool
             bitmap[w, ch] = 1 means an event was annotated at
             channel ch in window w  (window = 2048-sample / 102.4ms stride)
  *.json : 5 annotation curves (curve0..curve4)
             each curve is {x: [channel_idx, ...], y: [window_idx, ...]}
             tracing an event track through (channel, window) space

Output:  outputs/raw_events_explorer.png
"""

import os, sys, json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.das_patch_dataset import DASPatchDataset, CLASSES

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECORDING = "auto2_2023-04-17T124510+0100"
CLASS_DIR  = os.path.join(DATA_DIR, "car")
H5_PATH    = os.path.join(CLASS_DIR, RECORDING + ".h5")
NPY_PATH   = os.path.join(CLASS_DIR, RECORDING + ".npy")
JSON_PATH  = os.path.join(CLASS_DIR, RECORDING + ".json")

SR         = 20_000   # Hz
SHIFT      = 2048     # samples per bitmap window
FSIZE      = 8192     # DASDataLoader window size (samples) for waveform extraction

# ── Style ────────────────────────────────────────────────────────────────────
BG      = "#0e1117"
AX_BG   = "#161b22"
TC      = "#e0e0e0"
GC      = "#2a2a2a"
PALETTE = plt.cm.tab10.colors

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.xaxis.label.set_color("#aaaaaa"); ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color(TC)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    if title:  ax.set_title(title,  fontsize=9,  fontweight="bold", color=TC)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)

# ============================================================================
# Load data
# ============================================================================
print("Loading car recording ...")
with h5py.File(H5_PATH, "r") as f:
    raw_full = f["Acquisition"]["Raw[0]"]["RawData"][:, :]   # [n_time, 1700]
raw_full = raw_full.astype(np.float32)
n_time, n_ch = raw_full.shape
print(f"  HDF5:   shape={raw_full.shape}  duration={n_time/SR:.1f}s")

bitmap = np.load(NPY_PATH)
n_win  = bitmap.shape[0]
events = np.argwhere(bitmap)           # [[w, ch], ...]
density = bitmap.sum() / bitmap.size * 100
print(f"  Bitmap: shape={bitmap.shape}  events={len(events)}  density={density:.2f}%")

with open(JSON_PATH) as f:
    jdata = json.load(f)
curves = {k: jdata[k] for k in jdata if k.startswith("curve")}
print(f"  JSON:   {len(curves)} curves loaded")

# ── Waterfall slice: first 60 s, every 4th sample ───────────────────────────
MAX_S    = 60
MAX_SAMP = min(n_time, MAX_S * SR)
DS       = 4                           # time downsampling
wf_data  = raw_full[:MAX_SAMP:DS, :]  # [n_disp, 1700]
t_axis   = np.arange(wf_data.shape[0]) * DS / SR   # seconds
vmax     = 2 * wf_data.std()

# ── First event cell for waveform panels ────────────────────────────────────
# find an event in the first 60 s
early_events = [(w, ch) for w, ch in events if w * SHIFT < MAX_SAMP]
w0, ch0 = early_events[len(early_events) // 2]   # mid event, not first (richer signal)
t0_samp  = w0 * SHIFT                             # start sample of bitmap window
# raw 1-second context window
ctx_start = max(0, t0_samp - SR)
ctx_end   = min(n_time, t0_samp + 2 * SR)
t_ctx     = (np.arange(ctx_end - ctx_start) + ctx_start) / SR

# 1D waveforms
wf_event   = raw_full[ctx_start:ctx_end, ch0]
# non-event channel: find a column with all zeros in bitmap
zero_cols   = np.where(bitmap.sum(axis=0) == 0)[0]
ch_noise    = zero_cols[len(zero_cols) // 2]
wf_noise    = raw_full[ctx_start:ctx_end, ch_noise]

# ── DASDataLoader-style extracted window (FSIZE samples) ────────────────────
win_start = max(0, t0_samp)
win_end   = min(n_time, win_start + FSIZE)
win_event = raw_full[win_start:win_end, ch0]
win_noise = raw_full[win_start:win_end, ch_noise]

# ── FFT (same function as DASDataLoader) ────────────────────────────────────
def fft_log(x1d):
    """Single window FFT → log-magnitude spectrum."""
    spec = np.fft.rfft(x1d)[1:]          # remove DC
    spec = np.abs(spec) + 1
    spec = np.log10(spec)
    freqs = np.fft.rfftfreq(len(x1d), 1 / SR)[1:]
    return freqs, spec

freqs_e, spec_e = fft_log(win_event)
freqs_n, spec_n = fft_log(win_noise)

# ── DASPatchDataset patches for 3 classes ───────────────────────────────────
print("Loading example patches ...")
SHOW_CLASSES = ["car", "walk", "regular"]
patches = {}
for cls in SHOW_CLASSES:
    ds = DASPatchDataset(
        data_dir=DATA_DIR,
        patch_channels=32,
        patch_time=256,
        decimation={cls: 0.01},
        classes=[cls],
    )
    p, _, _ = ds[0]
    patches[cls] = p.squeeze(0).numpy()   # [32, 256]

# ============================================================================
# Build figure
# ============================================================================
print("Generating plots ...")

fig = plt.figure(figsize=(20, 28))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    5, 1, figure=fig,
    height_ratios=[2.2, 1.4, 1.4, 1.4, 1.4],
    hspace=0.52, left=0.07, right=0.97, top=0.95, bottom=0.04,
)

# ── Panel 1: DAS Waterfall with event overlay ────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style(ax1,
      title=f"Panel 1 — DAS Waterfall  '{RECORDING}'  (first {MAX_S} s, 4x downsampled in time)",
      xlabel="Time (s)", ylabel="Fiber channel")
im = ax1.imshow(
    wf_data.T,
    aspect="auto", origin="lower",
    extent=[t_axis[0], t_axis[-1], 0, n_ch],
    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    interpolation="nearest",
)
cb = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01)
cb.set_label("Amplitude (int16)", color="#aaaaaa", fontsize=7)
cb.ax.tick_params(labelsize=6, colors="#aaaaaa")

# Overlay bitmap events as small white dots
ev_plot = [(w, ch) for w, ch in events if w * SHIFT / SR < MAX_S]
if ev_plot:
    ew = np.array([w * SHIFT / SR for w, _ in ev_plot])
    ech = np.array([ch for _, ch in ev_plot])
    ax1.scatter(ew, ech, s=1.5, c="white", alpha=0.4, linewidths=0, label="Bitmap events")

# Overlay JSON curves (x=channel, y=window → convert y to seconds)
CURVE_COLORS = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff922b"]
for i, (k, cv) in enumerate(curves.items()):
    cx = np.array(cv["x"])            # channel index
    cy = np.array(cv["y"]) * SHIFT / SR  # window → seconds
    mask = cy < MAX_S
    if mask.sum() > 1:
        ax1.plot(cy[mask], cx[mask], color=CURVE_COLORS[i], lw=1.2,
                 alpha=0.85, label=f"JSON {k}")

ax1.legend(fontsize=7, ncol=3, facecolor="#1a1a2e", labelcolor=TC,
           edgecolor="#333", markerscale=5, loc="upper right")

# ── Panel 2: Bitmap heatmap ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
style(ax2,
      title=f"Panel 2 — Event Bitmap  shape={bitmap.shape}  (bool, {density:.2f}% filled)  "
            f"[n_windows={n_win}, n_channels={n_ch}]",
      xlabel="Bitmap window index  (1 window = 2048 samples = 102.4 ms)",
      ylabel="Fiber channel")
im2 = ax2.imshow(
    bitmap.T.astype(np.uint8),
    aspect="auto", origin="lower",
    extent=[0, n_win, 0, n_ch],
    cmap="hot", interpolation="nearest",
)
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01)
cb2.set_label("0=background  1=event", color="#aaaaaa", fontsize=7)
cb2.ax.tick_params(labelsize=6, colors="#aaaaaa")
for i, (k, cv) in enumerate(curves.items()):
    cx = np.array(cv["x"])   # channel
    cy = np.array(cv["y"])   # window index
    ax2.plot(cy, cx, color=CURVE_COLORS[i], lw=1.0, alpha=0.85)

# ── Panel 3: Raw 1D waveforms ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style(ax3,
      title=f"Panel 3 — Raw 1D Waveforms  (ch {ch0}=event  ch {ch_noise}=no-event)",
      xlabel="Time (s)", ylabel="Amplitude (int16)")
ax3.plot(t_ctx, wf_event, color=PALETTE[0], lw=0.8, alpha=0.9, label=f"ch {ch0} (event)")
ax3.plot(t_ctx, wf_noise, color=PALETTE[7], lw=0.8, alpha=0.7, label=f"ch {ch_noise} (no event)")
# shade the DASDataLoader extraction window
t_win_s = win_start / SR
t_win_e = win_end   / SR
ax3.axvspan(t_win_s, t_win_e, color=PALETTE[3], alpha=0.15,
            label=f"Extracted window ({FSIZE} samples = {FSIZE/SR*1000:.0f} ms)")
ax3.axvline(t0_samp / SR, color="white", lw=0.8, ls="--", alpha=0.6,
            label=f"Bitmap window w={w0} start")
ax3.yaxis.grid(True, color=GC, lw=0.5); ax3.set_axisbelow(True)
ax3.legend(fontsize=7, ncol=2, facecolor="#1a1a2e", labelcolor=TC, edgecolor="#333")

# ── Panel 4: FFT comparison ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
style(ax4,
      title=f"Panel 4 — FFT Spectrum  ({FSIZE} samples, log-magnitude)  "
            f"event ch {ch0} vs no-event ch {ch_noise}",
      xlabel="Frequency (kHz)", ylabel="log10(|FFT|)")
mask2k = freqs_e <= 2000
ax4.plot(freqs_e[mask2k] / 1000, spec_e[mask2k], color=PALETTE[0], lw=0.9, label=f"Event ch {ch0}")
ax4.plot(freqs_n[mask2k] / 1000, spec_n[mask2k], color=PALETTE[7], lw=0.9, alpha=0.8,
         label=f"No-event ch {ch_noise}")
ax4.yaxis.grid(True, color=GC, lw=0.5); ax4.set_axisbelow(True)
ax4.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=TC, edgecolor="#333")

# ── Panel 5: Example extracted patches ──────────────────────────────────────
gs5 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[4], wspace=0.35)
t_patch = np.arange(256) / SR * 1000   # ms

for j, cls in enumerate(SHOW_CLASSES):
    ax = fig.add_subplot(gs5[j])
    style(ax,
          title=f"Panel 5 — class: {cls}",
          xlabel="Time (ms)", ylabel="Channel (patch-relative)")
    patch = patches[cls]
    vp = np.percentile(np.abs(patch), 98)
    im5 = ax.imshow(
        patch,
        aspect="auto", origin="lower",
        extent=[t_patch[0], t_patch[-1], 0, 32],
        cmap="RdBu_r", vmin=-vp, vmax=vp,
        interpolation="nearest",
    )
    plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04,
                 label="z-score").ax.tick_params(labelsize=6, colors="#aaaaaa")

# ── Super-title & annotation box ─────────────────────────────────────────────
fig.suptitle(
    "DAS Raw Event Explorer  |  car / auto2_2023-04-17  |  SR=20 kHz  |  "
    "bitmap: 2048-sample windows (102.4 ms)  |  DASDataLoader fsize=8192 (409.6 ms)",
    fontsize=10, color=TC, fontweight="bold", y=0.975,
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "raw_events_explorer.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved -> {out_path}")
