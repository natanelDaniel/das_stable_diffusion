"""
plotting.py — shared visual helpers for DAS patches.

Spectrogram recipe matches scripts/das_viewer.py:
    nperseg=1024, noverlap=1008 (hop=16), Hann window, magnitude in dB.
    Input is assumed to already be at TARGET_SR_HZ (1000 Hz) — no extra decimation.

Public API:
    compute_spectrogram_db(sig, fs=1000, freq_max=None) -> (f, t, Sxx_db)
    plot_patch_panel(patch, title="", spec_channel=None, freq_max=200) -> matplotlib.figure.Figure
    plot_input_vs_recon(x, x_hat, title="", spec_channel=None, freq_max=200) -> Figure
    save_figure(fig, path)   — close-safe save helper
"""

from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # safe for headless training nodes
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram


TARGET_SR_HZ = 1000
SPEC_NPERSEG = 1024
SPEC_NOVERLAP = 1008
SPEC_DEFAULT_FREQ_MAX_HZ = 200


def _as_2d(patch: np.ndarray) -> np.ndarray:
    """Accept [1, C, T] or [C, T] -> [C, T]."""
    if patch.ndim == 3:
        if patch.shape[0] != 1:
            raise ValueError(f"Expected leading dim 1, got {patch.shape}")
        return patch[0]
    if patch.ndim == 2:
        return patch
    raise ValueError(f"Unsupported patch ndim {patch.ndim}; expected 2 or 3.")


def compute_spectrogram_db(
    sig: np.ndarray,
    fs: int = TARGET_SR_HZ,
    freq_max: Optional[float] = SPEC_DEFAULT_FREQ_MAX_HZ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT magnitude in dB. sig: 1D float array."""
    if sig.ndim != 1:
        raise ValueError(f"sig must be 1D, got shape {sig.shape}")
    f_s, t_s, Sxx = scipy_spectrogram(
        sig.astype(np.float64),
        fs=fs,
        nperseg=SPEC_NPERSEG,
        noverlap=SPEC_NOVERLAP,
        window="hann",
    )
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    if freq_max is not None:
        mask = f_s <= freq_max
        f_s = f_s[mask]
        Sxx_db = Sxx_db[mask, :]
    return f_s, t_s, Sxx_db


def _draw_waterfall(ax, patch_2d: np.ndarray, title: str = "", fs: int = TARGET_SR_HZ):
    C, T = patch_2d.shape
    vmax = float(np.percentile(np.abs(patch_2d), 99))
    if vmax <= 0:
        vmax = 1.0
    duration_s = T / fs
    ax.imshow(
        patch_2d,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=(0.0, duration_s, -0.5, C - 0.5),
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fiber channel")
    if title:
        ax.set_title(title)


def _draw_spectrogram(ax, sig_1d: np.ndarray, fs: int = TARGET_SR_HZ,
                      title: str = "", freq_max: Optional[float] = SPEC_DEFAULT_FREQ_MAX_HZ):
    f_s, t_s, Sxx_db = compute_spectrogram_db(sig_1d, fs=fs, freq_max=freq_max)
    vmin, vmax = float(np.percentile(Sxx_db, 5)), float(np.percentile(Sxx_db, 99))
    ax.imshow(
        Sxx_db,
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=(t_s[0], t_s[-1], f_s[0], f_s[-1]),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    if title:
        ax.set_title(title)


def plot_patch_panel(
    patch: np.ndarray,
    title: str = "",
    spec_channel: Optional[int] = None,
    freq_max: Optional[float] = SPEC_DEFAULT_FREQ_MAX_HZ,
    fs: int = TARGET_SR_HZ,
) -> plt.Figure:
    """Two-panel figure: waterfall on top, single-channel spectrogram below.

    patch: [C, T] or [1, C, T]
    spec_channel: which fiber to spectrogram; defaults to the middle channel.
    """
    img = _as_2d(patch)
    C, _T = img.shape
    ch = spec_channel if spec_channel is not None else C // 2
    ch = max(0, min(C - 1, ch))

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={"height_ratios": [1, 1]})
    _draw_waterfall(axes[0], img, title=title or "patch", fs=fs)
    _draw_spectrogram(axes[1], img[ch], fs=fs, title=f"spectrogram ch={ch}", freq_max=freq_max)
    fig.tight_layout()
    return fig


def plot_input_vs_recon(
    x: np.ndarray,
    x_hat: np.ndarray,
    title: str = "",
    spec_channel: Optional[int] = None,
    freq_max: Optional[float] = SPEC_DEFAULT_FREQ_MAX_HZ,
    fs: int = TARGET_SR_HZ,
) -> plt.Figure:
    """2x2 grid: top row waterfalls (input vs recon), bottom row spectrograms."""
    a = _as_2d(x)
    b = _as_2d(x_hat)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    C, _T = a.shape
    ch = spec_channel if spec_channel is not None else C // 2
    ch = max(0, min(C - 1, ch))

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    _draw_waterfall(axes[0, 0], a, title=f"{title} input" if title else "input", fs=fs)
    _draw_waterfall(axes[0, 1], b, title=f"{title} recon" if title else "recon", fs=fs)
    _draw_spectrogram(axes[1, 0], a[ch], fs=fs, title=f"input spec ch={ch}", freq_max=freq_max)
    _draw_spectrogram(axes[1, 1], b[ch], fs=fs, title=f"recon spec ch={ch}", freq_max=freq_max)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 120):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_multi_waterfall(patches, titles, fs: int = TARGET_SR_HZ) -> plt.Figure:
    """Stack N waterfalls vertically for side-by-side comparison.

    patches: list of [C, T] or [1, C, T] arrays.
    titles:  list of strings, same length as patches.
    """
    patches_2d = [_as_2d(p) for p in patches]
    n = len(patches_2d)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n))
    if n == 1:
        axes = [axes]
    for ax, p, title in zip(axes, patches_2d, titles):
        _draw_waterfall(ax, p, title=title, fs=fs)
    fig.tight_layout()
    return fig


def plot_per_channel_spectrograms(
    patches,
    row_titles,
    fs: int = TARGET_SR_HZ,
    freq_max: Optional[float] = SPEC_DEFAULT_FREQ_MAX_HZ,
) -> plt.Figure:
    """Grid of per-channel spectrograms: N rows (one per patch) x C cols (one per channel).

    patches:    list of [C, T] or [1, C, T] arrays — must all share the same C, T.
    row_titles: short labels for each row.
    """
    patches_2d = [_as_2d(p) for p in patches]
    if not patches_2d:
        raise ValueError("plot_per_channel_spectrograms: empty patches list")
    n = len(patches_2d)
    C = patches_2d[0].shape[0]
    fig, axes = plt.subplots(n, C, figsize=(1.5 * C + 0.5, 1.6 * n))
    if n == 1:
        axes = np.array(axes).reshape(1, -1)
    if C == 1:
        axes = np.array(axes).reshape(-1, 1)
    for i, (p, row_title) in enumerate(zip(patches_2d, row_titles)):
        for j in range(C):
            ax = axes[i, j]
            _draw_spectrogram(ax, p[j], fs=fs, title=None, freq_max=freq_max)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_title, fontsize=9)
            if i == 0:
                ax.set_title(f"ch{j}", fontsize=9)
    fig.tight_layout()
    return fig
