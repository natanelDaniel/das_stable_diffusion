"""
DAS Interactive Viewer
======================
Run:  python scripts/das_viewer.py
Open: http://127.0.0.1:8050

Features
--------
- Class selector  → picks event category
- Sample selector → picks the recording file
- Channel slider  → picks which fiber channel to show the spectrogram for
- Waterfall       → full 2D heatmap (time × channels) with bitmap event overlay
- Spectrogram     → STFT of the selected channel (full resolution, no patching)
- Data info panel → matrix shape, duration, sampling rate, bitmap density, spatial res
"""

import os, sys, json
import numpy as np
import h5py
from glob import glob
from scipy.signal import spectrogram as scipy_spectrogram

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"
SR       = 20_000        # Hz
SHIFT    = 2048          # bitmap window stride (samples)
SPAT_RES = 1.0209523     # m / channel

CLASSES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

def get_samples(cls):
    """Return list of (display_name, h5_path) for a class."""
    paths = sorted(glob(os.path.join(DATA_DIR, cls, "*.h5")))
    return [{"label": os.path.basename(p)[:-3], "value": p} for p in paths]

def load_data(h5_path):
    """Load and return (raw_full int16 numpy, bitmap bool numpy, n_ch, n_time)."""
    with h5py.File(h5_path, "r") as f:
        raw = f["Acquisition"]["Raw[0]"]["RawData"][:, :]
    npy_path = h5_path[:-3] + ".npy"
    bitmap = np.load(npy_path) if os.path.exists(npy_path) else None
    return raw, bitmap

# ── Dark plotly theme ─────────────────────────────────────────────────────────
DARK_BG    = "#0e1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#30363d"

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, size=11),
    margin=dict(l=60, r=20, t=40, b=50),
)

def axis_style(title="", tickformat=None):
    d = dict(
        title=title,
        gridcolor=GRID_COLOR, gridwidth=0.5,
        zerolinecolor=GRID_COLOR,
        color=TEXT_COLOR,
    )
    if tickformat:
        d["tickformat"] = tickformat
    return d

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="DAS Viewer")

app.layout = html.Div(
    style={"backgroundColor": DARK_BG, "minHeight": "100vh",
           "fontFamily": "monospace", "color": TEXT_COLOR},
    children=[

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={"backgroundColor": "#161b22", "padding": "12px 24px",
                   "borderBottom": "1px solid #30363d", "display": "flex",
                   "alignItems": "center", "gap": "16px"},
            children=[
                html.Span("DAS Interactive Viewer",
                          style={"fontSize": "18px", "fontWeight": "bold",
                                 "color": ACCENT}),
                html.Span("SR = 20 kHz  |  spatial res ≈ 1.02 m/ch  |  "
                          "bitmap window = 2048 samples (102.4 ms)",
                          style={"fontSize": "12px", "color": "#8b949e"}),
            ]
        ),

        # ── Body: sidebar + plots ─────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "0", "height": "calc(100vh - 53px)"},
            children=[

                # ── Sidebar ────────────────────────────────────────────────
                html.Div(
                    style={"width": "240px", "minWidth": "240px",
                           "backgroundColor": "#161b22",
                           "borderRight": "1px solid #30363d",
                           "padding": "16px", "overflowY": "auto"},
                    children=[

                        html.Label("Class", style={"fontSize": "11px",
                                                    "color": "#8b949e",
                                                    "textTransform": "uppercase",
                                                    "letterSpacing": "1px"}),
                        dcc.Dropdown(
                            id="class-dd",
                            options=[{"label": c, "value": c} for c in CLASSES],
                            value=CLASSES[0],
                            clearable=False,
                            style={"backgroundColor": PANEL_BG,
                                   "color": TEXT_COLOR, "border": "1px solid #30363d"},
                        ),
                        html.Br(),

                        html.Label("Sample", style={"fontSize": "11px",
                                                     "color": "#8b949e",
                                                     "textTransform": "uppercase",
                                                     "letterSpacing": "1px"}),
                        dcc.Dropdown(
                            id="sample-dd",
                            clearable=False,
                            style={"backgroundColor": PANEL_BG,
                                   "color": TEXT_COLOR, "border": "1px solid #30363d"},
                        ),
                        html.Br(),

                        html.Label("Channel for spectrogram",
                                   style={"fontSize": "11px", "color": "#8b949e",
                                          "textTransform": "uppercase",
                                          "letterSpacing": "1px"}),
                        dcc.Slider(
                            id="ch-slider",
                            min=0, max=1699, step=1, value=850,
                            marks={0: "0", 424: "424", 849: "849",
                                   1274: "1274", 1699: "1699"},
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                            updatemode="mouseup",
                        ),
                        html.Div(id="ch-distance",
                                 style={"fontSize": "11px", "color": "#8b949e",
                                        "marginTop": "4px", "marginBottom": "16px"}),

                        html.Label("Waterfall time-downsampling",
                                   style={"fontSize": "11px", "color": "#8b949e",
                                          "textTransform": "uppercase",
                                          "letterSpacing": "1px"}),
                        dcc.Slider(
                            id="ds-slider",
                            min=1, max=32, step=1, value=8,
                            marks={1: "1×", 8: "8×", 16: "16×", 32: "32×"},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="mouseup",
                        ),
                        html.Div(style={"fontSize": "10px", "color": "#6e7681",
                                        "marginBottom": "16px"},
                                 children="↑ larger = faster render"),

                        html.Hr(style={"borderColor": "#30363d"}),

                        # Info panel
                        html.Div(id="info-panel",
                                 style={"fontSize": "11px", "color": "#8b949e",
                                        "lineHeight": "1.8"}),
                    ]
                ),

                # ── Plots area ────────────────────────────────────────────
                html.Div(
                    style={"flex": "1", "overflowY": "auto", "padding": "8px 12px"},
                    children=[
                        dcc.Loading(
                            type="circle", color=ACCENT,
                            children=dcc.Graph(
                                id="main-graph",
                                config={"displayModeBar": True,
                                        "scrollZoom": True},
                                style={"height": "calc(100vh - 80px)"},
                            )
                        )
                    ]
                ),
            ]
        ),

        # hidden store for loaded data summary
        dcc.Store(id="data-store"),
    ]
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("sample-dd", "options"),
    Output("sample-dd", "value"),
    Input("class-dd", "value"),
)
def update_samples(cls):
    opts = get_samples(cls)
    return opts, opts[0]["value"] if opts else None


@app.callback(
    Output("ch-distance", "children"),
    Input("ch-slider", "value"),
)
def update_ch_label(ch):
    return f"≈ {ch * SPAT_RES:.1f} m along fiber"


@app.callback(
    Output("main-graph", "figure"),
    Output("info-panel", "children"),
    Input("sample-dd", "value"),
    Input("ch-slider", "value"),
    Input("ds-slider", "value"),
    prevent_initial_call=False,
)
def update_plots(h5_path, ch_idx, ds):
    if not h5_path or not os.path.exists(h5_path):
        return go.Figure(), "No data"

    # ── Load ──────────────────────────────────────────────────────────────
    raw, bitmap = load_data(h5_path)
    n_time, n_ch = raw.shape

    # ── Waterfall (downsampled in time) ────────────────────────────────
    wf = raw[::ds, :].astype(np.float32)         # [n_disp, n_ch]
    n_disp = wf.shape[0]
    t_wf   = np.arange(n_disp) * ds / SR          # seconds

    vmax = float(np.percentile(np.abs(wf), 99))

    # ── Spectrogram (full resolution, selected channel) ────────────────
    ch_sig = raw[:, ch_idx].astype(np.float32)
    nperseg  = 512
    noverlap = 384
    f_spec, t_spec, Sxx = scipy_spectrogram(
        ch_sig, fs=SR,
        nperseg=nperseg, noverlap=noverlap,
        window="hann",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    # Show only up to 4 kHz
    f_mask = f_spec <= 4000
    f_spec = f_spec[f_mask]
    Sxx_db = Sxx_db[f_mask, :]

    # ── Event overlay for waterfall ────────────────────────────────────
    ev_t, ev_ch = [], []
    if bitmap is not None:
        ev_rows, ev_cols = np.where(bitmap)
        ev_t  = list(ev_rows * SHIFT / SR)
        ev_ch = list(ev_cols.astype(float))

    # ── Subplot figure ─────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"Waterfall  [{n_time} × {n_ch}]  "
            f"(display {ds}× downsampled → {n_disp} time points)",
            f"Spectrogram — channel {ch_idx}  "
            f"(≈ {ch_idx * SPAT_RES:.0f} m)  |  "
            f"full resolution, window={nperseg} samp={nperseg/SR*1000:.1f} ms",
        ],
    )

    # ── Waterfall heatmap ──────────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=wf.T,
            x=t_wf,
            y=np.arange(n_ch) * SPAT_RES,      # meters
            colorscale="RdBu",
            reversescale=True,
            zmin=-vmax, zmax=vmax,
            colorbar=dict(
                title=dict(text="Amplitude", side="right"),
                len=0.48, y=0.78,
                thickness=10,
                tickfont=dict(size=9, color=TEXT_COLOR),
                titlefont=dict(size=10, color=TEXT_COLOR),
            ),
            hovertemplate="t=%{x:.3f}s  ch=%{y:.0f}m  amp=%{z}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Bitmap event overlay ───────────────────────────────────────────
    if ev_t:
        # subsample for display if too many
        n_ev = len(ev_t)
        step = max(1, n_ev // 4000)
        fig.add_trace(
            go.Scatter(
                x=ev_t[::step],
                y=[c * SPAT_RES for c in ev_ch[::step]],
                mode="markers",
                marker=dict(size=3, color="rgba(255,255,255,0.5)",
                            symbol="circle"),
                name="Bitmap events",
                hovertemplate="event t=%{x:.3f}s  ch=%{y:.0f}m<extra></extra>",
            ),
            row=1, col=1,
        )
        # vertical line for selected channel
        fig.add_hline(
            y=ch_idx * SPAT_RES,
            line=dict(color="#ffd93d", width=1.5, dash="dot"),
            annotation_text=f"ch {ch_idx}",
            annotation_font_color="#ffd93d",
            row=1, col=1,
        )

    # ── Spectrogram heatmap ────────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=Sxx_db,
            x=t_spec,
            y=f_spec / 1000,      # kHz
            colorscale="Inferno",
            colorbar=dict(
                title=dict(text="dB", side="right"),
                len=0.42, y=0.22,
                thickness=10,
                tickfont=dict(size=9, color=TEXT_COLOR),
                titlefont=dict(size=10, color=TEXT_COLOR),
            ),
            hovertemplate="t=%{x:.3f}s  f=%{y:.2f}kHz  %{z:.1f}dB<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        **LAYOUT_BASE,
        height=None,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
            font=dict(color=TEXT_COLOR, size=10),
            x=0.01, y=0.98,
        ),
    )
    fig.update_xaxes(
        **axis_style("Time (s)"),
        row=2, col=1,
    )
    fig.update_yaxes(
        **axis_style("Distance (m)"),
        row=1, col=1,
    )
    fig.update_yaxes(
        **axis_style("Frequency (kHz)"),
        row=2, col=1,
    )
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COLOR
        ann.font.size  = 12

    # ── Info panel ────────────────────────────────────────────────────
    fname    = os.path.basename(h5_path)
    dur_s    = n_time / SR
    bm_info  = ""
    if bitmap is not None:
        density = bitmap.sum() / bitmap.size * 100
        n_ev_total = int(bitmap.sum())
        bm_info = (
            f"Bitmap: {bitmap.shape[0]} × {bitmap.shape[1]}\n"
            f"Events: {n_ev_total:,}  ({density:.2f}%)"
        )
    info_lines = [
        f"File:    {fname}",
        f"Raw:     {n_time:,} × {n_ch}",
        f"         = {n_time*n_ch/1e6:.1f}M samples",
        f"Duration: {dur_s:.1f} s",
        f"SR:      {SR:,} Hz",
        f"Spatial: {SPAT_RES:.4f} m/ch",
        f"Range:   {n_ch * SPAT_RES:.0f} m total",
        bm_info,
        f"Display: {ds}× downsamp",
        f"Spec ch: {ch_idx} ({ch_idx*SPAT_RES:.0f} m)",
    ]
    info_children = [
        html.Div(line, style={"borderBottom": "1px solid #21262d",
                              "paddingBottom": "3px", "marginBottom": "3px"})
        for line in info_lines if line
    ]
    return fig, info_children


if __name__ == "__main__":
    print("Starting DAS Interactive Viewer ...")
    print(f"  Data: {DATA_DIR}")
    print(f"  Classes: {CLASSES}")
    print("  Open http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
