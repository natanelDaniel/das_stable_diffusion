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

import os, sys, json, time, traceback
import numpy as np
import h5py
from glob import glob
from scipy.signal import spectrogram as scipy_spectrogram
from datetime import datetime

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
    """Load and return (raw_full int16 numpy, bitmap bool numpy)."""
    with h5py.File(h5_path, "r") as f:
        raw = f["Acquisition"]["Raw[0]"]["RawData"][:, :]
    npy_path = h5_path[:-3] + ".npy"
    bitmap = np.load(npy_path) if os.path.exists(npy_path) else None
    return raw, bitmap

# ── Debug log helpers ─────────────────────────────────────────────────────────
_LOG_LEVELS = {"INFO": "#58a6ff", "OK": "#3fb950", "WARN": "#d29922", "ERROR": "#f85149"}

def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log_line(level, msg):
    """Return a styled html.Div for one log line."""
    color = _LOG_LEVELS.get(level, TEXT_COLOR)
    return html.Div(
        children=[
            html.Span(f"[{_ts()}] ", style={"color": "#6e7681"}),
            html.Span(f"{level:<5} ", style={"color": color, "fontWeight": "bold"}),
            html.Span(msg, style={"color": TEXT_COLOR}),
        ],
        style={"fontFamily": "monospace", "fontSize": "12px",
               "padding": "1px 4px", "borderBottom": "1px solid #21262d"},
    )

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

        # ── Body: column layout  [top: sidebar+plots] [bottom: debug] ──────
        html.Div(
            style={"display": "flex", "flexDirection": "column",
                   "height": "calc(100vh - 53px)"},
            children=[

                # ── Top row: sidebar + plots ───────────────────────────────
                html.Div(
                    style={"display": "flex", "flex": "1",
                           "minHeight": "0", "overflow": "hidden"},
                    children=[

                        # ── Sidebar ────────────────────────────────────────
                        html.Div(
                            style={"width": "240px", "minWidth": "240px",
                                   "backgroundColor": "#161b22",
                                   "borderRight": "1px solid #30363d",
                                   "padding": "16px", "overflowY": "auto"},
                            children=[

                                html.Label("Class",
                                           style={"fontSize": "11px", "color": "#8b949e",
                                                  "textTransform": "uppercase",
                                                  "letterSpacing": "1px"}),
                                dcc.Dropdown(
                                    id="class-dd",
                                    options=[{"label": c, "value": c} for c in CLASSES],
                                    value=CLASSES[0],
                                    clearable=False,
                                    style={"backgroundColor": PANEL_BG,
                                           "color": TEXT_COLOR,
                                           "border": "1px solid #30363d"},
                                ),
                                html.Br(),

                                html.Label("Sample",
                                           style={"fontSize": "11px", "color": "#8b949e",
                                                  "textTransform": "uppercase",
                                                  "letterSpacing": "1px"}),
                                dcc.Dropdown(
                                    id="sample-dd",
                                    clearable=False,
                                    style={"backgroundColor": PANEL_BG,
                                           "color": TEXT_COLOR,
                                           "border": "1px solid #30363d"},
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
                                                "marginTop": "4px",
                                                "marginBottom": "16px"}),

                                html.Label("Waterfall downsampling",
                                           style={"fontSize": "11px", "color": "#8b949e",
                                                  "textTransform": "uppercase",
                                                  "letterSpacing": "1px"}),
                                dcc.Slider(
                                    id="ds-slider",
                                    min=1, max=32, step=1, value=8,
                                    marks={1: "1×", 8: "8×", 16: "16×", 32: "32×"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": True},
                                    updatemode="mouseup",
                                ),
                                html.Div(style={"fontSize": "10px", "color": "#6e7681",
                                                "marginBottom": "16px"},
                                         children="larger = faster render"),

                                html.Hr(style={"borderColor": "#30363d"}),

                                html.Div(id="info-panel",
                                         style={"fontSize": "11px", "color": "#8b949e",
                                                "lineHeight": "1.8"}),
                            ]
                        ),

                        # ── Plots area ─────────────────────────────────────
                        html.Div(
                            style={"flex": "1", "overflowY": "auto",
                                   "padding": "8px 12px"},
                            children=[
                                dcc.Loading(
                                    type="circle", color=ACCENT,
                                    children=dcc.Graph(
                                        id="main-graph",
                                        config={"displayModeBar": True,
                                                "scrollZoom": True},
                                        style={"height": "100%",
                                               "minHeight": "500px"},
                                    )
                                )
                            ]
                        ),
                    ]
                ),

                # ── Debug console (bottom strip) ──────────────────────────
                html.Div(
                    style={"height": "200px", "minHeight": "200px",
                           "backgroundColor": "#0d1117",
                           "borderTop": "2px solid #30363d",
                           "display": "flex", "flexDirection": "column"},
                    children=[
                        # header bar
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "12px",
                                   "padding": "4px 12px",
                                   "backgroundColor": "#161b22",
                                   "borderBottom": "1px solid #30363d"},
                            children=[
                                html.Span("DEBUG CONSOLE",
                                          style={"fontSize": "11px",
                                                 "color": "#8b949e",
                                                 "letterSpacing": "2px",
                                                 "fontWeight": "bold"}),
                                html.Span(id="debug-status",
                                          style={"fontSize": "11px",
                                                 "color": "#3fb950"}),
                                html.Button(
                                    "Clear",
                                    id="clear-btn",
                                    n_clicks=0,
                                    style={"marginLeft": "auto",
                                           "backgroundColor": "#21262d",
                                           "color": "#8b949e",
                                           "border": "1px solid #30363d",
                                           "borderRadius": "4px",
                                           "padding": "2px 10px",
                                           "cursor": "pointer",
                                           "fontSize": "11px"},
                                ),
                            ]
                        ),
                        # scrollable log area
                        html.Div(
                            id="debug-log",
                            style={"flex": "1", "overflowY": "auto",
                                   "padding": "4px 8px",
                                   "backgroundColor": "#0d1117"},
                        ),
                    ]
                ),
            ]
        ),

        # stores
        dcc.Store(id="data-store"),
        dcc.Store(id="log-store", data=[]),   # persists log lines across renders
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
    Output("debug-log",  "children"),
    Output("debug-status", "children"),
    Input("sample-dd",  "value"),
    Input("ch-slider",  "value"),
    Input("ds-slider",  "value"),
    Input("clear-btn",  "n_clicks"),
    State("debug-log",  "children"),
    prevent_initial_call=False,
)
def update_plots(h5_path, ch_idx, ds, clear_clicks, existing_log):
    """Main render callback — also writes timestamped debug lines."""
    ctx      = dash.callback_context
    trigger  = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    log      = list(existing_log or [])

    # ── Clear button ──────────────────────────────────────────────────
    if trigger == "clear-btn.n_clicks":
        return dash.no_update, dash.no_update, [], "cleared"

    if not h5_path or not os.path.exists(h5_path):
        log.append(log_line("WARN", f"No file: {h5_path}"))
        return go.Figure(), "No data", log, "idle"

    t_total = time.perf_counter()
    fname   = os.path.basename(h5_path)
    log.append(log_line("INFO", f"--- render triggered by: {trigger or 'init'} ---"))
    log.append(log_line("INFO", f"File:  {fname}"))
    log.append(log_line("INFO", f"ch={ch_idx}  ds={ds}x"))

    try:
        # ── Load HDF5 + bitmap ────────────────────────────────────────
        t0 = time.perf_counter()
        raw, bitmap = load_data(h5_path)
        n_time, n_ch = raw.shape
        mb = raw.nbytes / 1e6
        dt = time.perf_counter() - t0
        log.append(log_line("OK",
            f"HDF5 loaded: {n_time:,} x {n_ch}  ({mb:.0f} MB)  in {dt:.2f}s"))

        if bitmap is not None:
            density     = bitmap.sum() / bitmap.size * 100
            n_ev_total  = int(bitmap.sum())
            log.append(log_line("OK",
                f"Bitmap: {bitmap.shape}  events={n_ev_total:,}  ({density:.2f}% filled)"))
        else:
            log.append(log_line("WARN", "No .npy bitmap found alongside HDF5"))

        # ── Waterfall ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        wf     = raw[::ds, :].astype(np.float32)
        n_disp = wf.shape[0]
        t_wf   = np.arange(n_disp) * ds / SR
        vmax   = float(np.percentile(np.abs(wf), 99))
        dt = time.perf_counter() - t0
        log.append(log_line("OK",
            f"Waterfall slice: {n_disp:,} x {n_ch}  vmax={vmax:.0f}  in {dt:.3f}s"))

        # ── Spectrogram ───────────────────────────────────────────────
        t0       = time.perf_counter()
        ch_sig   = raw[:, ch_idx].astype(np.float32)
        nperseg  = 512
        noverlap = 384
        f_spec, t_spec, Sxx = scipy_spectrogram(
            ch_sig, fs=SR, nperseg=nperseg, noverlap=noverlap, window="hann")
        Sxx_db  = 10 * np.log10(Sxx + 1e-12)
        f_mask  = f_spec <= 4000
        f_spec  = f_spec[f_mask]
        Sxx_db  = Sxx_db[f_mask, :]
        dt = time.perf_counter() - t0
        log.append(log_line("OK",
            f"Spectrogram ch{ch_idx}: {Sxx_db.shape}  "
            f"({len(f_spec)} freq bins x {len(t_spec)} frames)  in {dt:.3f}s"))
        log.append(log_line("INFO",
            f"  freq res={SR/nperseg:.1f}Hz  "
            f"time res={(nperseg-noverlap)/SR*1000:.1f}ms  "
            f"dB range=[{Sxx_db.min():.1f}, {Sxx_db.max():.1f}]"))

        # ── Event overlay ─────────────────────────────────────────────
        ev_t, ev_ch_list = [], []
        if bitmap is not None:
            ev_rows, ev_cols = np.where(bitmap)
            ev_t        = list(ev_rows * SHIFT / SR)
            ev_ch_list  = list(ev_cols.astype(float))
            n_ev        = len(ev_t)
            ev_step     = max(1, n_ev // 4000)
            log.append(log_line("INFO",
                f"Event overlay: {n_ev} points, display every {ev_step} "
                f"-> {n_ev // ev_step} rendered"))

        # ── Build figure ──────────────────────────────────────────────
        t0  = time.perf_counter()
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.55, 0.45],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[
                f"Waterfall  [{n_time:,} x {n_ch}]  "
                f"(display {ds}x downsampled -> {n_disp:,} time pts)",
                f"Spectrogram  ch {ch_idx}  (~{ch_idx * SPAT_RES:.0f} m)  |  "
                f"window={nperseg} samp ({nperseg/SR*1000:.1f} ms)  noverlap={noverlap}",
            ],
        )

        fig.add_trace(
            go.Heatmap(
                z=wf.T, x=t_wf,
                y=np.arange(n_ch) * SPAT_RES,
                colorscale="RdBu", reversescale=True,
                zmin=-vmax, zmax=vmax,
                colorbar=dict(
                    title=dict(text="Amplitude", side="right"),
                    len=0.48, y=0.78, thickness=10,
                    tickfont=dict(size=9, color=TEXT_COLOR),
                    titlefont=dict(size=10, color=TEXT_COLOR),
                ),
                hovertemplate="t=%{x:.3f}s  dist=%{y:.0f}m  amp=%{z}<extra></extra>",
            ), row=1, col=1,
        )

        if ev_t:
            fig.add_trace(
                go.Scatter(
                    x=ev_t[::ev_step],
                    y=[c * SPAT_RES for c in ev_ch_list[::ev_step]],
                    mode="markers",
                    marker=dict(size=3, color="rgba(255,255,255,0.5)"),
                    name="Bitmap events",
                    hovertemplate="event t=%{x:.3f}s  dist=%{y:.0f}m<extra></extra>",
                ), row=1, col=1,
            )
            fig.add_hline(
                y=ch_idx * SPAT_RES,
                line=dict(color="#ffd93d", width=1.5, dash="dot"),
                annotation_text=f"ch {ch_idx}",
                annotation_font_color="#ffd93d",
                row=1, col=1,
            )

        fig.add_trace(
            go.Heatmap(
                z=Sxx_db, x=t_spec, y=f_spec / 1000,
                colorscale="Inferno",
                colorbar=dict(
                    title=dict(text="dB", side="right"),
                    len=0.42, y=0.22, thickness=10,
                    tickfont=dict(size=9, color=TEXT_COLOR),
                    titlefont=dict(size=10, color=TEXT_COLOR),
                ),
                hovertemplate="t=%{x:.3f}s  f=%{y:.2f}kHz  %{z:.1f}dB<extra></extra>",
            ), row=2, col=1,
        )

        fig.update_layout(
            **LAYOUT_BASE, height=None, showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
                        font=dict(color=TEXT_COLOR, size=10), x=0.01, y=0.98),
        )
        fig.update_xaxes(**axis_style("Time (s)"),       row=2, col=1)
        fig.update_yaxes(**axis_style("Distance (m)"),   row=1, col=1)
        fig.update_yaxes(**axis_style("Frequency (kHz)"), row=2, col=1)
        for ann in fig.layout.annotations:
            ann.font.color = TEXT_COLOR
            ann.font.size  = 12

        dt_fig = time.perf_counter() - t0
        log.append(log_line("OK", f"Figure built in {dt_fig:.3f}s"))

        # ── Total ─────────────────────────────────────────────────────
        dt_total = time.perf_counter() - t_total
        log.append(log_line("OK", f"Total render: {dt_total:.2f}s"))

        # ── Info panel ────────────────────────────────────────────────
        dur_s = n_time / SR
        info_lines = [
            ("File",     fname),
            ("Shape",    f"{n_time:,} x {n_ch}"),
            ("Size",     f"{mb:.0f} MB  ({n_time*n_ch/1e6:.1f}M samples)"),
            ("Duration", f"{dur_s:.1f} s"),
            ("SR",       f"{SR:,} Hz"),
            ("Spatial",  f"{SPAT_RES:.4f} m/ch  ({n_ch*SPAT_RES:.0f} m total)"),
            ("Bitmap",
             f"{bitmap.shape[0]}x{bitmap.shape[1]}  {n_ev_total:,} events  {density:.2f}%"
             if bitmap is not None else "n/a"),
            ("Display",  f"ds={ds}x  -> {n_disp:,} pts"),
            ("Spec ch",  f"ch {ch_idx}  (~{ch_idx*SPAT_RES:.0f} m)"),
            ("Render",   f"{dt_total:.2f}s"),
        ]
        info_children = [
            html.Div(
                children=[
                    html.Span(k + ": ", style={"color": ACCENT}),
                    html.Span(v, style={"color": TEXT_COLOR}),
                ],
                style={"borderBottom": "1px solid #21262d",
                       "padding": "2px 0", "fontSize": "11px"},
            )
            for k, v in info_lines
        ]

        status = f"OK  {dt_total:.2f}s"
        return fig, info_children, log, status

    except Exception as exc:
        tb = traceback.format_exc()
        log.append(log_line("ERROR", str(exc)))
        for line in tb.splitlines():
            log.append(log_line("ERROR", line))
        return go.Figure(), [html.Div(str(exc), style={"color": "#f85149"})], log, "ERROR"


@app.callback(
    Output("debug-log",    "children", allow_duplicate=True),
    Output("debug-status", "children", allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_log(_):
    return [], "cleared"


if __name__ == "__main__":
    print("Starting DAS Interactive Viewer ...")
    print(f"  Data: {DATA_DIR}")
    print(f"  Classes: {CLASSES}")
    print("  Open http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
