"""
DAS Interactive Viewer
======================
Run:  python scripts/das_viewer.py
Open: http://127.0.0.1:8052

Controls
--------
- Class / Sample           → pick recording
- Distance range slider    → limits loaded channels (faster I/O + render)
- Waterfall downsampling   → coarser time axis = faster render
- Spec start + stride      → which 12 channels to show as spectrograms

Plots
-----
- Waterfall  : X=distance (m)  Y=time (s)   with bitmap event overlay
- 12 Spectrograms side-by-side : X=time (s)  Y=freq (Hz, 0-256)
"""

import os, sys, time, traceback
import numpy as np
import h5py
from glob import glob
from scipy.signal import spectrogram as scipy_spectrogram, resample_poly
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\netanel.daniel\DAS-dataset\DAS-dataset\data"
SR          = 20_000
SHIFT       = 2048
SPAT_RES    = 1.0209523   # m / channel
N_SPEC      = 12          # side-by-side spectrograms
MAX_CH      = 1700
SPEC_MAX_HZ = 256
SPEC_DEC    = 20          # decimate 20 000 Hz → 1 000 Hz before spectrogram
SR_DEC      = SR // SPEC_DEC   # 1 000 Hz  →  freq_res = 1000/nperseg

CLASSES = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)))

# ── Colours ───────────────────────────────────────────────────────────────────
DARK_BG    = "#0e1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#30363d"

_LOG_COLORS = {
    "INFO": "#58a6ff", "OK": "#3fb950",
    "WARN": "#d29922", "ERROR": "#f85149",
}

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, size=10),
    margin=dict(l=55, r=15, t=50, b=40),
)

# ── Small helpers ─────────────────────────────────────────────────────────────

def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_line(level, msg):
    color = _LOG_COLORS.get(level, TEXT_COLOR)
    return html.Div(
        children=[
            html.Span(f"[{_ts()}] ", style={"color": "#6e7681"}),
            html.Span(f"{level:<5} ", style={"color": color, "fontWeight": "bold"}),
            html.Span(msg, style={"color": TEXT_COLOR}),
        ],
        style={
            "fontFamily": "monospace", "fontSize": "12px",
            "padding": "1px 4px", "borderBottom": "1px solid #21262d",
        },
    )


def _lbl(text):
    return html.Label(
        text,
        style={"fontSize": "11px", "color": "#8b949e",
               "textTransform": "uppercase", "letterSpacing": "1px"},
    )


def ax(**kw):
    d = dict(gridcolor=GRID_COLOR, gridwidth=0.5,
             zerolinecolor=GRID_COLOR, color=TEXT_COLOR)
    d.update(kw)
    return d


def get_samples(cls):
    paths = sorted(glob(os.path.join(DATA_DIR, cls, "*.h5")))
    return [{"label": os.path.basename(p)[:-3], "value": p} for p in paths]


def load_recording(h5_path, ch_start, ch_end, spec_channels):
    """
    One contiguous HDF5 read covering both the waterfall range and
    all spectrogram channels.  Returns:
      raw_block   float32 [n_time, load_end-load_start]
      bitmap_crop bool    [n_windows, ch_end-ch_start]  or None
      load_start, load_end, n_time, n_fiber
    """
    with h5py.File(h5_path, "r") as f:
        raw_ds = f["Acquisition"]["Raw[0]"]["RawData"]
        n_time, n_fiber = raw_ds.shape

        ch_start = max(0, min(ch_start, n_fiber - 1))
        ch_end   = min(n_fiber, max(ch_end, ch_start + 1))

        valid_spec   = [c for c in spec_channels if 0 <= c < n_fiber]
        all_needed   = list(range(ch_start, ch_end)) + valid_spec
        load_start   = max(0, min(all_needed))
        load_end     = min(n_fiber, max(all_needed) + 1)

        raw_block = np.array(raw_ds[:, load_start:load_end], dtype=np.float32)

    npy_path = h5_path[:-3] + ".npy"
    bitmap_crop = None
    if os.path.exists(npy_path):
        bm = np.load(npy_path)                     # [n_win, n_fiber]
        bitmap_crop = bm[:, ch_start:ch_end]

    return raw_block, bitmap_crop, load_start, load_end, n_time, n_fiber


def compute_spectrograms(h5_path, spec_channels):
    """
    Load a minimal HDF5 slice covering spec_channels and return
    (valid_channels, [(f_s, t_s, Sxx_db), ...]).
    Reusable by both the full render and the spec-only Patch callback.
    """
    nperseg, noverlap = 1024, 1008
    with h5py.File(h5_path, "r") as f:
        raw_ds  = f["Acquisition"]["Raw[0]"]["RawData"]
        n_fiber = raw_ds.shape[1]
        valid   = [c for c in spec_channels if 0 <= c < n_fiber]
        if not valid:
            return [], []
        min_ch, max_ch = min(valid), max(valid) + 1
        block = np.array(raw_ds[:, min_ch:max_ch], dtype=np.float32)

    results = []
    for ch in valid:
        sig_dec = resample_poly(
            block[:, ch - min_ch].astype(np.float64), 1, SPEC_DEC
        ).astype(np.float32)
        f_s, t_s, Sxx = scipy_spectrogram(
            sig_dec, fs=SR_DEC, nperseg=nperseg, noverlap=noverlap, window="hann")
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        mask   = f_s <= SPEC_MAX_HZ
        results.append((f_s[mask], t_s, Sxx_db[mask, :]))
    return valid, results


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="DAS Viewer")

app.layout = html.Div(
    style={"backgroundColor": DARK_BG, "minHeight": "100vh",
           "fontFamily": "monospace", "color": TEXT_COLOR},
    children=[

        # ── Header ───────────────────────────────────────────────────────────
        html.Div(
            style={"backgroundColor": PANEL_BG, "padding": "10px 20px",
                   "borderBottom": "1px solid " + GRID_COLOR,
                   "display": "flex", "alignItems": "center", "gap": "16px"},
            children=[
                html.Span("DAS Interactive Viewer",
                          style={"fontSize": "16px", "fontWeight": "bold",
                                 "color": ACCENT}),
                html.Span("SR=20 kHz | 1.02 m/ch | bitmap window=102.4 ms",
                          style={"fontSize": "11px", "color": "#8b949e"}),
            ],
        ),

        # ── Body ─────────────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "flexDirection": "column",
                   "height": "calc(100vh - 45px)"},
            children=[

                # top: sidebar + graph
                html.Div(
                    style={"display": "flex", "flex": "1",
                           "minHeight": "0", "overflow": "hidden"},
                    children=[

                        # ── Sidebar ──────────────────────────────────────────
                        html.Div(
                            style={"width": "270px", "minWidth": "270px",
                                   "backgroundColor": PANEL_BG,
                                   "borderRight": "1px solid " + GRID_COLOR,
                                   "padding": "14px", "overflowY": "auto"},
                            children=[

                                _lbl("Class"),
                                dcc.Dropdown(
                                    id="class-dd",
                                    options=[{"label": c, "value": c} for c in CLASSES],
                                    value="running", clearable=False,
                                    style={"backgroundColor": DARK_BG,
                                           "color": TEXT_COLOR,
                                           "border": "1px solid " + GRID_COLOR},
                                ),
                                html.Br(),

                                _lbl("Sample"),
                                dcc.Dropdown(
                                    id="sample-dd", clearable=False,
                                    style={"backgroundColor": DARK_BG,
                                           "color": TEXT_COLOR,
                                           "border": "1px solid " + GRID_COLOR},
                                ),
                                html.Br(),

                                _lbl("Distance range (channels)"),
                                dcc.RangeSlider(
                                    id="ch-range",
                                    min=0, max=MAX_CH - 1, step=1,
                                    value=[0, 343],
                                    marks={0: "0", 424: "424", 849: "849",
                                           1274: "1274", 1699: "1699"},
                                    allowCross=False,
                                    tooltip={"placement": "bottom",
                                             "always_visible": True},
                                    updatemode="mouseup",
                                ),
                                html.Div(id="ch-range-label",
                                         style={"fontSize": "11px", "color": "#8b949e",
                                                "marginTop": "4px",
                                                "marginBottom": "14px"}),

                                _lbl("Waterfall time downsampling"),
                                dcc.Slider(
                                    id="ds-slider",
                                    min=1, max=32, step=1, value=20,
                                    marks={1: "1x", 8: "8x", 16: "16x", 32: "32x"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": True},
                                    updatemode="mouseup",
                                ),
                                html.Div("larger = faster render",
                                         style={"fontSize": "10px", "color": "#6e7681",
                                                "marginBottom": "14px"}),

                                html.Hr(style={"borderColor": GRID_COLOR}),

                                _lbl("Spectrogram start channel"),
                                dcc.Slider(
                                    id="spec-start",
                                    min=0, max=MAX_CH - 1, step=1, value=200,
                                    marks={0: "0", 424: "424", 849: "849",
                                           1274: "1274", 1699: "1699"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": True},
                                    updatemode="mouseup",
                                ),
                                html.Div(id="spec-start-label",
                                         style={"fontSize": "11px", "color": "#8b949e",
                                                "marginTop": "4px",
                                                "marginBottom": "14px"}),

                                _lbl(f"Spectrogram stride (between {N_SPEC} channels)"),
                                dcc.Slider(
                                    id="spec-stride",
                                    min=1, max=100, step=1, value=10,
                                    marks={1: "1", 10: "10", 50: "50", 100: "100"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": True},
                                    updatemode="mouseup",
                                ),
                                html.Div(f"spacing between each of the {N_SPEC} panels",
                                         style={"fontSize": "10px", "color": "#6e7681",
                                                "marginBottom": "14px"}),

                                html.Hr(style={"borderColor": GRID_COLOR}),
                                html.Div(id="info-panel",
                                         style={"fontSize": "11px", "color": "#8b949e",
                                                "lineHeight": "1.8"}),
                            ],
                        ),

                        # ── Graph ─────────────────────────────────────────────
                        html.Div(
                            style={"flex": "1", "overflowY": "auto",
                                   "padding": "8px 10px"},
                            children=[
                                dcc.Loading(
                                    type="circle", color=ACCENT,
                                    children=dcc.Graph(
                                        id="main-graph",
                                        config={"displayModeBar": True,
                                                "scrollZoom": True},
                                        style={"height": "950px"},
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),

                # ── Debug console ─────────────────────────────────────────────
                html.Div(
                    style={"height": "200px", "minHeight": "200px",
                           "backgroundColor": "#0d1117",
                           "borderTop": "2px solid " + GRID_COLOR,
                           "display": "flex", "flexDirection": "column"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "12px", "padding": "4px 12px",
                                   "backgroundColor": PANEL_BG,
                                   "borderBottom": "1px solid " + GRID_COLOR},
                            children=[
                                html.Span("DEBUG CONSOLE",
                                          style={"fontSize": "11px", "color": "#8b949e",
                                                 "letterSpacing": "2px",
                                                 "fontWeight": "bold"}),
                                html.Span(id="debug-status",
                                          style={"fontSize": "11px",
                                                 "color": "#3fb950"}),
                                html.Button(
                                    "Clear", id="clear-btn", n_clicks=0,
                                    style={"marginLeft": "auto",
                                           "backgroundColor": "#21262d",
                                           "color": "#8b949e",
                                           "border": "1px solid " + GRID_COLOR,
                                           "borderRadius": "4px",
                                           "padding": "2px 10px",
                                           "cursor": "pointer",
                                           "fontSize": "11px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="debug-log",
                            style={"flex": "1", "overflowY": "auto",
                                   "padding": "4px 8px",
                                   "backgroundColor": "#0d1117"},
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="data-store"),
        # tracks how many non-spectrogram traces are in main-graph
        # so the Patch callback knows which indices to update
        dcc.Store(id="trace-offset-store", data={"n": 1}),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("sample-dd", "options"),
    Output("sample-dd", "value"),
    Input("class-dd", "value"),
)
def update_samples(cls):
    opts = get_samples(cls)
    return opts, (opts[0]["value"] if opts else None)


@app.callback(
    Output("ch-range-label", "children"),
    Input("ch-range", "value"),
)
def update_ch_range_label(ch_range):
    a, b = ch_range
    return (f"ch {a}–{b}  "
            f"({a * SPAT_RES:.0f}–{b * SPAT_RES:.0f} m)  "
            f"[{b - a} channels]")


@app.callback(
    Output("spec-start-label", "children"),
    Input("spec-start", "value"),
    Input("spec-stride", "value"),
)
def update_spec_label(spec_start, spec_stride):
    chs = [spec_start + i * spec_stride for i in range(N_SPEC)]
    return (f"ch {chs[0]}–{chs[-1]}  "
            f"({chs[0] * SPAT_RES:.0f}–{chs[-1] * SPAT_RES:.0f} m)")


@app.callback(
    Output("main-graph",          "figure"),
    Output("info-panel",          "children"),
    Output("debug-log",           "children"),
    Output("debug-status",        "children"),
    Output("trace-offset-store",  "data"),
    Input("sample-dd",            "value"),
    Input("ch-range",             "value"),
    Input("ds-slider",            "value"),
    Input("clear-btn",            "n_clicks"),
    State("spec-start",           "value"),
    State("spec-stride",          "value"),
    State("debug-log",            "children"),
    prevent_initial_call=False,
)
def update_plots(h5_path, ch_range, ds, clear_clicks,
                 spec_start, spec_stride, existing_log):
    ctx     = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    log     = list(existing_log or [])

    if trigger == "clear-btn.n_clicks":
        return dash.no_update, dash.no_update, [], "cleared", dash.no_update

    if not h5_path or not os.path.exists(h5_path):
        log.append(log_line("WARN", f"No file: {h5_path}"))
        return go.Figure(), [], log, "idle", dash.no_update

    t_total = time.perf_counter()
    fname   = os.path.basename(h5_path)
    log.append(log_line("INFO", f"--- {trigger or 'init'} ---"))
    log.append(log_line("INFO", f"file={fname}  ds={ds}x"))

    try:
        ch_start, ch_end = ch_range[0], ch_range[1]
        if ch_start >= ch_end:
            ch_end = ch_start + 1

        spec_channels = [spec_start + i * spec_stride for i in range(N_SPEC)]

        # ── Load ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        raw_block, bitmap_crop, load_start, load_end, n_time, n_fiber = \
            load_recording(h5_path, ch_start, ch_end, spec_channels)
        mb = raw_block.nbytes / 1e6
        dt = time.perf_counter() - t0
        log.append(log_line("OK",
            f"loaded block [{load_start}:{load_end}]  "
            f"{n_time:,}t × {load_end - load_start}ch  "
            f"{mb:.0f} MB  {dt:.2f}s"))

        n_ev_total = 0
        if bitmap_crop is not None:
            density    = bitmap_crop.sum() / bitmap_crop.size * 100
            n_ev_total = int(bitmap_crop.sum())
            log.append(log_line("OK",
                f"bitmap {bitmap_crop.shape}  {n_ev_total:,} events  "
                f"{density:.2f}%"))

        # ── Waterfall ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        wf_local = ch_start - load_start
        wf_slice = raw_block[:, wf_local : wf_local + (ch_end - ch_start)]
        wf       = wf_slice[::ds, :]                        # [n_t, n_ch]
        n_t, n_ch = wf.shape
        t_wf     = np.arange(n_t) * ds / SR                 # time (s)
        ch_dist  = (np.arange(n_ch) + ch_start) * SPAT_RES  # distance (m)
        vmax     = float(np.percentile(np.abs(wf), 99))
        log.append(log_line("OK",
            f"waterfall {n_t:,}t × {n_ch}ch  vmax={vmax:.0f}  "
            f"{time.perf_counter()-t0:.3f}s"))

        # ── Spectrograms ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        valid_spec, spec_results = compute_spectrograms(h5_path, spec_channels)
        n_spec_ok = len(spec_results)
        log.append(log_line("OK",
            f"spectrograms: {n_spec_ok}ch  0–{SPEC_MAX_HZ} Hz  "
            f"freq_res={SR_DEC/1024:.1f} Hz  time_res=16 ms  "
            f"{time.perf_counter()-t0:.3f}s"))

        # ── Build figure ─────────────────────────────────────────────────────
        t0 = time.perf_counter()

        spec_titles = [f"{ch * SPAT_RES:.0f}m" for ch in valid_spec]
        spec_titles += [""] * (N_SPEC - len(spec_titles))

        fig = make_subplots(
            rows=2, cols=N_SPEC,
            row_heights=[0.58, 0.42],
            vertical_spacing=0.13,
            horizontal_spacing=0.005,
            specs=[[{"colspan": N_SPEC}] + [None] * (N_SPEC - 1),
                   [{}] * N_SPEC],
            subplot_titles=[
                (f"Waterfall  {ch_start*SPAT_RES:.0f}–{ch_end*SPAT_RES:.0f} m"
                 f"  [{ch_end-ch_start} ch]"
                 f"  ds={ds}x  →  {n_t:,}×{n_ch} pts"),
                *spec_titles,
            ],
        )

        # Row 1: Waterfall  X=distance  Y=time
        fig.add_trace(
            go.Heatmap(
                z=wf,               # [n_t, n_ch]  rows→Y(time), cols→X(distance)
                x=ch_dist,
                y=t_wf,
                colorscale="RdBu", reversescale=True,
                zmin=-vmax, zmax=vmax,
                colorbar=go.heatmap.ColorBar(
                    title_text="Amp",
                    len=0.50, y=0.80, thickness=10, tickfont_size=8,
                ),
                hovertemplate="dist=%{x:.0f}m  t=%{y:.3f}s  amp=%{z}<extra></extra>",
                name="Waterfall",
            ), row=1, col=1,
        )
        n_wf_traces = 1   # waterfall is always trace[0]

        # Event overlay
        if bitmap_crop is not None and n_ev_total > 0:
            ev_rows, ev_cols = np.where(bitmap_crop)
            ev_t   = ev_rows * SHIFT / SR
            ev_d   = (ev_cols + ch_start) * SPAT_RES
            step   = max(1, len(ev_t) // 4000)
            fig.add_trace(
                go.Scatter(
                    x=ev_d[::step], y=ev_t[::step],
                    mode="markers",
                    marker=dict(size=2.5, color="rgba(255,255,80,0.55)"),
                    name="Events",
                    hovertemplate="dist=%{x:.0f}m  t=%{y:.3f}s<extra></extra>",
                ), row=1, col=1,
            )
            n_wf_traces += 1   # event scatter
            log.append(log_line("INFO",
                f"events: {len(ev_t):,} → render every {step} "
                f"→ {len(ev_t)//step} pts"))

        # Mark spectrogram region on waterfall
        if valid_spec:
            d0 = valid_spec[0]  * SPAT_RES
            d1 = valid_spec[-1] * SPAT_RES
            fig.add_vrect(
                x0=d0, x1=d1,
                fillcolor="rgba(88,166,255,0.07)",
                line=dict(color="#58a6ff", width=1, dash="dot"),
                row=1, col=1,
            )

        # Row 2: 12 spectrograms
        # Compute a common dB range across all panels for consistent colour scale
        if spec_results:
            all_db = np.concatenate([r[2].ravel() for r in spec_results])
            db_min = float(np.percentile(all_db, 2))
            db_max = float(np.percentile(all_db, 98))
        else:
            db_min, db_max = -80, 0

        for col_idx, (ch, (f_s, t_s, Sxx_db)) in enumerate(
                zip(valid_spec, spec_results)):
            col_pos  = col_idx + 1
            show_cb  = (col_idx == n_spec_ok - 1)
            fig.add_trace(
                go.Heatmap(
                    z=Sxx_db.T,    # transpose: rows=time frames, cols=freq bins
                    x=f_s,         # X = frequency (Hz)
                    y=t_s,         # Y = time (s)
                    colorscale="Inferno",
                    zmin=db_min, zmax=db_max,
                    showscale=show_cb,
                    colorbar=go.heatmap.ColorBar(
                        title_text="dB",
                        len=0.35, y=0.19, thickness=8, tickfont_size=7,
                    ),
                    hovertemplate=(
                        f"ch{ch} f=%{{x:.1f}}Hz "
                        f"t=%{{y:.2f}}s %{{z:.1f}}dB<extra></extra>"
                    ),
                    name=f"ch{ch}",
                ), row=2, col=col_pos,
            )

        # ── Axes styling ─────────────────────────────────────────────────────
        fig.update_layout(
            **LAYOUT_BASE,
            height=950,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(22,27,34,0.8)", bordercolor=GRID_COLOR,
                font=dict(color=TEXT_COLOR, size=9),
                x=0.01, y=0.97,
            ),
        )

        fig.update_xaxes(**ax(title="Distance (m)"), row=1, col=1)
        fig.update_yaxes(**ax(title="Time (s)"),     row=1, col=1)

        for col_pos in range(1, N_SPEC + 1):
            first = col_pos == 1
            # X = frequency, Y = time  (axes swapped vs old layout)
            fig.update_xaxes(
                **ax(title="Hz" if first else ""),
                range=[0, SPEC_MAX_HZ],
                tickfont=dict(size=7), nticks=5,
                showticklabels=first,
                row=2, col=col_pos,
            )
            fig.update_yaxes(
                **ax(title="s" if first else ""),
                tickfont=dict(size=7), nticks=4,
                showticklabels=first,
                row=2, col=col_pos,
            )

        for ann in fig.layout.annotations:
            ann.font.color = TEXT_COLOR
            ann.font.size  = 9

        dt_fig   = time.perf_counter() - t0
        dt_total = time.perf_counter() - t_total
        log.append(log_line("OK", f"figure built {dt_fig:.3f}s  |  total {dt_total:.2f}s"))

        # ── Info panel ────────────────────────────────────────────────────────
        dur_s = n_time / SR
        rows = [
            ("File",       fname),
            ("Shape",      f"{n_time:,} × {n_fiber}"),
            ("Duration",   f"{dur_s:.1f}s  ({dur_s/60:.1f} min)"),
            ("SR",         f"{SR:,} Hz"),
            ("Spatial",    f"{SPAT_RES:.4f} m/ch"),
            ("Ch range",   f"ch {ch_start}–{ch_end}  ({ch_end-ch_start} ch)"),
            ("Dist range", f"{ch_start*SPAT_RES:.0f}–{ch_end*SPAT_RES:.0f} m"),
            ("Events",
             f"{n_ev_total:,} in range  {density:.2f}%"
             if bitmap_crop is not None else "n/a"),
            ("Spec chs",
             f"ch {valid_spec[0]}–{valid_spec[-1]}  stride={spec_stride}"
             if valid_spec else "n/a"),
            ("Render",     f"{dt_total:.2f}s"),
        ]
        info_children = [
            html.Div(
                children=[
                    html.Span(k + ": ", style={"color": ACCENT}),
                    html.Span(v,        style={"color": TEXT_COLOR}),
                ],
                style={"borderBottom": "1px solid #21262d", "padding": "2px 0"},
            )
            for k, v in rows
        ]

        return fig, info_children, log, f"OK {dt_total:.2f}s", {"n": n_wf_traces}

    except Exception as exc:
        tb = traceback.format_exc()
        log.append(log_line("ERROR", str(exc)))
        for line in tb.splitlines():
            log.append(log_line("ERROR", line))
        return (
            go.Figure(),
            [html.Div(str(exc), style={"color": "#f85149"})],
            log,
            "ERROR",
            {"n": 1},
        )


@app.callback(
    Output("main-graph",         "figure",   allow_duplicate=True),
    Output("debug-log",          "children", allow_duplicate=True),
    Output("debug-status",       "children", allow_duplicate=True),
    Input("spec-start",          "value"),
    Input("spec-stride",         "value"),
    State("sample-dd",           "value"),
    State("trace-offset-store",  "data"),
    State("debug-log",           "children"),
    prevent_initial_call=True,
)
def update_spectrograms_only(spec_start, spec_stride, h5_path,
                              offset_data, existing_log):
    """
    Triggered only by spec-start / spec-stride changes.
    Uses Patch() to replace just the 12 spectrogram traces in-place —
    the waterfall is never re-rendered.
    """
    log = list(existing_log or [])

    if not h5_path or not os.path.exists(h5_path):
        return dash.no_update, log, "idle"

    t0            = time.perf_counter()
    spec_channels = [spec_start + i * spec_stride for i in range(N_SPEC)]

    try:
        valid_spec, spec_results = compute_spectrograms(h5_path, spec_channels)
        n_spec_ok = len(spec_results)

        if not spec_results:
            log.append(log_line("WARN", "no valid spec channels"))
            return dash.no_update, log, "no channels"

        all_db = np.concatenate([r[2].ravel() for r in spec_results])
        db_min = float(np.percentile(all_db, 2))
        db_max = float(np.percentile(all_db, 98))

        n_wf = (offset_data or {}).get("n", 1)
        patched = Patch()

        for i, (ch, (f_s, t_s, Sxx_db)) in enumerate(zip(valid_spec, spec_results)):
            idx = n_wf + i
            patched["data"][idx]["z"]    = Sxx_db.T.tolist()
            patched["data"][idx]["x"]    = f_s.tolist()
            patched["data"][idx]["y"]    = t_s.tolist()
            patched["data"][idx]["zmin"] = db_min
            patched["data"][idx]["zmax"] = db_max
            patched["data"][idx]["name"] = f"ch{ch}"

        # Update the 12 subplot annotation titles (index 0 = waterfall title)
        for i, ch in enumerate(valid_spec):
            patched["layout"]["annotations"][i + 1]["text"] = f"{ch * SPAT_RES:.0f}m"

        dt = time.perf_counter() - t0
        log.append(log_line("OK",
            f"spec patch: {n_spec_ok}ch  "
            f"{valid_spec[0]*SPAT_RES:.0f}–{valid_spec[-1]*SPAT_RES:.0f} m  "
            f"{dt:.2f}s  (waterfall unchanged)"))

        return patched, log, f"spec {dt:.2f}s"

    except Exception as exc:
        tb = traceback.format_exc()
        log.append(log_line("ERROR", str(exc)))
        for line in tb.splitlines():
            log.append(log_line("ERROR", line))
        return dash.no_update, log, "ERROR"


@app.callback(
    Output("debug-log",    "children", allow_duplicate=True),
    Output("debug-status", "children", allow_duplicate=True),
    Input("clear-btn",     "n_clicks"),
    prevent_initial_call=True,
)
def clear_log(_):
    return [], "cleared"


if __name__ == "__main__":
    print("Starting DAS Interactive Viewer ...")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Classes  : {CLASSES}")
    print("  Open http://127.0.0.1:8052")
    app.run(debug=False, host="127.0.0.1", port=8052)
