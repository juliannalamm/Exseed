# app.py  (Dash 3+)
# UMAP left; Trajectory Viewer right.
# Toggle between fixed FOV (compare) and auto-fit (no clip). Centered, equal aspect.

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# ---------- Hardcoded paths ----------
POINTS_CSV  = Path("../train_track_df.csv")     # must contain: umap_1, umap_2, track_id, participant_id, subtype_label
FRAMES_ROOT = Path("../parquet_data")           # partitions: participant=<ID>/frames.parquet
# -------------------------------------

# ---------- FOV / view settings ----------
FOV_QUANTILE   = 0.95   # fixed "Compare" FOV = p95 of half-spans (change to 0.99 if you still see clipping)
MIN_VIEW_HALF  = 10.0   # smallest half-range so tiny tracks remain visible
AUTO_PAD       = 1.10   # padding for auto-fit (10% extra so tips don't touch the frame)
# ----------------------------------------

# ---------- Load UMAP points ----------
POINTS = pd.read_csv(POINTS_CSV)[["umap_1", "umap_2", "track_id", "participant_id", "subtype_label"]]

# ---------- Precompute per-track centers & spans; compute fixed FOV ----------
def build_track_index(frames_root: Path):
    """
    Returns:
      idx_df (pandas): [participant_id, track_id, cx, cy, half_span]
      view_half_fixed (float): fixed half-range for 'Compare' mode from quantile
      view_half_max   (float): global max half-range (useful if you want 'No-clip fixed')
    """
    pattern = str(frames_root / "participant=*/frames.parquet")
    lf = pl.scan_parquet(pattern)

    agg = (
        lf.group_by(["participant_id", "track_id"])
          .agg([
              pl.col("x").min().alias("xmin"),
              pl.col("x").max().alias("xmax"),
              pl.col("y").min().alias("ymin"),
              pl.col("y").max().alias("ymax"),
          ])
          .with_columns([
              ((pl.col("xmin") + pl.col("xmax")) / 2.0).alias("cx"),
              ((pl.col("ymin") + pl.col("ymax")) / 2.0).alias("cy"),
              (pl.max_horizontal(pl.col("xmax") - pl.col("xmin"),
                                 pl.col("ymax") - pl.col("ymin")) / 2.0).alias("half_span"),
          ])
          .select(["participant_id", "track_id", "cx", "cy", "half_span"])
    )

    tracks = agg.collect()
    if tracks.is_empty():
        df = pd.DataFrame(columns=["participant_id","track_id","cx","cy","half_span"])
        return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)

    df = tracks.to_pandas()
    hs = df["half_span"].to_numpy()
    hs = hs[np.isfinite(hs)]
    if hs.size == 0:
        return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)

    view_half_fixed = max(float(np.quantile(hs, FOV_QUANTILE)), MIN_VIEW_HALF)
    view_half_max   = max(float(hs.max()), MIN_VIEW_HALF)
    return df, view_half_fixed, view_half_max

TRACK_IDX, VIEW_HALF_FIXED, VIEW_HALF_MAX = build_track_index(FRAMES_ROOT)

# Fast lookups
CENTER_LOOKUP = {(r.participant_id, r.track_id): (float(r.cx), float(r.cy))
                 for r in TRACK_IDX.itertuples(index=False)}
HALF_LOOKUP   = {(r.participant_id, r.track_id): float(r.half_span)
                 for r in TRACK_IDX.itertuples(index=False)}

# ---------- Data access ----------
def get_trajectory(track_id: str, participant_id: str) -> pd.DataFrame:
    """Fetch a single track's frames from the participant Parquet."""
    p = FRAMES_ROOT / f"participant={participant_id}" / "frames.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["frame_num", "x", "y"])
    df = pl.read_parquet(p)
    out = (
        df.filter(pl.col("track_id") == track_id)
          .sort("frame_num")
          .select(["frame_num", "x", "y"])
          .to_pandas()
    )
    return out

# ---------- Figure builders ----------
def umap_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scattergl(
            x=df["umap_1"], y=df["umap_2"],
            mode="markers",
            marker=dict(size=4, opacity=0.75),
            customdata=df[["track_id","participant_id","subtype_label"]].values,
            hovertemplate="track: %{customdata[0]}<br>class: %{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        uirevision="umap-static",
    )
    return fig

def trajectory_fig_centered(traj: pd.DataFrame,
                            center: tuple[float,float] | None,
                            view_mode: str,
                            title: str) -> go.Figure:
    """
    Center the track by subtracting its bbox center (or precomputed center).
    view_mode:
      - 'compare': fixed ±VIEW_HALF_FIXED (consistent scale; outliers may clip)
      - 'auto':    per-track ±(half_span * AUTO_PAD) (no clip; scales differ)
    """
    fig = go.Figure()

    if not traj.empty:
        # center
        if center is None:
            cx = 0.5 * (float(traj["x"].min()) + float(traj["x"].max()))
            cy = 0.5 * (float(traj["y"].min()) + float(traj["y"].max()))
        else:
            cx, cy = center
        x0 = (traj["x"] - cx).to_numpy()
        y0 = (traj["y"] - cy).to_numpy()

        # light downsample for very long tracks
        if len(x0) > 1200:
            step = max(1, len(x0) // 1200)
            x0 = x0[::step]; y0 = y0[::step]

        fig.add_scatter(x=x0, y=y0, mode="lines+markers",
                        marker=dict(size=4), line=dict(width=2))
        show_title = title
    else:
        fig.add_annotation(text="Hover or click a point to view its trajectory",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        show_title = "Trajectory"

    # Choose half-range
    if view_mode == "auto":
        # derive per-track half-span from data in view (or fallback)
        if not traj.empty:
            # half-span from the (possibly downsampled) trajectory bbox
            hs_x = float(np.max(x0) - np.min(x0)) / 2.0 if len(x0) else 0.0
            hs_y = float(np.max(y0) - np.min(y0)) / 2.0 if len(y0) else 0.0
            half = max(hs_x, hs_y)
        else:
            half = MIN_VIEW_HALF
        R = max(half * AUTO_PAD, MIN_VIEW_HALF)
    else:
        # fixed compare mode
        R = VIEW_HALF_FIXED

    # Apply equal aspect & reverse Y for image-space (remove reverse if not image coords)
    fig.update_xaxes(range=[-R, R], visible=False, fixedrange=True)
    fig.update_yaxes(range=[R, -R], visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=show_title,
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="traj-static",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

# ---------- App ----------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "2fr 1fr",
        "gap": "16px",
        "padding": "12px",
        "alignItems": "start",
    },
    children=[
        dcc.Graph(
            id="umap",
            figure=umap_scatter(POINTS),
            style={"height": "640px"},
            config={"responsive": False},
            clear_on_unhover=False,
        ),
        html.Div(children=[
            html.Div(id="traj-meta", style={"marginBottom": "8px", "fontSize": "14px"}),
            # view mode toggle
            html.Div([
                dcc.RadioItems(
                    id="view-mode",
                    options=[
                        {"label": "Compare (fixed FOV)", "value": "compare"},
                        {"label": "Auto-fit (no clip)",  "value": "auto"},
                    ],
                    value="auto",
                    inline=True,
                    style={"fontSize":"13px", "marginBottom":"6px"}
                )
            ]),
            dcc.Graph(
                id="traj-view",
                style={"height": "380px"},
                config={"responsive": False},
                figure=go.Figure().update_layout(
                    title="Hover or click a point to view its trajectory",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
            ),
            html.Div(
                f"Fixed half-range (compare mode): {VIEW_HALF_FIXED:.1f} px  •  Quantile={FOV_QUANTILE}",
                style={"marginTop":"6px", "fontSize":"12px", "color":"#666"}
            ),
        ])
    ],
)

# ---------- Callbacks ----------
@app.callback(
    Output("traj-view", "figure"),
    Output("traj-meta", "children"),
    Input("umap", "hoverData"),
    Input("umap", "clickData"),
    Input("view-mode", "value"),
    prevent_initial_call=True,
)
def update_traj_view(hoverData, clickData, view_mode):
    # Prefer click over hover to reduce disk reads; change if you want hover-first
    ctx = callback_context
    ev = clickData if (ctx.triggered and ctx.triggered[0]["prop_id"].startswith("umap.clickData")) else hoverData
    if not ev or "points" not in ev:
        raise dash.exceptions.PreventUpdate

    p = ev["points"][0]
    track_id, participant_id, klass = p["customdata"]

    traj = get_trajectory(track_id, participant_id)
    center = CENTER_LOOKUP.get((participant_id, track_id))  # may be None; handled inside
    title  = f"{track_id}  (class={klass})"
    meta   = f"Frames: {len(traj)} • Participant: {participant_id} • View: {view_mode}"
    return trajectory_fig_centered(traj, center, view_mode, title), meta

# ---------- Entrypoint ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
