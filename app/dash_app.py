# app.py  (Dash 3+)
# UMAP on the left; Trajectory Viewer on the right. Fixed heights to avoid resize loops.

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
import pandas as pd
import polars as pl
from pathlib import Path

# ---------- Hardcoded paths ----------
POINTS_CSV  = Path("../train_track_df.csv")                # cols: umap_x, umap_y, track_id, participant_id, class, ...
FRAMES_ROOT = Path("../parquet_data")                   # partitions: participant=<ID>/frames.parquet
# -------------------------------------

# ---------- Load points once ----------
POINTS = pd.read_csv(POINTS_CSV)

# Optional: keep only needed columns to reduce payload
POINTS = POINTS[["umap_1", "umap_2", "track_id", "participant_id", "subtype_label"]]

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
            x=df["umap_1"],
            y=df["umap_2"],
            mode="markers",
            marker=dict(size=4, opacity=0.75),
            # Pack track_id, participant_id, class for callbacks
            customdata=df[["track_id", "participant_id", "subtype_label"]].values,
            hovertemplate="track: %{customdata[0]}<br>class: %{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        uirevision="umap-static",   # prevents reflow/re-zoom on updates
        # NOTE: do NOT set height here (we fix it on dcc.Graph)
    )
    return fig

def trajectory_fig(traj: pd.DataFrame, title: str = "Trajectory") -> go.Figure:
    if traj.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Hover or click a point to view its trajectory",
            margin=dict(l=10, r=10, t=40, b=10),
            # NOTE: no height here
        )
        return fig

    # optional downsample for very long tracks
    if len(traj) > 400:
        step = max(1, len(traj) // 400)
        traj = traj.iloc[::step]

    fig = go.Figure(
        go.Scatter(
            x=traj["x"],
            y=traj["y"],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=2),
        )
    )
    # If your coordinates are image-space (0,0 at top-left), flip Y:
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="traj-static",
        # NOTE: no height here
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
        "alignItems": "start",  # stop vertical stretching
    },
    children=[
        dcc.Graph(
            id="umap",
            figure=umap_scatter(POINTS),
            style={"height": "640px"},        # fix height on the component
            config={"responsive": False},     # avoid autosize feedback loop
            clear_on_unhover=False,
        ),
        html.Div(
            children=[
                html.Div(id="traj-meta", style={"marginBottom": "8px", "fontSize": "14px"}),
                dcc.Graph(
                    id="traj-view",
                    figure=trajectory_fig(pd.DataFrame()),
                    style={"height": "320px"},    # fix height on the component
                    config={"responsive": False}, # avoid autosize feedback loop
                ),
            ]
        ),
    ],
)

# ---------- Callbacks ----------
@app.callback(
    Output("traj-view", "figure"),
    Output("traj-meta", "children"),
    Input("umap", "hoverData"),
    Input("umap", "clickData"),
    prevent_initial_call=True,
)
def update_traj_view(hoverData, clickData):
    # Prefer click over hover to reduce disk reads; change if you want hover-first
    ctx = callback_context
    ev = clickData if (ctx.triggered and ctx.triggered[0]["prop_id"].startswith("umap.clickData")) else hoverData
    if not ev or "points" not in ev:
        raise dash.exceptions.PreventUpdate

    p = ev["points"][0]
    track_id, participant_id, klass = p["customdata"]

    traj = get_trajectory(track_id, participant_id)
    title = f"{track_id}  (class={klass})"
    meta  = f"Frames: {len(traj)} • Participant: {participant_id}"
    return trajectory_fig(traj, title), meta

# ---------- Entrypoint ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)   # Dash 3+: use app.run(...)
