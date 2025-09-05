# Trajectory viewer component
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Handle imports for both local development and container
try:
    from ..datastore import get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED, MIN_VIEW_HALF, AUTO_PAD, HALF_LOOKUP, POINTS
except ImportError:
    # For local development
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datastore import get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED, MIN_VIEW_HALF, AUTO_PAD, HALF_LOOKUP, POINTS

def trajectory_fig_centered(traj, center, view_mode, title):
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
        paper_bgcolor="black",  # Outer chart background
        plot_bgcolor="#000000",  # Inner plot area background
        showlegend=False,
    )
    return fig

def get_default_trajectory():
    """Get the first available trajectory for initial display"""
    if POINTS.empty:
        return go.Figure().update_layout(
            title="No data available",
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="black",
            plot_bgcolor="#000000",
            font=dict(color="white"),
        )
    
    # Get the first point from the UMAP data
    first_point = POINTS.iloc[0]
    track_id = first_point["track_id"]
    participant_id = first_point["participant_id"]
    subtype = first_point["subtype_label"]
    
    # Get the trajectory data
    traj = get_trajectory(track_id, participant_id)
    center = CENTER_LOOKUP.get((participant_id, track_id))
    title = f"{track_id} (class={subtype})"
    
    return trajectory_fig_centered(traj, center, "auto", title)

def create_trajectory_component():
    """Create the trajectory viewer component with controls"""
    # Get default trajectory info
    if not POINTS.empty:
        first_point = POINTS.iloc[0]
        track_id = first_point["track_id"]
        participant_id = first_point["participant_id"]
        traj = get_trajectory(track_id, participant_id)
        default_meta = f"Frames: {len(traj)} • Participant: {participant_id} • View: auto"
    else:
        default_meta = "No data available"
    
    return html.Div(children=[
        html.Div(id="traj-meta", style={"marginBottom": "8px", "fontSize": "14px", "color": "white"}, children=default_meta),
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
                style={
                    "fontSize":"13px", 
                    "marginBottom":"6px",
                    "color": "white",
                    "backgroundColor": "transparent"
                }
            )
        ]),
        dcc.Graph(
            id="traj-view",
            style={"height": "380px"},
            config={"responsive": False},
            figure=get_default_trajectory()
        ),
        html.Div(
            f"Fixed half-range (compare mode): {VIEW_HALF_FIXED:.1f} px  •  Quantile=0.95",
            style={"marginTop":"6px", "fontSize":"12px", "color":"#cccccc"}
        ),
    ])

def register_trajectory_callbacks(app):
    """Register the trajectory viewer callbacks"""
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

def register_tsne_trajectory_callbacks(app):
    """Register the t-SNE trajectory viewer callbacks - this will be called dynamically"""
    # This function is kept for compatibility but the actual callback registration
    # is handled by the unified callback in register_trajectory_callbacks
    pass