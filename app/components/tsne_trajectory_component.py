# t-SNE trajectory viewer component
import dash
from dash import dcc, html, Input, Output, callback_context
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

def trajectory_fig_centered(traj, center):
    """
    Center the track by subtracting its bbox center (or precomputed center).
    Uses a fixed compare field-of-view (no view mode, no title).
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
    else:
        fig.add_annotation(text="Hover or click a point to view its trajectory",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)

    # fixed compare mode
    R = VIEW_HALF_FIXED

    # Apply equal aspect & reverse Y for image-space (remove reverse if not image coords)
    fig.update_xaxes(range=[-R, R], visible=False, fixedrange=True)
    fig.update_yaxes(range=[R, -R], visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1)

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="traj-static",
        paper_bgcolor="#1a1a1a",  # Outer chart background
        plot_bgcolor="#1a1a1a",  # Inner plot area background
        showlegend=False,
    )
    return fig

def get_default_trajectory():
    """Get the first available trajectory for initial display"""
    if POINTS.empty:
        return go.Figure().update_layout(
            title="No data available",
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="white"),
        )
    
    # Get the first point from the data
    first_point = POINTS.iloc[0]
    track_id = first_point["track_id"]
    participant_id = first_point["participant_id"]
    # Get the trajectory data
    traj = get_trajectory(track_id, participant_id)
    center = CENTER_LOOKUP.get((participant_id, track_id))
    
    return trajectory_fig_centered(traj, center)

def create_tsne_trajectory_component():
    """Create the t-SNE trajectory viewer component (simple header, fixed FOV)."""
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "height": "100%",
        },
        children=[
            html.Div("Sperm Trajectory", style={"marginBottom": "8px", "fontSize": "14px", "color": "white", "fontWeight": "600", "textAlign": "center"}),
            html.Div(
                dcc.Graph(
                    id="tsne-traj-view",
                    style={"height": "380px"},
                    config={"responsive": False},
                    figure=get_default_trajectory()
                ),
                style={
                    "borderRadius": "12px",
                    "overflow": "hidden",
                    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                }
            ),
        ]
    )

def register_tsne_trajectory_callbacks(app):
    """Register the t-SNE trajectory viewer callbacks"""
    @app.callback(
        Output("tsne-traj-view", "figure"),
        Input("tsne", "hoverData"),
        Input("tsne", "clickData"),
        prevent_initial_call=True,
    )
    def update_tsne_traj_view(hoverData, clickData):
        # Prefer click over hover to reduce disk reads; change if you want hover-first
        ctx = callback_context
        ev = clickData if (ctx.triggered and ctx.triggered[0]["prop_id"].startswith("tsne.clickData")) else hoverData
        if not ev or "points" not in ev:
            raise dash.exceptions.PreventUpdate

        p = ev["points"][0]
        customdata = p["customdata"]
        track_id, participant_id, klass = customdata[0], customdata[1], customdata[2]

        traj = get_trajectory(track_id, participant_id)
        center = CENTER_LOOKUP.get((participant_id, track_id))  # may be None; handled inside
        return trajectory_fig_centered(traj, center)
