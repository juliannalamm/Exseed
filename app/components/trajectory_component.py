# Trajectory viewer component
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
from ..datastore import get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED, MIN_VIEW_HALF, AUTO_PAD

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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

def create_trajectory_component():
    """Create the trajectory viewer component with controls"""
    return html.Div(children=[
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
            f"Fixed half-range (compare mode): {VIEW_HALF_FIXED:.1f} px  •  Quantile=0.95",
            style={"marginTop":"6px", "fontSize":"12px", "color":"#666"}
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