# t-SNE trajectory component for drug-colored view
import dash
from dash import dcc, html, Input, Output, callback, callback_context
import plotly.graph_objects as go
import sys
import os

# Handle imports for both local development and container
try:
    from ..datastore import POINTS, get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED
except ImportError:
    # For local development
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datastore import POINTS, get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED

def trajectory_fig_centered_drug(traj, center, color="#636EFA"):
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
                        marker=dict(size=4, color=color), line=dict(width=2, color=color))
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
        uirevision="traj-drug-static",
        paper_bgcolor="rgba(26,26,26,0.5)",  # Match card background
        plot_bgcolor="rgba(26,26,26,0.5)",  # Match card background
        showlegend=False,
    )
    return fig

def get_default_trajectory_drug():
    """Get the first available trajectory for initial display"""
    if POINTS.empty:
        return go.Figure().update_layout(
            title="No data available",
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(26,26,26,0.5)",
            plot_bgcolor="rgba(26,26,26,0.5)",
            font=dict(color="white"),
        )
    
    # Get the first point from the data
    first_point = POINTS.iloc[0]
    track_id = first_point["track_id"]
    participant_id = first_point["participant_id"]
    # Get the trajectory data
    traj = get_trajectory(track_id, participant_id)
    center = CENTER_LOOKUP.get((participant_id, track_id))
    
    return trajectory_fig_centered_drug(traj, center)

def create_tsne_trajectory_drug_component():
    """Create the t-SNE trajectory component for drug view with integrated velocity component."""
    # Import velocity component here to avoid circular imports
    try:
        from .velocity_component import create_velocity_component
    except ImportError:
        from velocity_component import create_velocity_component
    
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "height": "100%",
            "width": "100%",
            "maxWidth": "100%",
            "boxSizing": "border-box",
            "overflow": "hidden",
        },
        children=[
            html.Div("Sperm Trajectory", style={"marginBottom": "8px", "fontSize": "14px", "color": "white", "fontWeight": "600", "textAlign": "center"}),
            # Unified card container for both trajectory and velocity
            html.Div(
                style={
                    "backgroundColor": "rgba(26,26,26,0.5)",
                    "borderRadius": "12px",
                    "overflow": "hidden",
                    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                    "width": "100%",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                },
                children=[
                    # Trajectory graph (no separate card styling)
                    dcc.Graph(
                        id="tsne-traj-view-drug",
                        style={"height": "320px", "width": "100%", "maxWidth": "100%"},
                        config={"responsive": False},
                        figure=get_default_trajectory_drug()
                    ),
                    # Velocity component integrated at the bottom
                    create_velocity_component("tsne-velocity-meters-drug"),
                ]
            ),
        ]
    )

def register_tsne_trajectory_drug_callbacks(app):
    """Register the drug t-SNE trajectory viewer callbacks"""
    @app.callback(
        Output("tsne-traj-view-drug", "figure"),
        Input("tsne-drug", "hoverData"),
        Input("tsne-drug", "clickData"),
        prevent_initial_call=True,
    )
    def update_tsne_traj_view_drug(hoverData, clickData):
        # Prefer click over hover to reduce disk reads; change if you want hover-first
        ctx = callback_context
        ev = clickData if (ctx.triggered and ctx.triggered[0]["prop_id"].startswith("tsne-drug.clickData")) else hoverData
        if not ev or "points" not in ev:
            raise dash.exceptions.PreventUpdate

        p = ev["points"][0]
        customdata = p["customdata"]
        track_id, participant_id, klass = customdata[0], customdata[1], customdata[2]
        
        # Get the drug color based on the drug ID
        drug_id = customdata[4] if len(customdata) > 4 else None
        drug_colors = [
            "#636EFA",  # blue
            "#EF553B",  # red  
            "#00CC96",  # teal/green
            "#AB63FA",  # purple
            "#FFA15A",  # orange
            "#19D3F3",  # cyan
            "#FF6692",  # pink
            "#B6E880",  # light green
            "#FF97FF",  # magenta
            "#FECB52"   # yellow
        ]
        
        # Get unique drug IDs to find the index
        if "experiment_media" in POINTS.columns:
            unique_drugs = sorted(POINTS["experiment_media"].unique())
            try:
                drug_index = unique_drugs.index(drug_id)
                subtype_color = drug_colors[drug_index % len(drug_colors)]
            except (ValueError, TypeError):
                subtype_color = "#636EFA"  # default blue
        else:
            subtype_color = "#636EFA"  # default blue

        traj = get_trajectory(track_id, participant_id)
        center = CENTER_LOOKUP.get((participant_id, track_id))  # may be None; handled inside
        return trajectory_fig_centered_drug(traj, center, color=subtype_color)
