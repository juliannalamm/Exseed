from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Handle imports for both local development and container
try:
    from ..datastore import POINTS, get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datastore import POINTS, get_trajectory, CENTER_LOOKUP, VIEW_HALF_FIXED

KINEMATIC_FEATURES = [
    "ALH", "BCF", "LIN", "MAD", "STR", "VAP", "VCL", "VSL", "WOB"
]

# Track mapping for each motility type (participant_id, track_id)
# Get example tracks from each motility type
def _get_example_tracks():
    """Get one example track for each motility type from POINTS data"""
    tracks = {}
    if not POINTS.empty:
        for subtype in ["erratic", "rapid_progressive", "non-progressive", "immotile"]:
            mask = POINTS["subtype_label"] == subtype
            if mask.any():
                first = POINTS[mask].iloc[0]
                tracks[subtype] = (first["participant_id"], first["track_id"])
    return tracks

TRACK_MAPPING = _get_example_tracks()


def _cluster_means(subtype: str) -> pd.DataFrame:
    df = POINTS
    if df.empty or subtype is None or "subtype_label" not in df.columns:
        return pd.DataFrame({"Feature": KINEMATIC_FEATURES, "Mean": [0] * len(KINEMATIC_FEATURES)})
    mask = df["subtype_label"] == subtype
    if not mask.any():
        return pd.DataFrame({"Feature": KINEMATIC_FEATURES, "Mean": [0] * len(KINEMATIC_FEATURES)})
    cols = [c for c in KINEMATIC_FEATURES if c in df.columns]
    means = df.loc[mask, cols].mean(numeric_only=True)
    return pd.DataFrame({"Feature": means.index, "Mean": means.values})


def _minmax(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mn = float(df["Mean"].min())
    mx = float(df["Mean"].max())
    rng = (mx - mn) or 1.0
    out = df.copy()
    out["Norm"] = (out["Mean"] - mn) / rng
    return out


def _get_default_track_figure():
    """Create a default empty track figure"""
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _create_track_figure(participant_id: str, track_id: str):
    """Create a trajectory figure for the specified track"""
    import numpy as np
    
    traj = get_trajectory(track_id, participant_id)
    fig = go.Figure()
    
    if not traj.empty:
        # Get center
        center = CENTER_LOOKUP.get((participant_id, track_id))
        if center is None:
            cx = 0.5 * (float(traj["x"].min()) + float(traj["x"].max()))
            cy = 0.5 * (float(traj["y"].min()) + float(traj["y"].max()))
        else:
            cx, cy = center
        
        # Center the trajectory
        x0 = (traj["x"] - cx).to_numpy()
        y0 = (traj["y"] - cy).to_numpy()
        
        # Downsample if needed
        if len(x0) > 1200:
            step = max(1, len(x0) // 1200)
            x0 = x0[::step]
            y0 = y0[::step]
        
        # Add trajectory trace
        fig.add_scatter(
            x=x0, y=y0, 
            mode="lines+markers",
            marker=dict(size=4), 
            line=dict(width=2)
        )
    else:
        fig.add_annotation(
            text="Track not found",
            showarrow=False, 
            xref="paper", 
            yref="paper", 
            x=0.5, 
            y=0.5,
            font=dict(size=16, color="white")
        )
    
    # Use fixed range for consistency
    R = VIEW_HALF_FIXED
    
    # Update layout with same styling as trajectory component
    fig.update_xaxes(range=[-R, R], visible=False, fixedrange=True)
    fig.update_yaxes(range=[R, -R], visible=False, fixedrange=True,
                     scaleanchor="x", scaleratio=1)
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        uirevision="track-static",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        showlegend=False,
    )
    
    return fig


def create_cluster_metrics_component():
    options = [
        ("erratic", "Erratic"),
        ("rapid_progressive", "Rapid Progressive"),
        ("non-progressive", "Non-progressive"),
        ("immotile", "Immotile"),
    ]
    return html.Div(
        children=[
            html.H3(
                "Click on the tabs below to explore the metrics associated with each motility type!",
                style={
                    "color": "white",
                    "textAlign": "center",
                    "marginBottom": "16px",
                    "fontSize": "18px",
                    "fontWeight": "500"
                }
            ),
            dcc.Tabs(
                id="metrics-tabs",
                value=options[0][0],
                children=[
                    dcc.Tab(label=label, value=val) for val, label in options
                ],
                style={
                    "color": "black",
                },
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginTop": "16px",
                },
                children=[
                    # Left side: Kinematic chart
                    html.Div(
                        dcc.Graph(id="cluster-metrics", style={"height": "300px"}),
                        style={
                            "borderRadius": "12px",
                            "overflow": "hidden",
                            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                        }
                    ),
                    # Right side: Track display
                    html.Div(
                        [
                            html.Div("Example Track", style={"marginBottom": "8px", "fontSize": "14px", "color": "white", "fontWeight": "600", "textAlign": "center"}),
                            dcc.Graph(
                                id="track-display",
                                style={"height": "300px"},
                                config={"responsive": False},
                                figure=_get_default_track_figure()
                            )
                        ],
                        style={
                            "borderRadius": "12px",
                            "overflow": "hidden",
                            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                        }
                    ),
                ]
            ),
        ]
    )


def register_cluster_metrics_callbacks(app):
    @app.callback(
        [Output("cluster-metrics", "figure"), Output("track-display", "figure")],
        Input("metrics-tabs", "value"),
        prevent_initial_call=False,
    )
    def update_metrics(active_subtype):
        # Update kinematic chart
        stats = _cluster_means(active_subtype)
        stats = _minmax(stats)
        fig = px.bar(
            stats,
            x="Feature",
            y="Norm",
            title=f"{active_subtype}: normalized (0-1) kinematic means",
            color_discrete_sequence=["#7aa2f7"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, title="0-1 scale"),
        )
        
        # Create track display
        if active_subtype in TRACK_MAPPING:
            participant_id, track_id = TRACK_MAPPING[active_subtype]
            track_fig = _create_track_figure(participant_id, track_id)
        else:
            track_fig = _get_default_track_figure()
            track_fig.add_annotation(
                text="No track available",
                showarrow=False, 
                xref="paper", 
                yref="paper", 
                x=0.5, 
                y=0.5,
                font=dict(size=16, color="white")
            )
        
        return fig, track_fig


