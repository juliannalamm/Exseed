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

# Color mapping for each motility type (matches t-SNE plot colors)
SUBTYPE_COLORS = {
    "progressive": "#636EFA",      # blue (Plotly default color 0)
    "rapid_progressive": "#EF553B", # red (Plotly default color 1)
    "non_progressive": "#00CC96",   # teal/green (Plotly default color 2)
    "erratic": "#AB63FA",           # purple (Plotly default color 3)
    "immotile": "#FFA15A"           # orange (Plotly default color 4)
}

# Track mapping for each motility type (participant_id, track_id)
# For Felipe data, FID serves as both participant_id and track_id
TRACK_MAPPING = {
    "progressive": ("142", "142"),
    "rapid_progressive": ("1134", "1134"),
    "non_progressive": ("255", "255"),
    "erratic": ("688", "688"),
    "immotile": ("1181", "1181")
}

# Descriptions for each motility type
SUBTYPE_DESCRIPTIONS = {
    "progressive": "A clear, forward trajectory with high linearity and moderate velocity, indicating efficient, directed motion toward an egg.",
    "rapid_progressive": "Rapid progressive sperm move quickly and directly forward. These are the most motile cells with high velocity and strong directional movement, ideal for fertilization.",
    "non_progressive": "Non-progressive sperm move but don't make forward progress. They may move in small circles or have flagellar movement without advancing position.",
    "erratic": "Erratic sperm display irregular, unpredictable movement patterns. They may change direction frequently and exhibit low Linearity (LIN), high (VCL), and moderate to high Beat-Cross Frequency (BCF).",
    "immotile": "Immotile sperm show no movement or only slight flagellar movement without any progression. These cells are stationary or barely moving."
}


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


def _create_track_figure(participant_id: str, track_id: str, color: str = "#636EFA"):
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
        
        # Add trajectory trace with cluster color
        fig.add_scatter(
            x=x0, y=y0, 
            mode="lines+markers",
            marker=dict(size=4, color=color), 
            line=dict(width=2, color=color)
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
        ("progressive", "Progressive"),
        ("rapid_progressive", "Rapid Progressive"),
        ("non_progressive", "Non-progressive"),
        ("erratic", "Erratic"),
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
                    dcc.Tab(
                        label=label,
                        value=val,
                        selected_className="tab--selected",
                        className="tab",
                        selected_style={
                            "color": "white",
                            "background": "linear-gradient(135deg, #636EFA 0%, #4a4ec4 100%)",
                            "borderRadius": "25px",
                            "fontWeight": "500"
                        },
                        style={
                            "color": "white",
                            "background": "linear-gradient(135deg, #404040 0%, #353535 100%)",
                            "borderRadius": "25px",
                            "fontWeight": "500"
                        }
                    ) for val, label in options
                ],
            ),
            # Description box
            html.Div(
                id="motility-description",
                style={
                    "backgroundColor": "#2a2a2a",
                    "padding": "16px 20px",
                    "borderRadius": "8px",
                    "marginTop": "16px",
                    "marginBottom": "16px",
                    "border": "1px solid #404040",
                    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)"
                },
                children=[
                    html.P(
                        SUBTYPE_DESCRIPTIONS.get(options[0][0], ""),
                        style={
                            "color": "#e0e0e0",
                            "margin": "0",
                            "fontSize": "14px",
                            "lineHeight": "1.6",
                            "textAlign": "center"
                        }
                    )
                ]
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
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
        [
            Output("cluster-metrics", "figure"), 
            Output("track-display", "figure"),
            Output("motility-description", "children")
        ],
        Input("metrics-tabs", "value"),
        prevent_initial_call=False,
    )
    def update_metrics(active_subtype):
        # Get color for this subtype
        subtype_color = SUBTYPE_COLORS.get(active_subtype, "#636EFA")
        
        # Update kinematic chart
        stats = _cluster_means(active_subtype)
        stats = _minmax(stats)
        fig = px.bar(
            stats,
            x="Feature",
            y="Norm",
            title=f"{active_subtype}: normalized (0-1) kinematic means",
            color_discrete_sequence=[subtype_color],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, title="0-1 scale"),
        )
        
        # Create track display with matching color
        if active_subtype in TRACK_MAPPING:
            participant_id, track_id = TRACK_MAPPING[active_subtype]
            track_fig = _create_track_figure(participant_id, track_id, color=subtype_color)
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
        
        # Update description text
        description = html.P(
            SUBTYPE_DESCRIPTIONS.get(active_subtype, ""),
            style={
                "color": "#e0e0e0",
                "margin": "0",
                "fontSize": "14px",
                "lineHeight": "1.6",
                "textAlign": "center"
            }
        )
        
        return fig, track_fig, description


