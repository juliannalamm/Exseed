# Main Dash app that orchestrates all components
import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, dcc
import plotly.graph_objects as go
import sys
import os

# Add the current directory to Python path for local development
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports for both local development and container
try:
    from components.tsne_component import create_tsne_component
    from components.pe_axis_component import create_pe_axis_component
    from components.tsne_trajectory_component import create_tsne_trajectory_component, register_tsne_trajectory_callbacks
    from components.header_component import create_header_component
    from components.cluster_metrics_component import create_cluster_metrics_component, register_cluster_metrics_callbacks
    from components.archetype_radar_section import create_archetype_radar_section, register_archetype_radar_callbacks
    from components.clean_comparison_component import create_clean_comparison_section, register_clean_comparison_callbacks
    from components.velocity_component import create_velocity_component, register_velocity_callbacks
except ImportError:
    # For container environment
    from app.components.tsne_component import create_tsne_component
    from app.components.pe_axis_component import create_pe_axis_component
    from app.components.tsne_trajectory_component import create_tsne_trajectory_component, register_tsne_trajectory_callbacks
    from app.components.header_component import create_header_component
    from app.components.cluster_metrics_component import create_cluster_metrics_component, register_cluster_metrics_callbacks
    from app.components.archetype_radar_section import create_archetype_radar_section, register_archetype_radar_callbacks
    from app.components.clean_comparison_component import create_clean_comparison_section, register_clean_comparison_callbacks
    from app.components.velocity_component import create_velocity_component, register_velocity_callbacks

# ---------- App ----------
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
)
server = app.server

app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "0",
        "padding": "0",
        "alignItems": "stretch",
        "minHeight": "100vh",
        "backgroundColor": "black",
    },
    children=[
        create_header_component(),
        # Content area with padding
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "40px",
                "padding": "20px 40px",
                "flex": "1",
            },
            children=[
                # Page title
                html.H1(
                    "Patient Motility Fingerprinting",
                    style={
                        "color": "white",
                        "textAlign": "center",
                        "marginBottom": "10px",
                        "fontSize": "32px",
                        "fontWeight": "700"
                    }
                ),
              
                # Introduction paragraph
                html.P(
                    [
                        "Sperm display diverse movement patterns throughout their lifecycle, each with implications for fertilization success. Traditional Computer-Assisted Sperm Analysis (CASA) reduces these dynamics into coarse categories, obscuring the heterogeneity that shapes fertilization potential. By applying a probabilistic clustering approach with Gaussian Mixture Models (GMMs), we can uncover distinct motility types directly from the data.",
                        html.Br(),
                        html.Br(),
                        html.Strong("Explore, click, and hover over the data below to learn about the different types of sperm movement", style={"fontWeight": "700"})
                    ],
                    style={
                        "color": "#e0e0e0",
                        "textAlign": "center",
                        "maxWidth": "900px",
                        "margin": "0 auto 30px auto",
                        "fontSize": "16px",
                        "lineHeight": "1.6",
                        "padding": "0 20px"
                    }
                ),
                # Container card for cell-level exploration
                html.Div(
                    style={
                        "backgroundColor": "rgba(26,26,26,0.5)",
                        "borderRadius": "12px",
                        "padding": "20px",
                        "border": "1px solid rgba(99,110,250,0.3)",
                    },
                    children=[
                        # First row: t-SNE chart and trajectory chart side by side
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "2fr 1fr",
                                "gap": "20px",
                                "alignItems": "center",
                                "marginBottom": "20px",
                            },
                            children=[
                                html.Div([
                                    html.Div(id="embedding-content", children=create_tsne_component())
                                ]),
                                html.Div(
                                    id="trajectory-content",
                                    children=[
                                        create_tsne_trajectory_component(),
                                    ],
                                ),
                            ],
                        ),
                        # Second row: Kinematic metrics tabs
                        html.Div(
                            style={
                                "width": "100%",
                                "marginTop": "40px",
                            },
                            children=[
                                create_cluster_metrics_component(),
                            ],
                        ),
                    ]
                ),
                # Section header for P/E axis
                html.Div(
                    style={"textAlign": "center", "marginTop": "20px", "marginBottom": "30px"},
                    children=[
                        html.H2(
                            "Generating Continuous Motility Scores",
                            style={
                                "color": "white",
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "marginBottom": "15px",
                            }
                        ),
                        html.P(
                            "Our cluster-derived labels serve as a soft 'ground truth' that condenses multivariate CASA measurements into a single progressivity–erraticity score, allowing transitional motility behaviors to be represented and providing clinically interpretable readouts.",
                            style={
                                "color": "#e0e0e0",
                                "fontSize": "15px",
                                "marginBottom": "20px",
                                "lineHeight": "1.6"
                            }
                        ),
                    ]
                ),
                # Third row: P/E axis chart and trajectory chart side by side
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "2fr 1fr",
                        "gap": "20px",
                        "alignItems": "center",
                    },
                    children=[
                        html.Div([
                            html.Div(id="pe-axis-content", children=create_pe_axis_component())
                        ]),
                        html.Div(
                            id="pe-trajectory-content",
                            children=[
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "center",
                                        "height": "100%",
                                    },
                                    children=[
                                        html.Div("Sperm Trajectory", style={"marginBottom": "8px", "fontSize": "14px", "color": "white", "fontWeight": "600", "textAlign": "center"}),
                                        # Unified card container for both trajectory and velocity
                                        html.Div(
                                            style={
                                                "backgroundColor": "rgba(26,26,26,0.5)",
                                                "borderRadius": "12px",
                                                "overflow": "hidden",
                                                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                                            },
                                            children=[
                                                # Trajectory graph
                                                dcc.Graph(
                                                    id="pe-traj-view",
                                                    style={"height": "320px"},  # Reduced height to make room for velocity
                                                    config={"responsive": False},
                                                ),
                                                # Add velocity component integrated at the bottom
                                                create_velocity_component("pe-velocity-meters"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
                # Fourth row: Clean Side-by-side Comparison
                html.Div(
                    style={
                        "width": "100%",
                        "marginTop": "40px",
                    },
                    children=[
                        create_clean_comparison_section(),
                    ],
                ),
            ],
        ),
        
        # Loading overlay for app startup
        html.Div(
            id="app-loading-overlay",
            style={
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "100%",
                "backgroundColor": "rgba(255, 255, 255, 0.9)",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "zIndex": 1000,
                "fontFamily": "Arial, sans-serif",
            },
            children=[
                html.Div(
                    style={
                        "width": "50px",
                        "height": "50px",
                        "border": "5px solid #f3f3f3",
                        "borderTop": "5px solid #3498db",
                        "borderRadius": "50%",
                        "animation": "spin 1s linear infinite",
                        "marginBottom": "20px",
                    }
                ),
                html.H3("Loading Dashboard...", style={"margin": "0", "color": "#333"}),
                html.P(
                    "Initializing data and components",
                    style={"margin": "10px 0 0 0", "color": "#666", "fontSize": "14px"}
                ),
            ]
        ),
        
        # Interval to hide loading overlay after 2 seconds
        dcc.Interval(
            id="loading-interval",
            interval=2000,  # 2 seconds
            n_intervals=0,
            max_intervals=1  # Only run once
        ),
    ],
)

# Add CSS for spinner animation and tab styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Custom tab styling - Force override everything */
            .custom-tab,
            div.custom-tab,
            div[role="tab"].custom-tab,
            div[role="tab"][aria-selected="false"],
            #metrics-tabs div[role="tab"]:not([aria-selected="true"]) {
                color: white !important;
                background: linear-gradient(135deg, #636EFA 0%, #4a4ec4 100%) !important;
                background-image: linear-gradient(135deg, #636EFA 0%, #4a4ec4 100%) !important;
                border: 1px solid rgba(99, 110, 250, 0.5) !important;
                border-radius: 25px !important;
                padding: 10px 16px !important;
                font-weight: 500 !important;
                font-size: 14px !important;
                margin-right: 12px !important;
                box-shadow: 0 4px 8px rgba(99, 110, 250, 0.3) !important;
                transition: all 0.3s ease !important;
                cursor: pointer !important;
            }
            
            .custom-tab--selected,
            .tab--selected,
            div[role="tab"][aria-selected="true"],
            div.custom-tab--selected,
            #metrics-tabs div[role="tab"][aria-selected="true"] {
                color: white !important;
                background: rgba(99, 110, 250, 0.35) !important;
                background-color: rgba(99, 110, 250, 0.35) !important;
                background-image: none !important;
                border: 1px solid rgba(99, 110, 250, 0.6) !important;
                border-radius: 25px !important;
                padding: 10px 16px !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 12px rgba(99, 110, 250, 0.3) !important;
            }
            
            /* Extra specific override for selected tabs to ensure translucency */
            #metrics-tabs .custom-tab--selected,
            #metrics-tabs div[role="tab"].custom-tab--selected {
                background: rgba(99, 110, 250, 0.35) !important;
                background-color: rgba(99, 110, 250, 0.35) !important;
                background-image: none !important;
            }
            
            .tab--selected *,
            .tab--selected div,
            div[role="tab"][aria-selected="true"] * {
                color: white !important;
            }
            
            /* Hover effects */
            .custom-tab:hover,
            .tab:hover,
            div[role="tab"][aria-selected="false"]:hover,
            #metrics-tabs div[role="tab"]:not([aria-selected="true"]):hover {
                background: linear-gradient(135deg, #7d7eff 0%, #636EFA 100%) !important;
                background-image: linear-gradient(135deg, #7d7eff 0%, #636EFA 100%) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 5px 10px rgba(99, 110, 250, 0.4) !important;
            }
            
            .custom-tab--selected:hover,
            .tab--selected:hover,
            div[role="tab"][aria-selected="true"]:hover,
            #metrics-tabs div[role="tab"][aria-selected="true"]:hover {
                background: rgba(99, 110, 250, 0.45) !important;
                background-color: rgba(99, 110, 250, 0.45) !important;
                background-image: none !important;
                box-shadow: 0 5px 14px rgba(99, 110, 250, 0.4) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Register t-SNE callbacks and metrics tabs
register_tsne_trajectory_callbacks(app)
register_cluster_metrics_callbacks(app)
register_archetype_radar_callbacks(app)
register_clean_comparison_callbacks(app)
register_velocity_callbacks(app)

# Register P/E axis trajectory callback
try:
    from components.tsne_trajectory_component import trajectory_fig_centered, SUBTYPE_COLORS
    from datastore import get_trajectory, CENTER_LOOKUP
except ImportError:
    from app.components.tsne_trajectory_component import trajectory_fig_centered, SUBTYPE_COLORS
    from app.datastore import get_trajectory, CENTER_LOOKUP

def get_default_pe_trajectory():
    """Get a default progressive trajectory for initial display"""
    # Use the same progressive track as cluster metrics (track 142)
    try:
        traj = get_trajectory("142", "142")
        center = CENTER_LOOKUP.get(("142", "142"))
        return trajectory_fig_centered(traj, center, color=SUBTYPE_COLORS.get("progressive", "#636EFA"))
    except:
        # Fallback to empty figure if data not available
        return go.Figure().update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

@app.callback(
    Output("pe-traj-view", "figure"),
    Input("pe-axis", "hoverData"),
    Input("pe-axis", "clickData"),
    prevent_initial_call=False,
)
def update_pe_traj_view(hoverData, clickData):
    from dash import callback_context
    # Prefer click over hover
    ctx = callback_context
    ev = clickData if (ctx.triggered and ctx.triggered[0]["prop_id"].startswith("pe-axis.clickData")) else hoverData
    
    # On initial call or no data, show default trajectory
    if not ev or "points" not in ev:
        return get_default_pe_trajectory()

    p = ev["points"][0]
    customdata = p["customdata"]
    track_id, participant_id, klass = customdata[0], customdata[1], customdata[2]
    
    # Get the color for this subtype
    subtype_color = SUBTYPE_COLORS.get(klass, "#636EFA")

    traj = get_trajectory(track_id, participant_id)
    center = CENTER_LOOKUP.get((participant_id, track_id))
    return trajectory_fig_centered(traj, center, color=subtype_color)

# Callback to hide loading overlay after timer
@app.callback(
    Output("app-loading-overlay", "style"),
    Input("loading-interval", "n_intervals"),
    prevent_initial_call=True,
)
def hide_loading_overlay(n_intervals):
    return {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100%",
        "height": "100%",
        "backgroundColor": "rgba(255, 255, 255, 0.9)",
        "display": "none",  # Hide after timer
        "zIndex": 1000,
    }

# ---------- Entrypoint ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
