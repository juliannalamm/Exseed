# Main Dash app that orchestrates all components
import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, dcc
import sys
import os

# Add the current directory to Python path for local development
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports for both local development and container
try:
    from components.tsne_component import create_tsne_component
    from components.tsne_trajectory_component import create_tsne_trajectory_component, register_tsne_trajectory_callbacks
    from components.header_component import create_header_component
    from components.cluster_metrics_component import create_cluster_metrics_component, register_cluster_metrics_callbacks
except ImportError:
    # For container environment
    from app.components.tsne_component import create_tsne_component
    from app.components.tsne_trajectory_component import create_tsne_trajectory_component, register_tsne_trajectory_callbacks
    from app.components.header_component import create_header_component
    from app.components.cluster_metrics_component import create_cluster_metrics_component, register_cluster_metrics_callbacks

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
                "gap": "20px",
                "padding": "20px 40px",
                "flex": "1",
            },
            children=[
                # Top row: t-SNE chart and trajectory chart side by side
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "2fr 1fr",
                        "gap": "20px",
                        "alignItems": "center",
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
                # Bottom row: bar chart
                html.Div(
                    style={
                        "width": "100%",
                    },
                    children=[
                        create_cluster_metrics_component(),
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
            
            /* Custom tab styling */
            .tab {
                color: black !important;
            }
            
            .tab--selected {
                color: white !important;
            }
            
            .tab:hover {
                color: white !important;
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

# No tab switching needed; t-SNE is fixed


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
