# Main Dash app that orchestrates all components
import dash
from dash import html, Input, Output, dcc
import sys
import os

# Add the current directory to Python path for local development
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports for both local development and container
try:
    from components.umap_component import create_umap_component
    from components.tsne_component import create_tsne_component
    from components.trajectory_component import create_trajectory_component, register_trajectory_callbacks
    from components.tsne_trajectory_component import create_tsne_trajectory_component, register_tsne_trajectory_callbacks
except ImportError:
    # For container environment
    from app.components.umap_component import create_umap_component
    from app.components.tsne_component import create_tsne_component
    from app.components.trajectory_component import create_trajectory_component, register_trajectory_callbacks
    from app.components.tsne_trajectory_component import create_tsne_trajectory_component, register_tsne_trajectory_callbacks

# ---------- App ----------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "2fr 1fr",
        "gap": "16px",
        "padding": "12px",
        "alignItems": "start",
        "minHeight": "100vh",
        "backgroundColor": "#000000",
    },
    children=[
        # Embedding visualization component (left side) with tabs
        html.Div([
            dcc.Tabs(
                id="embedding-tabs",
                value="umap-tab",
                children=[
                    dcc.Tab(
                        label="UMAP",
                        value="umap-tab",
                        style={"backgroundColor": "#1a1a1a", "color": "white"},
                        selected_style={"backgroundColor": "#2a2a2a", "color": "white"}
                    ),
                    dcc.Tab(
                        label="t-SNE",
                        value="tsne-tab",
                        style={"backgroundColor": "#1a1a1a", "color": "white"},
                        selected_style={"backgroundColor": "#2a2a2a", "color": "white"}
                    ),
                ],
                style={"backgroundColor": "#000000"}
            ),
            html.Div(id="embedding-content", children=create_umap_component())
        ]),
        
        # Trajectory component (right side) with tabs
        html.Div([
            dcc.Tabs(
                id="trajectory-tabs",
                value="umap-traj-tab",
                children=[
                    dcc.Tab(
                        label="UMAP Trajectory",
                        value="umap-traj-tab",
                        style={"backgroundColor": "#1a1a1a", "color": "white"},
                        selected_style={"backgroundColor": "#2a2a2a", "color": "white"}
                    ),
                    dcc.Tab(
                        label="t-SNE Trajectory",
                        value="tsne-traj-tab",
                        style={"backgroundColor": "#1a1a1a", "color": "white"},
                        selected_style={"backgroundColor": "#2a2a2a", "color": "white"}
                    ),
                ],
                style={"backgroundColor": "#000000"}
            ),
            html.Div(id="trajectory-content", children=create_trajectory_component())
        ]),
        
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

# Add CSS for spinner animation
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

# Register all callbacks
register_trajectory_callbacks(app)
register_tsne_trajectory_callbacks(app)

# Callback to handle tab switching between UMAP and t-SNE
@app.callback(
    Output("embedding-content", "children"),
    Input("embedding-tabs", "value")
)
def update_embedding_content(active_tab):
    if active_tab == "umap-tab":
        return create_umap_component()
    elif active_tab == "tsne-tab":
        return create_tsne_component()
    else:
        return create_umap_component()  # Default fallback

# Callback to handle tab switching between trajectory components
@app.callback(
    Output("trajectory-content", "children"),
    Input("trajectory-tabs", "value")
)
def update_trajectory_content(active_tab):
    if active_tab == "umap-traj-tab":
        return create_trajectory_component()
    elif active_tab == "tsne-traj-tab":
        return create_tsne_trajectory_component()
    else:
        return create_trajectory_component()  # Default fallback

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
