# Main Dash app that orchestrates all components
import dash
from dash import html
import sys
import os

# Add the current directory to Python path for local development
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports for both local development and container
try:
    from components.umap_component import create_umap_component
    from components.trajectory_component import create_trajectory_component, register_trajectory_callbacks
except ImportError:
    # For container environment
    from app.components.umap_component import create_umap_component
    from app.components.trajectory_component import create_trajectory_component, register_trajectory_callbacks

# ---------- App ----------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "2fr 1fr",
        "gap": "16px",
        "padding": "12px",
        "alignItems": "start",
    },
    children=[
        create_umap_component(),
        create_trajectory_component(),
    ],
)

# Register all callbacks
register_trajectory_callbacks(app)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
