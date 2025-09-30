"""
Example Dash app page showing archetype fingerprints.

This demonstrates how to use the exported data to recreate the patient fingerprint plots.
"""

from dash import Dash, html, dcc, callback, Input, Output
import plotly.graph_objects as go

from archetype_data_loader import ArchetypeDataLoader
from components import create_archetype_fingerprint, create_simple_trajectory_grid


# Initialize data loader
loader = ArchetypeDataLoader("dash_data")

# Get available archetypes
archetypes = loader.get_archetype_list()
archetype_options = [
    {'label': f"{name}: {loader.get_archetype_info(name).get('title', 'Unknown')}", 
     'value': name}
    for name in archetypes
]


def create_archetype_layout():
    """Create the layout for the archetype fingerprint page."""
    
    return html.Div([
        html.H1("Patient Archetype Fingerprints", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        html.Div([
            html.Label("Select Archetype:", 
                      style={'fontWeight': 'bold', 'marginRight': 10}),
            dcc.Dropdown(
                id='archetype-selector',
                options=archetype_options,
                value=archetypes[0] if archetypes else None,
                style={'width': '400px', 'display': 'inline-block'}
            ),
        ], style={'textAlign': 'center', 'marginBottom': 20}),
        
        html.Div([
            html.H3(id='archetype-info-title', style={'textAlign': 'center'}),
            html.P(id='archetype-info-description', 
                   style={'textAlign': 'center', 'color': '#666'}),
        ]),
        
        # Main fingerprint plot
        dcc.Graph(id='fingerprint-plot', style={'marginBottom': 30}),
        
        # Trajectory grid
        html.H3("Sample Trajectories", style={'textAlign': 'center', 'marginTop': 30}),
        dcc.Graph(id='trajectory-grid'),
        
    ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})


# Callbacks
def register_archetype_callbacks(app):
    """Register callbacks for the archetype page."""
    
    @app.callback(
        [Output('archetype-info-title', 'children'),
         Output('archetype-info-description', 'children'),
         Output('fingerprint-plot', 'figure'),
         Output('trajectory-grid', 'figure')],
        Input('archetype-selector', 'value')
    )
    def update_archetype_display(archetype_name):
        """Update all displays when archetype is selected."""
        
        if not archetype_name:
            return "No archetype selected", "", go.Figure(), go.Figure()
        
        # Get data
        tracks, frames, patient, info = loader.get_archetype_data(archetype_name)
        
        # Get title and description
        title = info.get('title', 'Unknown Archetype')
        description = info.get('description', '')
        pid = info['participant_id']
        
        # Add participant ID to title
        full_title = f"{title} (Participant: {pid})"
        
        # Get CASA scales
        casa_scales = loader.get_casa_scales()
        
        # Pick diverse tracks
        selected_tids = loader.pick_diverse_tracks(tracks, n_total=120, rng_seed=42)
        
        # Create fingerprint plot
        fingerprint_fig = create_archetype_fingerprint(
            tracks_df=tracks,
            frames_df=frames,
            patient_summary=patient,
            casa_scales=casa_scales,
            archetype_title=title,
            selected_track_ids=selected_tids,
            n_tracks_display=120,
            invert_y=True
        )
        
        # Create trajectory grid (show first 40 for cleaner display)
        trajectory_fig = create_simple_trajectory_grid(
            frames_df=frames,
            track_ids=selected_tids[:40],
            title=f"Sample Trajectories for {pid}",
            n_cols=10,
            invert_y=True
        )
        
        return full_title, description, fingerprint_fig, trajectory_fig


# Example standalone app
if __name__ == '__main__':
    app = Dash(__name__)
    
    app.layout = create_archetype_layout()
    register_archetype_callbacks(app)
    
    app.run_server(debug=True, port=8051)
    
    print("\n" + "="*70)
    print("Archetype Fingerprint Viewer")
    print("="*70)
    print(f"Available archetypes: {archetypes}")
    print("Open http://127.0.0.1:8051 in your browser")
    print("="*70)
