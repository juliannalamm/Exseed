"""
Side-by-side comparison component for Felipe data vs Participant 158d356b.
Includes radar charts and animated trajectories.
"""

from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from components.radar_component import create_beautiful_radar
    from archetype_data_loader import ArchetypeDataLoader
except ImportError:
    from app.components.radar_component import create_beautiful_radar
    from app.archetype_data_loader import ArchetypeDataLoader


def load_felipe_data():
    """Load Felipe's reference data."""
    try:
        # Try relative to app directory first, then parent directory
        base_paths = [
            Path('felipe_data'),
            Path('../felipe_data'),
            Path(__file__).parent.parent.parent / 'felipe_data'
        ]
        
        for base_path in base_paths:
            fid_path = base_path / 'fid_level_data.csv'
            traj_path = base_path / 'trajectory.csv'
            
            if fid_path.exists() and traj_path.exists():
                print(f"Loading Felipe data from: {base_path}")
                fid_data = pd.read_csv(fid_path)
                traj_data = pd.read_csv(traj_path)
                return fid_data, traj_data
        
        print("Could not find Felipe data in any expected location")
        return None, None
        
    except Exception as e:
        print(f"Could not load Felipe data: {e}")
        return None, None


def create_animated_trajectories(frames_df, track_ids, title="Trajectories", n_display=120, invert_y=True):
    """
    Create animated trajectory plot.
    
    Parameters
    ----------
    frames_df : pd.DataFrame
        Frame data with columns: track_id, frame_num, x, y
    track_ids : list
        List of track IDs to animate
    title : str
        Plot title
    n_display : int
        Number of tracks to display
    invert_y : bool
        Whether to invert Y axis
        
    Returns
    -------
    go.Figure
        Animated Plotly figure
    """
    
    # Sample tracks
    if len(track_ids) > n_display:
        track_ids = np.random.choice(track_ids, size=n_display, replace=False)
    
    # Prepare frames for animation
    all_frames = []
    max_frame = 0
    
    # Collect all trajectory data
    trajectories = {}
    for tid in track_ids:
        traj = frames_df[frames_df['track_id'] == tid].sort_values('frame_num')
        if traj.empty:
            continue
        
        # Use actual coordinates (no centering)
        x = traj['x'].values
        y = traj['y'].values
        
        if invert_y:
            y = -y
        
        trajectories[tid] = {'x': x, 'y': y, 'frames': traj['frame_num'].values}
        max_frame = max(max_frame, len(x))
    
    # Create figure
    fig = go.Figure()
    
    # Add initial empty traces for each trajectory
    for tid in trajectories.keys():
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=1.5, color='rgba(99, 110, 250, 0.6)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Create animation frames
    frames = []
    for frame_idx in range(max_frame):
        frame_data = []
        for tid in trajectories.keys():
            traj = trajectories[tid]
            end_idx = min(frame_idx + 1, len(traj['x']))
            frame_data.append(go.Scatter(
                x=traj['x'][:end_idx],
                y=traj['y'][:end_idx],
                mode='lines',
                line=dict(width=1.5, color='rgba(99, 110, 250, 0.6)'),
            ))
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    fig.frames = frames
    
    # Update layout
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color='white')),
        xaxis=dict(
            title="x position (pixels)",
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)',
            color='#e0e0e0',
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="y position (pixels)",
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)',
            color='#e0e0e0',
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(26, 26, 26, 0.5)',
        showlegend=False,
        height=400,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': '▶ Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 50, 'redraw': True},
                                'fromcurrent': True, 'mode': 'immediate'}]},
                {'label': '⏸ Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate'}]},
            ],
            'x': 0.1, 'y': 1.15,
        }],
        sliders=[{
            'active': 0,
            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                          'mode': 'immediate'}],
                      'label': str(k), 'method': 'animate'}
                     for k, f in enumerate(fig.frames)],
            'y': -0.1,
            'len': 0.9,
            'x': 0.1,
        }]
    )
    
    return fig


def create_comparison_section():
    """
    Create the comparison section with Felipe data vs Participant 158d356b.
    
    Returns
    -------
    html.Div
        Dash HTML component
    """
    
    # Try to load data
    felipe_fid, felipe_traj = load_felipe_data()
    
    try:
        # Try multiple possible paths for dash_data
        data_paths = ["dash_data", "app/dash_data", "./app/dash_data"]
        loader = None
        for data_path in data_paths:
            try:
                print(f"Trying to load comparison data from: {data_path}")
                loader = ArchetypeDataLoader(data_path)
                print(f"Successfully loaded comparison data from: {data_path}")
                break
            except Exception as path_error:
                print(f"Failed to load from {data_path}: {path_error}")
                continue
        
        if loader is None:
            raise Exception("Could not find dash_data in any expected location")
            
        participant_tracks, participant_frames, participant_summary, participant_info = \
            loader.get_archetype_data('A')  # Assuming 'A' is 158d356b
        has_participant_data = True
    except Exception as e:
        print(f"Could not load participant data: {e}")
        has_participant_data = False
        participant_tracks = None
    
    has_data = (felipe_fid is not None) and has_participant_data
    
    if not has_data:
        return html.Div(
            style={"padding": "40px", "textAlign": "center", "color": "#999"},
            children="Comparison data not available"
        )
    
    # Extract GMM composition for Felipe data
    felipe_post_cols = ['P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 
                       'P_immotile', 'P_erratic']
    felipe_gmm_values = felipe_fid[felipe_post_cols].mean(axis=0).values.tolist()
    felipe_gmm_labels = [c.replace('P_', '').replace('_', ' ').title() 
                        for c in felipe_post_cols]
    
    # Extract GMM composition for participant
    participant_gmm_values = participant_tracks[felipe_post_cols].mean(axis=0).values.tolist()
    participant_gmm_labels = felipe_gmm_labels
    
    # Create radars
    felipe_radar = create_beautiful_radar(
        values=felipe_gmm_values,
        labels=felipe_gmm_labels,
        title="Felipe Data (Reference)",
        color="#636EFA",
        height=450
    )
    
    participant_radar = create_beautiful_radar(
        values=participant_gmm_values,
        labels=participant_gmm_labels,
        title=f"Participant {participant_info['participant_id']}",
        color="#EF553B",
        height=450
    )
    
    return html.Div(
        style={
            "backgroundColor": "rgba(26, 26, 26, 0.5)",
            "borderRadius": "16px",
            "padding": "30px",
            "border": "1px solid rgba(255, 255, 255, 0.1)",
            "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.3)",
        },
        children=[
            # Section header
            html.H2(
                f"Reference Comparison: Felipe Data vs Participant {participant_info['participant_id']}",
                style={
                    "color": "white",
                    "textAlign": "center",
                    "marginBottom": "12px",
                    "fontSize": "26px",
                    "fontWeight": "700",
                }
            ),
            
            html.P(
                "Direct comparison of motility fingerprints between our reference dataset "
                "and a representative individual participant.",
                style={
                    "color": "#e0e0e0",
                    "textAlign": "center",
                    "maxWidth": "800px",
                    "margin": "0 auto 30px auto",
                    "fontSize": "15px",
                    "lineHeight": "1.6",
                }
            ),
            
            # Two-column layout for radar charts
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "30px",
                    "marginBottom": "40px",
                },
                children=[
                    # Left: Felipe data radar
                    html.Div(
                        style={
                            "backgroundColor": "rgba(99, 110, 250, 0.05)",
                            "borderRadius": "12px",
                            "padding": "20px",
                            "border": "2px solid rgba(99, 110, 250, 0.3)",
                        },
                        children=[
                            dcc.Graph(
                                figure=felipe_radar,
                                config={'displayModeBar': False}
                            )
                        ]
                    ),
                    
                    # Right: Participant radar
                    html.Div(
                        style={
                            "backgroundColor": "rgba(239, 85, 59, 0.05)",
                            "borderRadius": "12px",
                            "padding": "20px",
                            "border": "2px solid rgba(239, 85, 59, 0.3)",
                        },
                        children=[
                            dcc.Graph(
                                figure=participant_radar,
                                config={'displayModeBar': False}
                            )
                        ]
                    ),
                ]
            ),
            
            # Trajectory animations section
            html.H3(
                "Representative Trajectories (120 tracks)",
                style={
                    "color": "white",
                    "textAlign": "center",
                    "marginTop": "30px",
                    "marginBottom": "20px",
                    "fontSize": "22px",
                    "fontWeight": "600",
                }
            ),
            
            # Two-column layout for trajectories
            html.Div(
                id='trajectory-animations-container',
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "30px",
                },
                children=[
                    # Left: Felipe trajectories
                    html.Div(
                        style={
                            "backgroundColor": "rgba(99, 110, 250, 0.05)",
                            "borderRadius": "12px",
                            "padding": "20px",
                            "border": "2px solid rgba(99, 110, 250, 0.3)",
                        },
                        children=[
                            dcc.Graph(
                                id='felipe-trajectories',
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ]
                    ),
                    
                    # Right: Participant trajectories  
                    html.Div(
                        style={
                            "backgroundColor": "rgba(239, 85, 59, 0.05)",
                            "borderRadius": "12px",
                            "padding": "20px",
                            "border": "2px solid rgba(239, 85, 59, 0.3)",
                        },
                        children=[
                            dcc.Graph(
                                id='participant-trajectories',
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ]
                    ),
                ]
            ),
        ]
    )


def register_comparison_callbacks(app):
    """Register callbacks for the comparison component."""
    
    # Import the cached data function from clean_comparison_component
    try:
        from app.components.clean_comparison_component import get_cached_data
    except ImportError:
        from components.clean_comparison_component import get_cached_data
    
    # Get cached data
    cached_data = get_cached_data()
    
    if not cached_data['has_data']:
        return
    
    @app.callback(
        [Output('felipe-trajectories', 'figure'),
         Output('participant-trajectories', 'figure')],
        Input('felipe-trajectories', 'id'),  # Trigger on mount
    )
    def create_trajectory_animations(_):
        """Create animated trajectory figures."""
        
        # Use the cached data that was loaded during registration
        felipe_traj = cached_data['felipe_traj']
        loader = cached_data['loader']
        
        # Get participant data
        participant_tracks, participant_frames, _, _ = loader.get_archetype_data('A')
        
        # Felipe trajectories
        felipe_track_ids = felipe_traj['fid'].unique()[:120]
        felipe_frames_formatted = felipe_traj[['fid', 'frame_number', 'x', 'y']].copy()
        felipe_frames_formatted.columns = ['track_id', 'frame_num', 'x', 'y']
        
        felipe_fig = create_animated_trajectories(
            felipe_frames_formatted,
            felipe_track_ids,
            title="Felipe Data Trajectories",
            n_display=120
        )
        felipe_fig.update_traces(line=dict(color='rgba(99, 110, 250, 0.6)'))
        
        # Participant trajectories
        participant_track_ids = participant_tracks['track_id'].unique()[:120]
        participant_fig = create_animated_trajectories(
            participant_frames,
            participant_track_ids,
            title=f"Participant {participant_tracks['participant_id'].iloc[0]} Trajectories",
            n_display=120
        )
        participant_fig.update_traces(line=dict(color='rgba(239, 85, 59, 0.6)'))
        
        return felipe_fig, participant_fig
