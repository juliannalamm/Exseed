"""
Clean side-by-side comparison component for patient fingerprints.
Professional design with dropdown selection and comprehensive visualizations.
"""

from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from archetype_data_loader import ArchetypeDataLoader
except ImportError:
    from app.archetype_data_loader import ArchetypeDataLoader


# Cluster colors matching your existing theme
CLUSTER_COLORS = {
    'progressive': '#00CC96',
    'rapid_progressive': '#AB63FA',
    'nonprogressive': '#FFA15A',
    'immotile': '#636EFA',
    'erratic': '#EF553B',
}


def load_felipe_data():
    """Load Felipe's reference data."""
    try:
        base_paths = [
            Path('felipe_data'),
            Path('../felipe_data'),
            Path(__file__).parent.parent.parent / 'felipe_data'
        ]
        
        for base_path in base_paths:
            fid_path = base_path / 'fid_level_data.csv'
            traj_path = base_path / 'trajectory.csv'
            
            if fid_path.exists() and traj_path.exists():
                fid_data = pd.read_csv(fid_path)
                traj_data = pd.read_csv(traj_path)
                return fid_data, traj_data
        
        return None, None
    except Exception as e:
        print(f"Could not load Felipe data: {e}")
        return None, None


def create_pe_scatter(tracks_df, title="P-E Scatter"):
    """Create P-E axis scatter plot."""
    fig = go.Figure()
    
    # Handle different column names
    p_col = 'progressivity' if 'progressivity' in tracks_df.columns else 'P_axis_byls'
    e_col = 'erraticity' if 'erraticity' in tracks_df.columns else 'E_axis_byls'
    
    # Color by cluster if available
    if 'subtype_label' in tracks_df.columns:
        for cluster in tracks_df['subtype_label'].unique():
            cluster_data = tracks_df[tracks_df['subtype_label'] == cluster]
            color = CLUSTER_COLORS.get(cluster, '#636EFA')
            
            fig.add_trace(go.Scatter(
                x=cluster_data[p_col],
                y=cluster_data[e_col],
                mode='markers',
                name=cluster.replace('_', ' ').title(),
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.6,
                    line=dict(width=0)
                ),
                showlegend=False,
            ))
    else:
        fig.add_trace(go.Scatter(
            x=tracks_df[p_col],
            y=tracks_df[e_col],
            mode='markers',
            marker=dict(size=5, color='#636EFA', opacity=0.6),
            showlegend=False,
        ))
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color='white'), x=0.5),
        xaxis=dict(
            title="Progressivity", 
            gridcolor='rgba(255,255,255,0.1)', 
            color='#e0e0e0', 
            range=[0, 1],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Erraticity", 
            gridcolor='rgba(255,255,255,0.1)', 
            color='#e0e0e0', 
            range=[0, 1],
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,26,0.3)',
        height=300,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    
    return fig


def create_cluster_distribution(tracks_df, title="Cluster Distribution"):
    """Create bar chart of cluster distribution."""
    if 'subtype_label' not in tracks_df.columns:
        return go.Figure()
    
    # Count clusters
    cluster_counts = tracks_df['subtype_label'].value_counts()
    cluster_pcts = (cluster_counts / len(tracks_df) * 100).sort_index()
    
    colors = [CLUSTER_COLORS.get(cluster, '#636EFA') for cluster in cluster_pcts.index]
    labels = [c.replace('_', ' ').title() for c in cluster_pcts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=cluster_pcts.values,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f'{v:.1f}%' for v in cluster_pcts.values],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color='white'), x=0.5),
        xaxis=dict(title="", color='#e0e0e0', tickangle=-45),
        yaxis=dict(title="Percentage (%)", gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,26,0.3)',
        height=280,
        margin=dict(l=50, r=20, t=50, b=70),
        showlegend=False,
    )
    
    return fig


def create_clean_trajectories(frames_df, tracks_df, title="Trajectories", n_display=120):
    """Create clean trajectory animation with cluster colors, no grid."""
    
    # Handle different column names (Felipe uses 'fid', participants use 'track_id')
    track_col = 'fid' if 'fid' in tracks_df.columns else 'track_id'
    frame_track_col = 'fid' if 'fid' in frames_df.columns else 'track_id'
    
    # Sample tracks
    track_ids = tracks_df[track_col].unique()
    if len(track_ids) > n_display:
        track_ids = np.random.choice(track_ids, size=n_display, replace=False)
    
    # Get cluster info for coloring
    track_clusters = {}
    if 'subtype_label' in tracks_df.columns:
        for tid in track_ids:
            cluster = tracks_df[tracks_df[track_col] == tid]['subtype_label'].iloc[0]
            track_clusters[tid] = cluster
    
    # Collect trajectories
    trajectories = {}
    max_frame = 0
    
    # Handle different frame number column names
    frame_num_col = 'frame_number' if 'frame_number' in frames_df.columns else 'frame_num'
    
    for tid in track_ids:
        traj = frames_df[frames_df[frame_track_col] == tid].sort_values(frame_num_col)
        if traj.empty:
            continue
        
        x = traj['x'].values
        y = -traj['y'].values  # Invert Y
        
        cluster = track_clusters.get(tid, 'unknown')
        color = CLUSTER_COLORS.get(cluster, '#636EFA')
        
        trajectories[tid] = {
            'x': x, 
            'y': y, 
            'cluster': cluster,
            'color': color
        }
        max_frame = max(max_frame, len(x))
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each trajectory
    for tid, traj in trajectories.items():
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=1.2, color=traj['color']),
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Create animation frames
    frames = []
    for frame_idx in range(max_frame):
        frame_data = []
        for tid, traj in trajectories.items():
            end_idx = min(frame_idx + 1, len(traj['x']))
            frame_data.append(go.Scatter(
                x=traj['x'][:end_idx],
                y=traj['y'][:end_idx],
                mode='lines',
                line=dict(width=1.2, color=traj['color']),
                opacity=0.7,
            ))
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    fig.frames = frames
    
    # Clean layout - NO GRID
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color='white'), x=0.5),
        xaxis=dict(
            title="",
            showgrid=False,  # NO GRID
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="",
            showgrid=False,  # NO GRID
            zeroline=False,
            showticklabels=False,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,26,0.3)',
        showlegend=False,
        height=450,  # Big but not too big
        margin=dict(l=20, r=20, t=50, b=20),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': '▶', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 50, 'redraw': True},
                                'fromcurrent': True, 'mode': 'immediate'}]},
                {'label': '⏸', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate'}]},
            ],
            'x': 0.5, 'y': -0.05, 'xanchor': 'center',
            'font': dict(size=16, color='white'),
            'bgcolor': 'rgba(99, 110, 250, 0.3)',
            'bordercolor': 'rgba(99, 110, 250, 0.5)',
        }],
    )
    
    return fig


def create_clean_radar(values, labels, title=""):
    """Create clean radar with fewer rings."""
    
    # Close the loop
    theta = labels + [labels[0]]
    r = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.3)',
        line=dict(color='#636EFA', width=2.5),
        hovertemplate='<b>%{theta}</b><br>%{r:.1%}<extra></extra>',
    ))
    
    # Determine range
    max_val = max(values) if values else 1.0
    range_max = max(1.0, max_val * 1.1)
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, range_max],
                tickformat='.0%',
                tickfont=dict(size=10, color='#e0e0e0'),
                gridcolor='rgba(255,255,255,0.15)',
                nticks=4,  # FEWER RINGS
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='white'),
                gridcolor='rgba(255,255,255,0.15)',
            ),
            bgcolor='rgba(26,26,26,0.3)',
        ),
        showlegend=False,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color='white'), x=0.5),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=50, b=40),
    )
    
    return fig


def create_clean_comparison_section():
    """Create clean comparison section with dropdown."""
    
    # Load data
    felipe_fid, felipe_traj = load_felipe_data()
    
    try:
        loader = ArchetypeDataLoader("dash_data")
        has_data = True
        participant_list = loader.patient_df['participant_id'].unique().tolist()
    except:
        has_data = False
        participant_list = []
    
    if not has_data or felipe_fid is None:
        return html.Div("Data not available", style={"color": "#999", "padding": "40px"})
    
    return html.Div(
        style={
            "backgroundColor": "rgba(0,0,0,0)",
            "padding": "20px 0",
        },
        children=[
            # Header and dropdown
            html.Div(
                style={"textAlign": "center", "marginBottom": "30px"},
                children=[
                    html.H2(
                        "Patient Comparison",
                        style={
                            "color": "white",
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "marginBottom": "15px",
                        }
                    ),
                    html.P(
                        "Compare motility fingerprints between reference data and individual participants",
                        style={"color": "#e0e0e0", "fontSize": "15px", "marginBottom": "20px"}
                    ),
                    html.Div(
                        style={"display": "flex", "justifyContent": "center", "alignItems": "center", "gap": "15px"},
                        children=[
                            html.Label("Select Participant:", style={"color": "white", "fontWeight": "600"}),
                            dcc.Dropdown(
                                id='comparison-participant-dropdown',
                                options=[{'label': pid, 'value': pid} for pid in participant_list],
                                value=participant_list[0] if participant_list else None,
                                style={"width": "250px"},
                                clearable=False,
                            ),
                        ]
                    ),
                ]
            ),
            
            # Two-column comparison
            html.Div(
                id='comparison-container',
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                },
                children=[
                    # LEFT: Felipe Data
                    html.Div(
                        style={
                            "backgroundColor": "rgba(26,26,26,0.5)",
                            "borderRadius": "12px",
                            "padding": "20px",
                            "border": "1px solid rgba(99,110,250,0.3)",
                        },
                        children=[
                            html.H3(
                                "Reference Dataset (Felipe)",
                                style={
                                    "color": "#636EFA",
                                    "fontSize": "18px",
                                    "fontWeight": "600",
                                    "marginBottom": "15px",
                                    "textAlign": "center",
                                }
                            ),
                            html.Div([dcc.Graph(id='felipe-pe-scatter', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            html.Div([dcc.Graph(id='felipe-trajectories', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            html.Div([dcc.Graph(id='felipe-distribution', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            html.Div([dcc.Graph(id='felipe-radar', config={'displayModeBar': False})], 
                                    style={"marginBottom": "0px"}),
                        ]
                    ),
                    
                    # RIGHT: Selected Participant
                    html.Div(
                        style={
                            "backgroundColor": "rgba(26,26,26,0.5)",
                            "borderRadius": "12px",
                            "padding": "20px",
                            "border": "1px solid rgba(239,85,59,0.3)",
                        },
                        children=[
                            html.H3(
                                id='participant-title',
                                style={
                                    "color": "#EF553B",
                                    "fontSize": "18px",
                                    "fontWeight": "600",
                                    "marginBottom": "15px",
                                    "textAlign": "center",
                                }
                            ),
                            html.Div([dcc.Graph(id='participant-pe-scatter', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            html.Div([dcc.Graph(id='participant-trajectories', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            html.Div([dcc.Graph(id='participant-distribution', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            html.Div([dcc.Graph(id='participant-radar', config={'displayModeBar': False})], 
                                    style={"marginBottom": "0px"}),
                        ]
                    ),
                ]
            ),
        ]
    )


def register_clean_comparison_callbacks(app):
    """Register callbacks for clean comparison."""
    
    # Load data
    felipe_fid, felipe_traj = load_felipe_data()
    
    try:
        loader = ArchetypeDataLoader("dash_data")
        has_data = True
    except:
        has_data = False
    
    if not has_data or felipe_fid is None:
        return
    
    # Prepare Felipe data once
    felipe_post_cols = ['P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 'P_immotile', 'P_erratic']
    
    @app.callback(
        [Output('participant-title', 'children'),
         Output('felipe-pe-scatter', 'figure'),
         Output('felipe-trajectories', 'figure'),
         Output('felipe-distribution', 'figure'),
         Output('felipe-radar', 'figure'),
         Output('participant-pe-scatter', 'figure'),
         Output('participant-trajectories', 'figure'),
         Output('participant-distribution', 'figure'),
         Output('participant-radar', 'figure')],
        Input('comparison-participant-dropdown', 'value'),
    )
    def update_comparison(participant_id):
        """Update all comparison plots."""
        
        if not participant_id:
            return ["No participant selected"] + [go.Figure()] * 8
        
        # Get participant data
        participant_tracks = loader.get_patient_tracks(participant_id)
        participant_frames = loader.get_patient_frames(participant_id)
        participant_summary = loader.get_patient_summary(participant_id)
        
        # === FELIPE DATA ===
        # P-E Scatter
        felipe_pe = create_pe_scatter(felipe_fid, "P-E Scatter")
        
        # Trajectories (keep original column names)
        felipe_traj_fig = create_clean_trajectories(
            felipe_traj,
            felipe_fid,
            "Sperm Trajectories (n=120)",
            n_display=120
        )
        
        # Distribution
        felipe_dist = create_cluster_distribution(felipe_fid, "Cluster Distribution")
        
        # Radar
        felipe_gmm_values = felipe_fid[felipe_post_cols].mean(axis=0).values.tolist()
        felipe_gmm_labels = [c.replace('P_', '').replace('_', ' ').title() for c in felipe_post_cols]
        felipe_radar = create_clean_radar(felipe_gmm_values, felipe_gmm_labels, "GMM Composition")
        
        # === PARTICIPANT DATA ===
        # P-E Scatter
        participant_pe = create_pe_scatter(participant_tracks, "P-E Scatter")
        
        # Trajectories
        participant_traj_fig = create_clean_trajectories(
            participant_frames,
            participant_tracks,
            "Sperm Trajectories (n=120)",
            n_display=120
        )
        
        # Distribution
        participant_dist = create_cluster_distribution(participant_tracks, "Cluster Distribution")
        
        # Radar
        participant_gmm_values = participant_tracks[felipe_post_cols].mean(axis=0).values.tolist()
        participant_radar = create_clean_radar(participant_gmm_values, felipe_gmm_labels, "GMM Composition")
        
        # Title
        title = f"Participant {participant_id}"
        
        return [
            title,
            felipe_pe,
            felipe_traj_fig,
            felipe_dist,
            felipe_radar,
            participant_pe,
            participant_traj_fig,
            participant_dist,
            participant_radar,
        ]
