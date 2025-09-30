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
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
        autosize=False,
        uirevision='constant',
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
    
    # Add traces for each trajectory - START WITH FULL TRAJECTORIES
    for tid, traj in trajectories.items():
        fig.add_trace(go.Scatter(
            x=traj['x'],  # Show full trajectory by default
            y=traj['y'],
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
                 'args': [None, {'frame': {'duration': 50, 'redraw': False},
                                'fromcurrent': True, 'mode': 'immediate',
                                'transition': {'duration': 0}}]},
                {'label': '⏸', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate',
                                  'transition': {'duration': 0}}]},
            ],
            'x': 0.5, 'y': -0.05, 'xanchor': 'center',
            'font': dict(size=16, color='white'),
            'bgcolor': 'rgba(99, 110, 250, 0.3)',
            'bordercolor': 'rgba(99, 110, 250, 0.5)',
        }],
        autosize=False,  # Prevent auto-resizing
        uirevision='constant',  # Prevent re-initialization on updates
    )
    
    return fig


def create_clean_radar(values, labels, title="", show_average=False, average_values=None, max_range=None):
    """Create clean radar with fewer rings."""
    
    # Close the loop
    theta = labels + [labels[0]]
    r = values + [values[0]]
    
    fig = go.Figure()
    
    # Add average patient trace first (grey, behind)
    if show_average and average_values:
        r_avg = average_values + [average_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=r_avg,
            theta=theta,
            fill='toself',
            fillcolor='rgba(150, 150, 150, 0.15)',
            line=dict(color='rgba(150, 150, 150, 0.6)', width=2, dash='dash'),
            hovertemplate='<b>Average Patient</b><br>%{theta}: %{r:.1%}<extra></extra>',
            name='Average Patient',
        ))
    
    # Add main data trace (on top)
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.3)',
        line=dict(color='#636EFA', width=2.5),
        hovertemplate='<b>%{theta}</b><br>%{r:.1%}<extra></extra>',
        name='Patient',
    ))
    
    # Determine range
    if max_range is not None:
        range_max = max_range
    else:
        max_val = max(values) if values else 1.0
        range_max = max(1.0, max_val * 1.1)
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,  # Hide radial axis completely
                range=[0, range_max],
                gridcolor='rgba(255,255,255,0.15)',
                nticks=3,  # FEWER RINGS
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='white'),
                gridcolor='rgba(255,255,255,0.15)',
            ),
            bgcolor='rgba(26,26,26,0.3)',
        ),
        showlegend=False,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color='white'), x=0.5),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=45, b=30),
        autosize=False,
        uirevision='constant',
    )
    
    return fig


def create_clean_comparison_section():
    """Create clean comparison section with dropdown."""
    
    # Load data
    felipe_fid, felipe_traj = load_felipe_data()
    
    try:
        # Debug: List current directory contents
        print(f"Current working directory: {Path.cwd()}")
        print(f"Contents of current directory: {list(Path.cwd().iterdir())}")
        print(f"Contents of app directory: {list(Path('app').iterdir()) if Path('app').exists() else 'app directory not found'}")
        
        # Try multiple possible paths for dash_data
        data_paths = ["dash_data", "app/dash_data", "./app/dash_data"]
        loader = None
        for data_path in data_paths:
            try:
                print(f"Trying to load data from: {data_path}")
                print(f"Path exists: {Path(data_path).exists()}")
                if Path(data_path).exists():
                    print(f"Contents of {data_path}: {list(Path(data_path).iterdir())}")
                loader = ArchetypeDataLoader(data_path)
                print(f"Successfully loaded data from: {data_path}")
                break
            except Exception as path_error:
                print(f"Failed to load from {data_path}: {path_error}")
                continue
        
        if loader is None:
            raise Exception("Could not find dash_data in any expected location")
            
        has_data = True
        participant_list = loader.patient_df['participant_id'].unique().tolist()
    except Exception as e:
        print(f"Error loading dash data: {e}")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Looking for dash_data in: {Path('dash_data').absolute()}")
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
                        "From Cells to Samples: Patient-Level Profiles",
                        style={
                            "color": "white",
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "marginBottom": "15px",
                        }
                    ),
                    html.P(
                        "By aggregating cell-level motility patterns, we create unique patient profiles that reveal the composition and quality of each semen sample. These profiles enable direct comparison between patients and reference populations, translating complex movement heterogeneity into clinically actionable insights.",
                        style={
                            "color": "#e0e0e0",
                            "fontSize": "15px",
                            "marginBottom": "20px",
                            "lineHeight": "1.6"
                        }
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
                                "Patient Sample A (high motility)",
                                style={
                                    "color": "#636EFA",
                                    "fontSize": "18px",
                                    "fontWeight": "600",
                                    "marginBottom": "15px",
                                    "textAlign": "center",
                                }
                            ),
                            # Top: Trajectories
                            html.Div([dcc.Graph(id='felipe-trajectories', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            # 2x2 grid: PE + CASA (top), GMM + paragraph (bottom)
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "15px",
                                    "marginBottom": "15px",
                                    "alignItems": "start"
                                },
                                children=[
                                    # Top-left: P-E scatter
                                    html.Div([
                                        dcc.Graph(id='felipe-pe-scatter', config={'displayModeBar': False}),
                                    ]),
                                    # Top-right: CASA radar
                                    html.Div([
                                        dcc.Graph(id='felipe-casa-radar', config={'displayModeBar': False}),
                                    ]),
                                    # Bottom-left: GMM radar
                                    html.Div([
                                        dcc.Graph(id='felipe-radar', config={'displayModeBar': False})
                                    ]),
                                    # Bottom-right: Paragraph element
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                        },
                                        children=[
                                            html.P(
                                                "This sample exhibits predominantly rapid progressive motility with high velocity (VCL) and lateral head movement (ALH), consistent with hyperactivated sperm. Kinematic metrics exceed the average patient baseline (grey line), suggesting strong fertilization potential. The motility distribution skews heavily toward progressive clusters, indicating a high-quality sample.",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "14px",
                                                    "lineHeight": "1.4",
                                                    "padding": "5px 15px",
                                                    "textAlign": "center",
                                                }
                                            )
                                        ]
                                    )
                                ]
                            ),
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
                            # Top: Trajectories
                            html.Div([dcc.Graph(id='participant-trajectories', config={'displayModeBar': False})], 
                                    style={"marginBottom": "15px"}),
                            # 2x2 grid: PE + CASA (top), GMM + paragraph (bottom)
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "15px",
                                    "marginBottom": "15px",
                                    "alignItems": "start"
                                },
                                children=[
                                    # Top-left: P-E scatter
                                    html.Div([
                                        dcc.Graph(id='participant-pe-scatter', config={'displayModeBar': False}),
                                    ]),
                                    # Top-right: CASA radar + description
                                    html.Div([
                                        dcc.Graph(id='participant-casa-radar', config={'displayModeBar': False}),
                                    ]),
                                    # Bottom-left: GMM radar
                                    html.Div([
                                        dcc.Graph(id='participant-radar', config={'displayModeBar': False})
                                    ]),
                                    # Bottom-right: Paragraph element
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                        },
                                        children=[
                                            html.P(
                                                "This sample shows reduced motility with lower velocity and linearity compared to both the reference and average patient. The distribution reveals a higher proportion of non-progressive and immotile sperm clusters. Kinematic metrics fall below clinical thresholds, suggesting potential challenges for natural fertilization and a candidate for assisted reproductive interventions.",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "14px",
                                                    "lineHeight": "1.4",
                                                    "padding": "5px 15px",
                                                    "textAlign": "center",
                                                }
                                            )
                                        ]
                                    )
                                ]
                            ),
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
        # Debug: List current directory contents
        print(f"CALLBACK - Current working directory: {Path.cwd()}")
        print(f"CALLBACK - Contents of current directory: {list(Path.cwd().iterdir())}")
        print(f"CALLBACK - Contents of app directory: {list(Path('app').iterdir()) if Path('app').exists() else 'app directory not found'}")
        
        # Try multiple possible paths for dash_data
        data_paths = ["dash_data", "app/dash_data", "./app/dash_data"]
        loader = None
        for data_path in data_paths:
            try:
                print(f"CALLBACK - Trying to load data from: {data_path}")
                print(f"CALLBACK - Path exists: {Path(data_path).exists()}")
                if Path(data_path).exists():
                    print(f"CALLBACK - Contents of {data_path}: {list(Path(data_path).iterdir())}")
                loader = ArchetypeDataLoader(data_path)
                print(f"CALLBACK - Successfully loaded data from: {data_path}")
                break
            except Exception as path_error:
                print(f"CALLBACK - Failed to load from {data_path}: {path_error}")
                continue
        
        if loader is None:
            raise Exception("Could not find dash_data in any expected location")
            
        has_data = True
    except Exception as e:
        print(f"CALLBACK - Error loading dash data in callbacks: {e}")
        print(f"CALLBACK - Current working directory: {Path.cwd()}")
        print(f"CALLBACK - Looking for dash_data in: {Path('dash_data').absolute()}")
        has_data = False
    
    if not has_data or felipe_fid is None:
        return
    
    # Prepare Felipe data once
    felipe_post_cols = ['P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 'P_immotile', 'P_erratic']
    
    # Create synthetic "Average Patient" data (between Felipe and participants)
    # CASA values (normalized to 0-1)
    synthetic_casa_avg = {
        'ALH': 0.35,   # Between Felipe (higher) and participant (lower)
        'VCL': 0.45,   # Medium velocity
        'LIN': 0.55,   # Medium linearity
        'VAP': 0.48,   # Medium VAP
        'VSL': 0.50,   # Medium VSL
        'WOB': 0.40,   # Between Felipe and participant
    }
    
    # GMM composition (typical clinical values)
    # Progressive motility: ~45-55% (split between rapid and progressive)
    # Non-progressive motility: ~5-15%
    # Immotile: ~30-40%
    synthetic_gmm_avg = {
        'P_progressive': 0.30,          # 30%
        'P_rapid_progressive': 0.20,    # 20% (total progressive = 50%)
        'P_nonprogressive': 0.10,       # 10%
        'P_immotile': 0.35,             # 35%
        'P_erratic': 0.05,              # 5%
    }
    
    @app.callback(
        [Output('participant-title', 'children'),
         Output('felipe-pe-scatter', 'figure'),
         Output('felipe-casa-radar', 'figure'),
         Output('felipe-trajectories', 'figure'),
         Output('felipe-radar', 'figure'),
         Output('participant-pe-scatter', 'figure'),
         Output('participant-casa-radar', 'figure'),
         Output('participant-trajectories', 'figure'),
         Output('participant-radar', 'figure')],
        Input('comparison-container', 'id'),  # Trigger on page load
    )
    def update_comparison(_):
        """Update all comparison plots."""
        
        # Hardcode participant ID (previously selected via dropdown)
        participant_id = 'b7f96273'
        
        # Get participant data
        participant_tracks = loader.get_patient_tracks(participant_id)
        participant_frames = loader.get_patient_frames(participant_id)
        participant_summary = loader.get_patient_summary(participant_id)
        
        # === FELIPE DATA ===
        # P-E Scatter
        felipe_pe = create_pe_scatter(felipe_fid, "P-E Scatter")
        
        # CASA Radar
        felipe_casa_cols = ['ALH', 'VCL', 'LIN', 'VAP', 'VSL', 'WOB']
        felipe_casa_values = []
        for col in felipe_casa_cols:
            if col in felipe_fid.columns:
                val = felipe_fid[col].mean()
                # Normalize to 0-1 range (rough normalization)
                if col in ['VCL', 'VAP', 'VSL']:  # Velocity metrics
                    normalized = val / 200.0  # Assume max ~200
                elif col == 'ALH':
                    normalized = val / 10.0  # Assume max ~10
                else:  # LIN, WOB are already 0-1
                    normalized = val
                felipe_casa_values.append(min(normalized, 1.0))
            else:
                felipe_casa_values.append(0.5)
        
        # Create average patient CASA values
        avg_casa_values = [synthetic_casa_avg.get(col, 0.5) for col in felipe_casa_cols]
        
        felipe_casa_radar = create_clean_radar(
            felipe_casa_values, 
            felipe_casa_cols, 
            "CASA Kinematics",
            show_average=True,
            average_values=avg_casa_values
        )
        
        # Trajectories (keep original column names)
        felipe_traj_fig = create_clean_trajectories(
            felipe_traj,
            felipe_fid,
            "Sperm Trajectories (n=120)",
            n_display=120
        )
        
        # GMM Radar
        felipe_gmm_values = felipe_fid[felipe_post_cols].mean(axis=0).values.tolist()
        felipe_gmm_labels = [c.replace('P_', '').replace('_', ' ').title() for c in felipe_post_cols]
        
        # Create average patient GMM values
        avg_gmm_values = [synthetic_gmm_avg.get(col, 0.2) for col in felipe_post_cols]
        
        felipe_radar = create_clean_radar(
            felipe_gmm_values, 
            felipe_gmm_labels, 
            "GMM Composition",
            show_average=True,
            average_values=avg_gmm_values,
            max_range=0.6  # Set max to 60% so radar fills more of circle
        )
        
        # === PARTICIPANT DATA ===
        # P-E Scatter
        participant_pe = create_pe_scatter(participant_tracks, "P-E Scatter")
        
        # CASA Radar  
        participant_casa_values = []
        for col in felipe_casa_cols:
            if col in participant_tracks.columns:
                val = participant_tracks[col].mean()
                # Normalize to 0-1 range
                if col in ['VCL', 'VAP', 'VSL']:
                    normalized = val / 200.0
                elif col == 'ALH':
                    normalized = val / 10.0
                else:
                    normalized = val
                participant_casa_values.append(min(normalized, 1.0))
            else:
                participant_casa_values.append(0.5)
        
        participant_casa_radar = create_clean_radar(
            participant_casa_values, 
            felipe_casa_cols, 
            "CASA Kinematics",
            show_average=True,
            average_values=avg_casa_values
        )
        
        # Trajectories
        participant_traj_fig = create_clean_trajectories(
            participant_frames,
            participant_tracks,
            "Sperm Trajectories (n=120)",
            n_display=120
        )
        
        # GMM Radar
        participant_gmm_values = participant_tracks[felipe_post_cols].mean(axis=0).values.tolist()
        participant_radar = create_clean_radar(
            participant_gmm_values, 
            felipe_gmm_labels, 
            "GMM Composition",
            show_average=True,
            average_values=avg_gmm_values,
            max_range=0.6  # Set max to 60% so radar fills more of circle
        )
        
        # Title
        title = f"Patient Sample B (Lower Motility))"
        
        return [
            title,
            felipe_pe,
            felipe_casa_radar,
            felipe_traj_fig,
            felipe_radar,
            participant_pe,
            participant_casa_radar,
            participant_traj_fig,
            participant_radar,
        ]
