"""
Component to display patient fingerprint plots similar to plot_patient_fingerprint_compact_with_casa.

This recreates the 2x2 grid layout with:
- Top-left: P-E scatter
- Top-right: GMM composition radar
- Bottom-left: Trajectory grid
- Bottom-right: CASA metrics radar
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def create_archetype_fingerprint(
    tracks_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    patient_summary: pd.Series,
    casa_scales: Dict[str, Tuple[float, float]],
    archetype_title: str = "Patient Fingerprint",
    radar_order: Tuple[str, ...] = ('progressive', 'rapid_progressive', 'nonprogressive', 'immotile', 'erratic'),
    selected_track_ids: List[str] = None,
    n_tracks_display: int = 120,
    invert_y: bool = True,
) -> go.Figure:
    """
    Create a patient fingerprint visualization.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Per-track data for this patient (already filtered)
    frames_df : pd.DataFrame
        Frame data for this patient (already filtered)
    patient_summary : pd.Series
        Patient-level summary data
    casa_scales : Dict[str, Tuple[float, float]]
        CASA metric scales for normalization
    archetype_title : str
        Title for the plot
    radar_order : tuple
        Order of GMM clusters for radar plot
    selected_track_ids : List[str], optional
        Pre-selected track IDs to display. If None, will sample randomly
    n_tracks_display : int
        Number of trajectories to display (if selected_track_ids is None)
    invert_y : bool
        Whether to invert Y axis for trajectories
        
    Returns
    -------
    plotly.graph_objects.Figure
        Combined figure with 4 subplots
    """
    
    # Create subplot layout: 2 rows, 2 cols
    # Note: Plotly doesn't support polar in subplots easily, so we'll use separate figures
    # For a full implementation, you might want to use Dash with multiple graph components
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Progressivity vs Erraticity",
            "GMM Composition",
            "Sample Trajectories",
            "CASA Metrics"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "polar"}],
            [{"type": "scatter"}, {"type": "polar"}]
        ],
        row_heights=[0.5, 0.5],
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )
    
    # ========================================================================
    # 1. Top-left: P-E scatter
    # ========================================================================
    if 'progressivity' in tracks_df.columns and 'erraticity' in tracks_df.columns:
        xs = tracks_df['progressivity'].values
        ys = tracks_df['erraticity'].values
        cs = tracks_df.get('entropy', np.zeros(len(tracks_df))).values
        
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(
                    size=4,
                    color=cs,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Entropy", x=0.45, len=0.4, y=0.75)
                ),
                name='Tracks',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Progressivity (P)", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Erraticity (E)", row=1, col=1, range=[0, 1])
    
    # ========================================================================
    # 2. Top-right: GMM composition radar
    # ========================================================================
    post_cols = [f'P_{c}' for c in radar_order]
    available_posts = [c for c in post_cols if c in tracks_df.columns]
    
    if available_posts:
        composition = tracks_df[available_posts].mean(axis=0).values
        labels = [c.replace('P_', '').replace('_', ' ').title() for c in available_posts]
        
        # Close the radar loop
        theta = labels + [labels[0]]
        r = list(composition) + [composition[0]]
        
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                name='Composition',
                line=dict(color='#1f77b4', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_polars(
            radialaxis=dict(range=[0, 1], tickformat='.1%'),
            row=1, col=2
        )
    
    # ========================================================================
    # 3. Bottom-left: Sample trajectories (simplified - show a few)
    # ========================================================================
    if selected_track_ids is None:
        # Sample random tracks
        unique_tids = tracks_df['track_id'].unique()
        n_sample = min(n_tracks_display, len(unique_tids))
        selected_track_ids = np.random.choice(unique_tids, size=n_sample, replace=False)
    
    # For visualization, show just a subset to avoid overcrowding
    display_tids = selected_track_ids[:min(20, len(selected_track_ids))]
    
    for tid in display_tids:
        traj = frames_df[frames_df['track_id'] == tid].sort_values('frame_num')
        if traj.empty:
            continue
        
        # Center trajectory at origin
        x = traj['x'].values - traj['x'].iloc[0]
        y = traj['y'].values - traj['y'].iloc[0]
        
        if invert_y:
            y = -y
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(width=1, color='gray'),
                opacity=0.5,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Δx (μm)", row=2, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Δy (μm)", row=2, col=1)
    
    # ========================================================================
    # 4. Bottom-right: CASA radar
    # ========================================================================
    casa_cols_full = sorted([c for c in patient_summary.index if c.startswith('CASA_') and c.endswith('_mean')])
    
    if casa_cols_full:
        casa_vals = [float(patient_summary[c]) if c in patient_summary.index else np.nan 
                     for c in casa_cols_full]
        casa_labels = [c.replace('CASA_', '').replace('_mean', '') for c in casa_cols_full]
        
        # Normalize using scales
        normalized_vals = []
        for label, val in zip(casa_labels, casa_vals):
            if label in casa_scales and np.isfinite(val):
                lo, hi = casa_scales[label]
                normalized = (val - lo) / (hi - lo + 1e-12)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = 0
            normalized_vals.append(normalized)
        
        # Close the radar loop
        theta_casa = casa_labels + [casa_labels[0]]
        r_casa = normalized_vals + [normalized_vals[0]]
        
        fig.add_trace(
            go.Scatterpolar(
                r=r_casa,
                theta=theta_casa,
                fill='toself',
                name='CASA',
                line=dict(color='#ff7f0e', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_polars(
            radialaxis=dict(range=[0, 1], tickformat='.0%'),
            row=2, col=2
        )
    
    # ========================================================================
    # Overall layout
    # ========================================================================
    fig.update_layout(
        title=dict(
            text=archetype_title,
            font=dict(size=20),
            x=0.5,
            xanchor='center'
        ),
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_simple_trajectory_grid(
    frames_df: pd.DataFrame,
    track_ids: List[str],
    title: str = "Trajectories",
    n_cols: int = 10,
    invert_y: bool = True,
) -> go.Figure:
    """
    Create a grid of trajectory plots (alternative to embedding in main figure).
    
    Parameters
    ----------
    frames_df : pd.DataFrame
        Frame data (already filtered to one participant)
    track_ids : List[str]
        List of track IDs to display
    title : str
        Plot title
    n_cols : int
        Number of columns in the grid
    invert_y : bool
        Whether to invert Y axis
        
    Returns
    -------
    plotly.graph_objects.Figure
        Grid of trajectory subplots
    """
    n_tracks = len(track_ids)
    n_rows = int(np.ceil(n_tracks / n_cols))
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"T{i+1}" for i in range(n_tracks)],
        horizontal_spacing=0.02,
        vertical_spacing=0.03
    )
    
    for idx, tid in enumerate(track_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        traj = frames_df[frames_df['track_id'] == tid].sort_values('frame_num')
        if traj.empty:
            continue
        
        # Center at origin
        x = traj['x'].values - traj['x'].iloc[0]
        y = traj['y'].values - traj['y'].iloc[0]
        
        if invert_y:
            y = -y
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                line=dict(width=1.5, color='steelblue'),
                marker=dict(size=3),
                showlegend=False,
                hovertemplate='<b>Track %s</b><br>x: %%{x:.1f}<br>y: %%{y:.1f}<extra></extra>' % tid
            ),
            row=row, col=col
        )
        
        # Mark start (green) and end (red)
        fig.add_trace(
            go.Scatter(
                x=[x[0]], y=[y[0]],
                mode='markers',
                marker=dict(size=6, color='green'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=[x[-1]], y=[y[-1]],
                mode='markers',
                marker=dict(size=6, color='red'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # Equal aspect ratio
        fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor=f"x{idx+1}", row=row, col=col)
    
    fig.update_layout(
        title=title,
        height=150 * n_rows,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig
