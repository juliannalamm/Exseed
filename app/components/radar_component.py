"""
Beautiful radar chart component for displaying patient archetype fingerprints.
Shows both GMM composition and CASA metrics with modern styling.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def create_beautiful_radar(
    values: List[float],
    labels: List[str],
    title: str = "",
    color: str = "#636EFA",
    fill_opacity: float = 0.3,
    line_width: int = 3,
    show_scale: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Create a single beautiful radar chart.
    
    Parameters
    ----------
    values : List[float]
        Values for each axis (should be 0-1 normalized)
    labels : List[str]
        Labels for each axis
    title : str
        Chart title
    color : str
        Line and fill color
    fill_opacity : float
        Opacity of the filled area
    line_width : int
        Width of the radar line
    show_scale : bool
        Whether to show the radial scale
    height : int
        Chart height in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure with radar chart
    """
    
    # Close the loop
    theta = labels + [labels[0]]
    r = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        fill='toself',
        fillcolor=color,
        opacity=fill_opacity,
        line=dict(
            color=color,
            width=line_width,
        ),
        hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>',
        name=title
    ))
    
    # Determine appropriate range based on data
    max_val = max(values) if values else 1.0
    range_max = max(1.0, max_val * 1.15)  # At least 1.0, or 15% above max value
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=show_scale,
                range=[0, range_max],
                tickfont=dict(size=11, color='#e0e0e0'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                showticklabels=show_scale,
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color='white', family='Arial Black'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.3)',
            ),
            bgcolor='rgba(0, 0, 0, 0.3)',
        ),
        showlegend=False,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color='white', family='Arial'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        height=height,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=80, r=80, t=80, b=80),
    )
    
    return fig


def create_dual_radar(
    gmm_values: List[float],
    gmm_labels: List[str],
    casa_values: List[float],
    casa_labels: List[str],
    archetype_title: str = "Patient Archetype",
    participant_id: str = "",
) -> go.Figure:
    """
    Create a dual radar chart showing GMM composition and CASA metrics side by side.
    
    Parameters
    ----------
    gmm_values : List[float]
        GMM cluster composition values (0-1)
    gmm_labels : List[str]
        GMM cluster labels
    casa_values : List[float]
        CASA metric values (0-1 normalized)
    casa_labels : List[str]
        CASA metric labels
    archetype_title : str
        Title for the archetype
    participant_id : str
        Participant ID
        
    Returns
    -------
    go.Figure
        Plotly figure with dual radar charts
    """
    
    # Create subplots with polar charts
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=(
            "<b>GMM Composition</b>",
            "<b>CASA Metrics</b>"
        ),
        horizontal_spacing=0.15,
    )
    
    # GMM radar (left)
    theta_gmm = gmm_labels + [gmm_labels[0]]
    r_gmm = gmm_values + [gmm_values[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=r_gmm,
        theta=theta_gmm,
        fill='toself',
        fillcolor='#636EFA',
        opacity=0.4,
        line=dict(color='#636EFA', width=3),
        hovertemplate='<b>%{theta}</b><br>Proportion: %{r:.1%}<extra></extra>',
        name='GMM'
    ), row=1, col=1)
    
    # CASA radar (right)
    theta_casa = casa_labels + [casa_labels[0]]
    r_casa = casa_values + [casa_values[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=r_casa,
        theta=theta_casa,
        fill='toself',
        fillcolor='#EF553B',
        opacity=0.4,
        line=dict(color='#EF553B', width=3),
        hovertemplate='<b>%{theta}</b><br>Normalized: %{r:.0%}<extra></extra>',
        name='CASA'
    ), row=1, col=2)
    
    # Update polar axes styling
    for i in range(1, 3):
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                tickfont=dict(size=10, color='#e0e0e0'),
                gridcolor='rgba(255, 255, 255, 0.15)',
                showline=True,
                linecolor='rgba(255, 255, 255, 0.2)',
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='white', family='Arial'),
                gridcolor='rgba(255, 255, 255, 0.15)',
                linecolor='rgba(255, 255, 255, 0.25)',
            ),
            bgcolor='rgba(26, 26, 26, 0.5)',
            row=1, col=i
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{archetype_title}</b><br><sub>Participant: {participant_id}</sub>",
            font=dict(size=20, color='white', family='Arial'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        showlegend=False,
        height=450,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=40, r=40, t=100, b=40),
        annotations=[
            dict(
                text="<b>GMM Composition</b>",
                x=0.2,
                y=1.05,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=14, color='#636EFA', family='Arial'),
            ),
            dict(
                text="<b>CASA Metrics</b>",
                x=0.8,
                y=1.05,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=14, color='#EF553B', family='Arial'),
            ),
        ]
    )
    
    return fig


def create_archetype_radar_grid(
    archetypes_data: List[Dict],
    cols: int = 2,
) -> go.Figure:
    """
    Create a grid of radar charts for multiple archetypes.
    
    Parameters
    ----------
    archetypes_data : List[Dict]
        List of dicts, each containing:
        - title: str
        - participant_id: str
        - gmm_values: List[float]
        - gmm_labels: List[str]
        - casa_values: List[float]
        - casa_labels: List[str]
    cols : int
        Number of columns in the grid
        
    Returns
    -------
    go.Figure
        Plotly figure with radar chart grid
    """
    
    n_archetypes = len(archetypes_data)
    rows = int(np.ceil(n_archetypes / cols))
    
    # Create subplot specifications
    specs = []
    for _ in range(rows):
        row_specs = []
        for _ in range(cols):
            row_specs.append({"type": "polar"})
        specs.append(row_specs)
    
    subplot_titles = [f"<b>{arch['title']}</b>" for arch in archetypes_data]
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )
    
    # Define beautiful colors for each archetype
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    for idx, arch_data in enumerate(archetypes_data):
        row = idx // cols + 1
        col = idx % cols + 1
        color = colors[idx % len(colors)]
        
        # Use GMM data for the radar
        theta = arch_data['gmm_labels'] + [arch_data['gmm_labels'][0]]
        r = arch_data['gmm_values'] + [arch_data['gmm_values'][0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            fillcolor=color,
            opacity=0.4,
            line=dict(color=color, width=3),
            hovertemplate=f"<b>{arch_data['title']}</b><br>%{{theta}}: %{{r:.1%}}<extra></extra>",
            name=arch_data['title']
        ), row=row, col=col)
        
        # Update polar styling for this subplot
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                tickfont=dict(size=9, color='#e0e0e0'),
                gridcolor='rgba(255, 255, 255, 0.15)',
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='white', family='Arial'),
                gridcolor='rgba(255, 255, 255, 0.15)',
            ),
            bgcolor='rgba(26, 26, 26, 0.5)',
            row=row, col=col
        )
    
    fig.update_layout(
        showlegend=False,
        height=450 * rows,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    # Update subplot title styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='white', family='Arial')
    
    return fig
