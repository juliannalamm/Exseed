#!/usr/bin/env python3
"""
Utility functions for creating global UMAP plots with training data and highlighted new data.
"""

import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from sklearn.preprocessing import MinMaxScaler
from full_pipeline import json_to_df, MODEL_PATH

def load_or_create_training_umap_data():
    """
    Load real training data with UMAP coordinates.
    Returns training data with UMAP coordinates.
    """
    # Check if we have the real training data file
    training_cache_file = "training_data_with_umap.pkl"
    
    if os.path.exists(training_cache_file):
        print("Loading real training data with UMAP coordinates...")
        with open(training_cache_file, 'rb') as f:
            training_data = pickle.load(f)
        return training_data
    
    # If no real training data, try to create it
    print("Real training data not found. Please run extract_training_data.py first.")
    print("This will extract the real training data from your combined_views_tracks directory.")
    return None

def create_global_umap_plot(new_data_df, training_data_df=None):
    """
    Create a global UMAP plot showing training data with highlighted new data points.
    
    Args:
        new_data_df: DataFrame with new participant data (must have umap_1, umap_2, cluster_id, subtype_label)
        training_data_df: DataFrame with training data (if None, will be loaded/created)
    
    Returns:
        plotly figure object
    """
    if training_data_df is None:
        training_data_df = load_or_create_training_umap_data()
    
    if training_data_df is None:
        print("Could not load training data. Creating plot with new data only.")
        return create_new_data_only_plot(new_data_df)
    
    # Create the plot
    fig = go.Figure()
    
    # Color mapping for clusters
    cluster_colors = {
        0: '#1f77b4',  # blue
        1: '#ff7f0e',  # orange  
        2: '#2ca02c',  # green
        3: '#d62728',  # red
    }
    
    # Add training data points (smaller, more transparent)
    for cluster_id in sorted(training_data_df['cluster_id'].unique()):
        cluster_data = training_data_df[training_data_df['cluster_id'] == cluster_id]
        subtype = cluster_data['subtype_label'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=cluster_data['umap_1'],
            y=cluster_data['umap_2'],
            mode='markers',
            marker=dict(
                size=4,
                color=cluster_colors[cluster_id],
                opacity=0.3
            ),
            name=f'Training: {subtype}',
            showlegend=True,
            hovertemplate='<b>Training Data</b><br>' +
                         f'Subtype: {subtype}<br>' +
                         'Cluster: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=cluster_data['cluster_id']
        ))
    
    # Add new data points (larger, more prominent)
    for cluster_id in sorted(new_data_df['cluster_id'].unique()):
        cluster_data = new_data_df[new_data_df['cluster_id'] == cluster_id]
        subtype = cluster_data['subtype_label'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=cluster_data['umap_1'],
            y=cluster_data['umap_2'],
            mode='markers',
            marker=dict(
                size=8,
                color=cluster_colors[cluster_id],
                opacity=0.8,
                line=dict(color='black', width=1)
            ),
            name=f'New: {subtype}',
            showlegend=True,
            hovertemplate='<b>New Data</b><br>' +
                         'Track: %{text}<br>' +
                         f'Subtype: {subtype}<br>' +
                         'Cluster: %{customdata}<br>' +
                         '<extra></extra>',
            text=cluster_data['track_id'],
            customdata=cluster_data['cluster_id']
        ))
    
    # Update layout
    fig.update_layout(
        title="Global UMAP: Training Data + New Participant",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=800,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_new_data_only_plot(new_data_df):
    """
    Create a UMAP plot with only the new data (fallback when training data unavailable).
    """
    fig = px.scatter(
        new_data_df,
        x='umap_1',
        y='umap_2',
        color='subtype_label',
        hover_data=['track_id'],
        title="UMAP Projection (New Data Only)",
        width=800,
        height=600
    )
    
    return fig

def get_global_umap_comparison(new_data_df):
    """
    Main function to get global UMAP comparison.
    Returns both the global plot and summary statistics.
    """
    training_data = load_or_create_training_umap_data()
    
    if training_data is not None:
        # Create global plot
        global_fig = create_global_umap_plot(new_data_df, training_data)
        
        # Calculate summary statistics
        training_summary = training_data['subtype_label'].value_counts()
        new_summary = new_data_df['subtype_label'].value_counts()
        
        comparison_stats = {
            'training_distribution': training_summary,
            'new_data_distribution': new_summary,
            'training_total': len(training_data),
            'new_data_total': len(new_data_df)
        }
        
        return global_fig, comparison_stats
    else:
        # Fallback to new data only
        fig = create_new_data_only_plot(new_data_df)
        return fig, None 