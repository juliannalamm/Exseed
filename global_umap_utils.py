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

# Color mapping for clusters - shared across all functions
cluster_colors = {
    0: '#1f77b4',  # blue
    1: '#d62728',  # red  
    2: '#2ca02c',  # green
    3: '#ff7f0e',  # orange
}

def load_or_create_training_umap_data():
    """
    Load real training data with UMAP coordinates.
    Returns training data with UMAP coordinates.
    """
    # Preferred source: CSV provided by the user
    csv_path = "train_track_df.csv"
    if os.path.exists(csv_path):
        print("Loading training data with UMAP coordinates from train_track_df.csv...")
        training_data = pd.read_csv(csv_path)
        # Ensure required columns
        required_cols = {'umap_1','umap_2','cluster_id','subtype_label','track_id'}
        missing = required_cols - set(training_data.columns)
        if missing:
            print(f"Warning: train_track_df.csv missing columns: {missing}. Global plot may be limited.")
       
        return training_data
    
    print("Training data file not found. Expected train_track_df.csv in project root.")
    return None

def create_global_umap_plot(training_data_df, highlight_track_ids=None):
    """
    Create a global UMAP plot showing training data with optional highlighted points
    belonging to a specific participant or a provided set of track_ids.
    
    Args:
        training_data_df: DataFrame with training data (must have umap_1, umap_2, cluster_id, subtype_label)
        highlight_track_ids: Optional set/list of track_ids to highlight on the map
    
    Returns:
        plotly figure object
    """
    if training_data_df is None or training_data_df.empty:
        print("Could not load training data. Cannot create global plot.")
        return go.Figure()
    
    # Create the plot
    fig = go.Figure()
    
    # Add training data points with opacity based on confidence
    for cluster_id in sorted(training_data_df['cluster_id'].unique()):
        cluster_data = training_data_df[training_data_df['cluster_id'] == cluster_id]
        subtype = cluster_data['subtype_label'].iloc[0]
        
        # Get the probability of the assigned cluster for each track
        opacity_values = []
        for _, track in cluster_data.iterrows():
            # Get the probability of the cluster this track was assigned to
            assigned_subtype = track['subtype_label']
            prob_col = f'P_{assigned_subtype}'
            
            if prob_col in track:
                prob_value = track[prob_col]
                # These are standard 0-1 probabilities, so we can use them directly
                # Scale opacity: 0.4 to 1.0 (minimum opacity for visibility)
                opacity = 0.4 + (prob_value * 0.6)
            else:
                opacity = 0.8  # Fallback
            opacity_values.append(opacity)
        
        opacity_values = np.array(opacity_values)
        
        fig.add_trace(go.Scatter(
            x=cluster_data['umap_1'],
            y=cluster_data['umap_2'],
            mode='markers',
            marker=dict(
                size=4,
                color=cluster_colors.get(cluster_id, '#888'),
                opacity=opacity_values
            ),
            name=f' {subtype}',
            showlegend=True,
            hovertemplate='<b>Training Data</b><br>' +
                         f'Subtype: {subtype}<br>' +
                         'Cluster: %{customdata}<br>' +
                         'Confidence: %{marker.opacity:.2f}<br>' +
                         'VCL: %{text[0]:.1f}<br>' +
                         'ALH: %{text[1]:.2f}<br>' +
                         'VSL: %{text[2]:.1f}<br>' +
                         'VAP: %{text[3]:.1f}<br>' +
                         '<extra></extra>',
            customdata=cluster_data['cluster_id'],
            text=cluster_data[['VCL', 'ALH', 'VSL', 'VAP']].values
        ))
    
    # Highlight specific tracks, if provided
    if highlight_track_ids:
        highlight_set = set(highlight_track_ids)
        highlight_df = training_data_df[training_data_df['track_id'].isin(highlight_set)]
        if not highlight_df.empty:
            for cluster_id in sorted(highlight_df['cluster_id'].unique()):
                sub = highlight_df[highlight_df['cluster_id'] == cluster_id]
                subtype = sub['subtype_label'].iloc[0]
                
                # Get the probability of the assigned cluster for highlighted tracks
                opacity_values = []
                for _, track in sub.iterrows():
                    # Get the probability of the cluster this track was assigned to
                    assigned_subtype = track['subtype_label']
                    prob_col = f'P_{assigned_subtype}'
                    
                    if prob_col in track:
                        prob_value = track[prob_col]
                        # These are standard 0-1 probabilities, so we can use them directly
                        # Scale opacity: 0.6 to 1.0 for highlighted tracks
                        opacity = 0.6 + (prob_value * 0.4)
                    else:
                        opacity = 1.0  # Fallback
                    opacity_values.append(opacity)
                
                opacity_values = np.array(opacity_values)
                
                fig.add_trace(go.Scatter(
                    x=sub['umap_1'],
                    y=sub['umap_2'],
                    mode='markers',
                    marker=dict(
                        size=25,
                        color=cluster_colors.get(cluster_id, '#444'),
                        opacity=opacity_values,
                        line=dict(color='red', width=5),
                        symbol='diamond'
                    ),
                    name=f'Your Participant: {subtype}',
                    showlegend=True,
                    hovertemplate='<b>Your Participant</b><br>' +
                                 'Track: %{text}<br>' +
                                 f'Subtype: {subtype}<br>' +
                                 'Cluster: %{customdata}<br>' +
                                 'Confidence: %{marker.opacity:.2f}<br>' +
                                 'VCL: %{customdata2[0]:.1f}<br>' +
                                 'ALH: %{customdata2[1]:.2f}<br>' +
                                 'VSL: %{customdata2[2]:.1f}<br>' +
                                 'VAP: %{customdata2[3]:.1f}<br>' +
                                 '<extra></extra>',
                    text=sub['track_id'],
                    customdata=sub['cluster_id'],
                    customdata2=sub[['VCL', 'ALH', 'VSL', 'VAP']].values
                ))
    
    # Update layout
    title = "Global UMAP: Training Data"
    if highlight_track_ids:
        title += " (highlighted tracks in red outline)"
    
    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=800,
        height=600,
        legend=dict(
            yanchor="bottom",
            y=0,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig



def create_single_feature_plot(training_data_df, feature):
    """
    Create a single box plot for one feature
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Get data for each cluster
    for cluster_id in sorted(training_data_df['cluster_id'].unique()):
        cluster_data = training_data_df[training_data_df['cluster_id'] == cluster_id]
        subtype = cluster_data['subtype_label'].iloc[0]
        values = cluster_data[feature].dropna()
        
        if len(values) > 0:
            fig.add_trace(
                go.Box(
                    y=values,
                    name=f"{subtype} (n={len(values)})",
                    marker_color=cluster_colors[cluster_id],
                    opacity=0.7,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f"{feature} Distribution by Cluster",
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_yaxes(title_text=feature)
    
    return fig



def calculate_feature_statistics(training_data_df, features):
    """
    Calculate feature statistics for each cluster
    """
    stats = {}
    
    for cluster_id in training_data_df['cluster_id'].unique():
        cluster_data = training_data_df[training_data_df['cluster_id'] == cluster_id]
        subtype = cluster_data['subtype_label'].iloc[0]
        
        cluster_stats = {}
        for feature in features:
            values = cluster_data[feature].dropna()
            if len(values) > 0:
                cluster_stats[feature] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'count': len(values)
                }
        
        stats[subtype] = cluster_stats
    
    return stats



def suggest_gmm_based_cutoffs(training_data_df, features):
    """
    Suggest cutoff ranges based on GMM model decision boundaries
    """
    import numpy as np
    import pickle
    from sklearn.mixture import GaussianMixture
    
    cutoffs = {}
    
    # Load the trained GMM model
    try:
        with open("trained_gmm_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            gmm_model = model_data['gmm']
            scaler = model_data['scaler']
            print("✅ Successfully loaded GMM model")
    except Exception as e:
        print(f"⚠️ Warning: Could not load GMM model ({e})")
        return {}
    
    for feature in features:
        feature_cutoffs = {}
        
        # Get feature data for all clusters
        cluster_means = {}
        cluster_stds = {}
        
        for cluster_id in training_data_df['cluster_id'].unique():
            cluster_data = training_data_df[training_data_df['cluster_id'] == cluster_id]
            subtype = cluster_data['subtype_label'].iloc[0]
            values = cluster_data[feature].dropna()
            
            if len(values) > 0:
                cluster_means[subtype] = values.mean()
                cluster_stds[subtype] = values.std()
        
        # Sort clusters by mean value
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
        
        # Calculate GMM-based cutoffs between adjacent clusters
        for i in range(len(sorted_clusters) - 1):
            current_subtype, current_mean = sorted_clusters[i]
            next_subtype, next_mean = sorted_clusters[i + 1]
            
            # For GMM, we can use the decision boundary where two Gaussians intersect
            # This is approximately where the means are weighted by their variances
            current_std = cluster_stds[current_subtype]
            next_std = cluster_stds[next_subtype]
            
            # Weighted average based on cluster variances (inverse variance weighting)
            if current_std > 0 and next_std > 0:
                weight1 = 1 / (current_std ** 2)
                weight2 = 1 / (next_std ** 2)
                total_weight = weight1 + weight2
                
                cutoff_point = (current_mean * weight1 + next_mean * weight2) / total_weight
                
                # Debug: Show the calculation
                simple_midpoint = (current_mean + next_mean) / 2
                print(f"  {feature}: {current_subtype}(mean={current_mean:.3f}, std={current_std:.3f}) vs {next_subtype}(mean={next_mean:.3f}, std={next_std:.3f})")
                print(f"    Simple: {simple_midpoint:.3f}, GMM-weighted: {cutoff_point:.3f}, Diff: {cutoff_point-simple_midpoint:.3f}")
            else:
                # Fallback to simple midpoint
                cutoff_point = (current_mean + next_mean) / 2
                print(f"  {feature}: Using simple midpoint for {current_subtype} to {next_subtype}")
            
            feature_cutoffs[f"{current_subtype}_to_{next_subtype}"] = cutoff_point
        
        cutoffs[feature] = feature_cutoffs
    
    return cutoffs

def get_global_umap_comparison(new_data_df, participant_id=None):
    """
    Main function to get global UMAP comparison.
    Returns both the global plot and summary statistics.
    If participant_id is provided, we highlight the training points that match the
    `track_id`s of this participant in the cached training dataframe.
    """
    training_data = load_or_create_training_umap_data()
    
    if training_data is not None:
        highlight_ids = None
        if participant_id is not None:
            # Find tracks in training data that belong to this participant
            # Training track_ids are stored as e.g., "<participant>_track_<n>"
            if 'participant_id' in training_data.columns:
                # Direct match if participant_id column exists
                highlight_ids = training_data.loc[
                    training_data['participant_id'] == participant_id, 'track_id'
                ].unique().tolist()
            elif 'track_id' in training_data.columns:
                # Extract participant from track_id if participant_id column doesn't exist
                highlight_ids = training_data.loc[
                    training_data['track_id'].str.startswith(f"{participant_id}_"), 'track_id'
                ].unique().tolist()
                # print(f"Debug: Looking for tracks starting with '{participant_id}_'")
                # print(f"Debug: Sample track IDs in training data: {training_data['track_id'].head().tolist()}")
            
            print(f"Found {len(highlight_ids)} tracks for participant {participant_id}: {highlight_ids[:5]}...")
        
        # Create global plot (no highlighting when participant_id is None)
        global_fig = create_global_umap_plot(training_data, highlight_track_ids=highlight_ids)
        
        # Calculate summary statistics
        training_summary = training_data['subtype_label'].value_counts()
        
        # If new_data_df is provided, use it; otherwise, calculate from highlighted participant
        if new_data_df is not None and len(new_data_df) > 0:
            new_summary = new_data_df['subtype_label'].value_counts()
            new_data_total = len(new_data_df)
        else:
            # Calculate distribution from the highlighted participant's tracks in training data
            if highlight_ids and len(highlight_ids) > 0:
                participant_data = training_data[training_data['track_id'].isin(highlight_ids)]
                new_summary = participant_data['subtype_label'].value_counts()
                new_data_total = len(participant_data)
                print(f"Debug: Found {len(participant_data)} tracks for {participant_id}")
                print(f"Debug: Track IDs: {participant_data['track_id'].tolist()}")
                print(f"Debug: Subtype distribution: {new_summary.to_dict()}")
            else:
                # No participant selected or no tracks found
                new_summary = pd.Series(dtype='object')
                new_data_total = 0
                print(f"Debug: No tracks found for participant {participant_id}")
                print(f"Debug: Available participant IDs: {training_data['participant_id'].unique() if 'participant_id' in training_data.columns else 'No participant_id column'}")
        
        comparison_stats = {
            'training_distribution': training_summary,
            'new_data_distribution': new_summary,
            'training_total': len(training_data),
            'new_data_total': new_data_total
        }
        
        return global_fig, comparison_stats
    else:
        # No training data available
        print("No training data available for global comparison")
        return go.Figure(), None

def get_feature_analysis(training_data_df):
    """
    Get feature statistics and cutoff suggestions with clean formatting for Streamlit
    """
    # Define features
    features = ['ALH', 'BCF', 'LIN', 'MAD', 'STR', 'VAP', 'VCL', 'VSL', 'WOB']
    
    # Calculate feature statistics
    feature_stats = calculate_feature_statistics(training_data_df, features)
    
    # Get GMM-based cutoffs
    gmm_cutoffs = suggest_gmm_based_cutoffs(training_data_df, features)
    
    # Create a clean format with pre-formatted ranges for each subtype
    clean_cutoffs = {}
    
    for feature in features:
        if feature in gmm_cutoffs:
            feature_cutoffs = gmm_cutoffs[feature]
            clean_cutoffs[feature] = {}
            
            # Map the cutoff values to ranges for each subtype based on feature
            if feature in ['VCL', 'ALH', 'VSL', 'STR', 'VAP']:
                # Standard order: immotile -> nonprogressive -> progressive -> vigorous
                if 'immotile_to_nonprogressive' in feature_cutoffs and 'nonprogressive_to_progressive' in feature_cutoffs and 'progressive_to_vigorous' in feature_cutoffs:
                    clean_cutoffs[feature]['immotile'] = f"< {feature_cutoffs['immotile_to_nonprogressive']:.2f}"
                    clean_cutoffs[feature]['nonprogressive'] = f"{feature_cutoffs['immotile_to_nonprogressive']:.2f} - {feature_cutoffs['nonprogressive_to_progressive']:.2f}"
                    clean_cutoffs[feature]['progressive'] = f"{feature_cutoffs['nonprogressive_to_progressive']:.2f} - {feature_cutoffs['progressive_to_vigorous']:.2f}"
                    clean_cutoffs[feature]['vigorous'] = f"> {feature_cutoffs['progressive_to_vigorous']:.2f}"
            
            elif feature == 'BCF':
                # Special order: nonprogressive -> vigorous -> immotile -> progressive
                if 'nonprogressive_to_vigorous' in feature_cutoffs and 'vigorous_to_immotile' in feature_cutoffs and 'immotile_to_progressive' in feature_cutoffs:
                    clean_cutoffs[feature]['nonprogressive'] = f"< {feature_cutoffs['nonprogressive_to_vigorous']:.1f}"
                    clean_cutoffs[feature]['vigorous'] = f"{feature_cutoffs['nonprogressive_to_vigorous']:.1f} - {feature_cutoffs['vigorous_to_immotile']:.1f}"
                    clean_cutoffs[feature]['immotile'] = f"{feature_cutoffs['vigorous_to_immotile']:.1f} - {feature_cutoffs['immotile_to_progressive']:.1f}"
                    clean_cutoffs[feature]['progressive'] = f"> {feature_cutoffs['immotile_to_progressive']:.1f}"
            
            elif feature == 'LIN':
                # Special order: immotile -> nonprogressive -> vigorous -> progressive
                if 'immotile_to_nonprogressive' in feature_cutoffs and 'nonprogressive_to_vigorous' in feature_cutoffs and 'vigorous_to_progressive' in feature_cutoffs:
                    clean_cutoffs[feature]['immotile'] = f"< {feature_cutoffs['immotile_to_nonprogressive']:.2f}"
                    clean_cutoffs[feature]['nonprogressive'] = f"{feature_cutoffs['immotile_to_nonprogressive']:.2f} - {feature_cutoffs['nonprogressive_to_vigorous']:.2f}"
                    clean_cutoffs[feature]['vigorous'] = f"{feature_cutoffs['nonprogressive_to_vigorous']:.2f} - {feature_cutoffs['vigorous_to_progressive']:.2f}"
                    clean_cutoffs[feature]['progressive'] = f"> {feature_cutoffs['vigorous_to_progressive']:.2f}"
            
            elif feature == 'MAD':
                # Special order: immotile -> progressive -> nonprogressive -> vigorous
                if 'immotile_to_progressive' in feature_cutoffs and 'progressive_to_nonprogressive' in feature_cutoffs and 'nonprogressive_to_vigorous' in feature_cutoffs:
                    clean_cutoffs[feature]['immotile'] = f"< {feature_cutoffs['immotile_to_progressive']:.1f}"
                    clean_cutoffs[feature]['progressive'] = f"{feature_cutoffs['immotile_to_progressive']:.1f} - {feature_cutoffs['progressive_to_nonprogressive']:.1f}"
                    clean_cutoffs[feature]['nonprogressive'] = f"{feature_cutoffs['progressive_to_nonprogressive']:.1f} - {feature_cutoffs['nonprogressive_to_vigorous']:.1f}"
                    clean_cutoffs[feature]['vigorous'] = f"> {feature_cutoffs['nonprogressive_to_vigorous']:.1f}"
            
            elif feature == 'WOB':
                # Standard order: immotile -> nonprogressive -> vigorous -> progressive
                if 'immotile_to_nonprogressive' in feature_cutoffs and 'nonprogressive_to_vigorous' in feature_cutoffs and 'vigorous_to_progressive' in feature_cutoffs:
                    clean_cutoffs[feature]['immotile'] = f"< {feature_cutoffs['immotile_to_nonprogressive']:.2f}"
                    clean_cutoffs[feature]['nonprogressive'] = f"{feature_cutoffs['immotile_to_nonprogressive']:.2f} - {feature_cutoffs['nonprogressive_to_vigorous']:.2f}"
                    clean_cutoffs[feature]['vigorous'] = f"{feature_cutoffs['nonprogressive_to_vigorous']:.2f} - {feature_cutoffs['vigorous_to_progressive']:.2f}"
                    clean_cutoffs[feature]['progressive'] = f"> {feature_cutoffs['vigorous_to_progressive']:.2f}"
    
    return feature_stats, clean_cutoffs 