#!/usr/bin/env python3
"""
Calculate median participant metrics for sperm motility analysis.
"""

import pandas as pd
import numpy as np

def calculate_median_participant_metrics():
    """
    Calculate median metrics across all participants.
    
    Returns:
        dict: Dictionary containing median values for various metrics
    """
    # Load the training data
    df = pd.read_csv('train_track_df.csv')
    
    # Extract participant_id from track_id if not already present
    if 'participant_id' not in df.columns and 'track_id' in df.columns:
        df['participant_id'] = df['track_id'].str.split('_track_').str[0]
    
    # Calculate metrics for each participant
    participant_metrics = []
    
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant]
        total_tracks = len(participant_data)
        
        if total_tracks == 0:
            continue
            
        # Calculate subtype percentages
        subtype_counts = participant_data['subtype_label'].value_counts()
        percentages = {}
        
        for subtype in ['progressive', 'vigorous', 'immotile', 'nonprogressive']:
            count = subtype_counts.get(subtype, 0)
            percentage = (count / total_tracks) * 100
            percentages[subtype] = percentage
        
        # Calculate motile percentage (progressive + vigorous + nonprogressive)
        motile_percentage = percentages['progressive'] + percentages['vigorous'] + percentages['nonprogressive']
        
        participant_metrics.append({
            'participant_id': participant,
            'total_tracks': total_tracks,
            'progressive': percentages['progressive'],
            'vigorous': percentages['vigorous'],
            'immotile': percentages['immotile'],
            'nonprogressive': percentages['nonprogressive'],
            'motile': motile_percentage
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(participant_metrics)
    
    # Calculate median values
    median_metrics = {
        'total_tracks': metrics_df['total_tracks'].median(),
        'progressive': metrics_df['progressive'].median(),
        'vigorous': metrics_df['vigorous'].median(),
        'immotile': metrics_df['immotile'].median(),
        'nonprogressive': metrics_df['nonprogressive'].median(),
        'motile': metrics_df['motile'].median()
    }
    
    return median_metrics, metrics_df

def calculate_typical_patient_feature_profile():
    """
    Calculate the typical feature profile by taking the median of average feature values per cluster across participants.
    
    Returns:
        tuple: (typical_profile_dict, features_df)
            - typical_profile_dict: Dictionary with cluster names as keys and feature values as nested dicts
            - features_df: DataFrame with participant-level feature averages
    """
    # Load the training data
    df = pd.read_csv('train_track_df.csv')
    
    # Extract participant_id from track_id if not already present
    if 'participant_id' not in df.columns and 'track_id' in df.columns:
        df['participant_id'] = df['track_id'].str.split('_track_').str[0]
    
    # Define features to analyze
    features = ['ALH', 'BCF', 'LIN', 'VCL', 'VSL', 'WOB', 'MAD', 'STR', 'VAP']
    
    # Calculate average feature values per cluster per participant
    participant_features = []
    
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant]
        
        if len(participant_data) == 0:
            continue
        
        # Calculate average feature values for each cluster
        for cluster in participant_data['subtype_label'].unique():
            cluster_data = participant_data[participant_data['subtype_label'] == cluster]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate mean for each feature
            feature_means = {}
            for feature in features:
                if feature in cluster_data.columns:
                    feature_means[feature] = cluster_data[feature].mean()
                else:
                    feature_means[feature] = 0
            
            participant_features.append({
                'participant_id': participant,
                'cluster': cluster,
                **feature_means
            })
    
    # Convert to DataFrame
    features_df = pd.DataFrame(participant_features)
    
    # Calculate median feature values for each cluster (typical profile)
    typical_profile = {}
    
    for cluster in ['vigorous', 'progressive', 'nonprogressive', 'immotile']:
        cluster_data = features_df[features_df['cluster'] == cluster]
        
        if len(cluster_data) > 0:
            typical_profile[cluster] = {}
            for feature in features:
                typical_profile[cluster][feature] = cluster_data[feature].median()
        else:
            typical_profile[cluster] = {feature: 0 for feature in features}
    
    return typical_profile, features_df

def get_metric_status(value, metric_type):
    """
    Determine the status (Normal, Low, High) for a given metric value.
    
    Args:
        value (float): The metric value
        metric_type (str): Type of metric ('progressive', 'vigorous', 'immotile', 'nonprogressive', 'motile')
    
    Returns:
        str: Status ('Normal', 'Low', 'High')
    """
    # Define thresholds for each metric type
    thresholds = {
        'progressive': {'low': 30, 'high': 60},
        'vigorous': {'low': 10, 'high': 30},
        'immotile': {'low': 5, 'high': 20},
        'nonprogressive': {'low': 10, 'high': 25},
        'motile': {'low': 40, 'high': 80}
    }
    
    if metric_type not in thresholds:
        return 'Normal'
    
    low_threshold = thresholds[metric_type]['low']
    high_threshold = thresholds[metric_type]['high']
    
    if value < low_threshold:
        return 'Low'
    elif value > high_threshold:
        return 'High'
    else:
        return 'Normal'

if __name__ == "__main__":
    # Test the functions
    median_metrics, metrics_df = calculate_median_participant_metrics()
    
    print("Median Participant Metrics:")
    print("=" * 40)
    for metric, value in median_metrics.items():
        status = get_metric_status(value, metric)
        print(f"{metric.title()}: {value:.1f}% ({status})")
    
    print(f"\nTotal participants analyzed: {len(metrics_df)}")
    print(f"Median tracks per participant: {median_metrics['total_tracks']:.0f}") 