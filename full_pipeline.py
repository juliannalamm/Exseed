import pickle
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import json
import os
import numpy as np

def json_to_df(json_file_path, participant_id):
    """Convert JSON file to DataFrame with one row per track"""
    with open(json_file_path, 'r') as file:
        data = json.load(file) 
    track_rows = []
    frame_rows = []
    for i, track in enumerate(data["Tracks"]):
        motility_params = track["Motility Parameters"]
        # create a dictionary for each track
        track_row = {
            'participant_id': participant_id,
            'track_id': f"{participant_id}_track_{i}",
            'ALH':motility_params['ALH'],
            'BCF': motility_params['BCF'],
            'LIN': motility_params['LIN'],
            'MAD': motility_params['MAD'],
            'STR': motility_params['STR'],
            'track_length_frames': motility_params['Track Length (frames)'],
            'VAP': motility_params['VAP'],
            'VCL': motility_params['VCL'],
            'VSL': motility_params['VSL'],
            'WOB': motility_params['WOB'],
        }
        track_rows.append(track_row)
        
        # Frame-level data
        track_coords = track['Track']
        for frame_name, coords in track_coords.items():
            frame_num = int(frame_name.replace('Frame ', ''))
            frame_row = {
                'participant_id': participant_id,
                'track_id': f"{participant_id}_track_{i}",
                'frame_num': frame_num,
                'x': coords[0],
                'y': coords[1],
            }
            frame_rows.append(frame_row)
    
    return pd.DataFrame(track_rows), pd.DataFrame(frame_rows)


def predict_sperm_motility(new_data, model_path='trained_gmm_model.pkl', include_umap=True):
    """
    Predict sperm motility subtypes for new data with optional UMAP projection
    
    Parameters:
    new_data: DataFrame with motility features
    model_path: Path to saved model
    include_umap: Whether to include UMAP projection
    
    Returns:
    DataFrame with predictions, probabilities, and UMAP coordinates
    """
    # Load the model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    gmm = model_data['gmm']
    scaler = model_data['scaler']
    cluster_to_label = model_data['cluster_to_label']
    features = model_data['features']
    umap_model = model_data.get('umap_model', None)
    
    # Prepare the data
    if isinstance(new_data, dict):
        new_df = pd.DataFrame([new_data])
    else:
        new_df = new_data.copy()
    
    # Ensure all required features are present
    missing_features = set(features) - set(new_df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Scale the features
    X_scaled = scaler.transform(new_df[features])
    
    # Make predictions
    cluster_labels = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    # Add predictions to DataFrame
    result_df = new_df.copy()
    result_df['cluster_id'] = cluster_labels
    result_df['subtype_label'] = result_df['cluster_id'].map(cluster_to_label)
    
    # Add probabilities
    for i in range(gmm.n_components):
        result_df[f'P_cluster_{i}'] = probabilities[:, i]
        if i in cluster_to_label:
            result_df[f'P_{cluster_to_label[i]}'] = probabilities[:, i]
    
    # Add UMAP projection if requested and available
    if include_umap and umap_model is not None:
        umap_embedding = umap_model.transform(X_scaled)
        result_df['umap_1'] = umap_embedding[:, 0]
        result_df['umap_2'] = umap_embedding[:, 1]
    
    return result_df


def plot_umap_with_predictions(predictions_df, output_path='umap_predictions.png'):
    """
    Create UMAP visualization of new data with predictions
    
    Parameters:
    predictions_df: DataFrame with predictions and UMAP coordinates
    output_path: Path to save the plot
    """
    if 'umap_1' not in predictions_df.columns or 'umap_2' not in predictions_df.columns:
        print("UMAP coordinates not found. Run prediction with include_umap=True")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple'}
    cluster_to_label = {0: 'nonprogressive', 1: 'immotile', 2: 'vigorous', 3: 'progressive'}
    
    for cluster_id in predictions_df['cluster_id'].unique():
        cluster_data = predictions_df[predictions_df['cluster_id'] == cluster_id]
        plt.scatter(cluster_data['umap_1'], cluster_data['umap_2'], 
                   c=colors[cluster_id], label=cluster_to_label[cluster_id], 
                   alpha=0.7, s=50)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Projection of New Sperm Motility Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"UMAP plot saved to {output_path}")


def visualize_trajectories_with_predictions(frame_df, track_df, output_path='predicted_trajectories.mp4'):
    """
    Create animated visualization of trajectories with cluster predictions
    
    Parameters:
    frame_df: DataFrame with ['track_id', 'frame_num', 'x', 'y']
    track_df: DataFrame with predictions (from predict_sperm_motility)
    output_path: Path to save the animation
    """
    # Merge predictions with frame data
    merged_df = frame_df.merge(
        track_df[['track_id', 'cluster_id', 'subtype_label']], 
        on='track_id', 
        how='left'
    )
    
    # Sort by frame
    merged_df = merged_df.sort_values(by=['frame_num'])
    
    # Get unique frame numbers
    frames = sorted(merged_df['frame_num'].unique())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple'}
    
    # Initialize track history
    track_history = {}
    trail_lines = {}
    
    for track_id in merged_df['track_id'].unique():
        cluster = track_df.loc[track_df['track_id'] == track_id, 'cluster_id'].values[0]
        color = colors[cluster]
        trail_lines[track_id], = ax.plot([], [], '-', color=color, linewidth=1.5, alpha=0.7)
        track_history[track_id] = {'x': [], 'y': []}
    
    # Set axis limits
    ax.set_xlim(merged_df['x'].min() - 10, merged_df['x'].max() + 10)
    ax.set_ylim(merged_df['y'].min() - 10, merged_df['y'].max() + 10)
    ax.invert_yaxis()
    ax.set_title("Predicted Sperm Motility Trajectories")
    
    # Add legend
    cluster_to_label = {0: 'nonprogressive', 1: 'immotile', 2: 'vigorous', 3: 'progressive'}
    legend_elements = [plt.Line2D([0], [0], color=colors[i], label=cluster_to_label[i]) 
                      for i in range(4)]
    ax.legend(handles=legend_elements)
    
    def update(frame_num):
        current_df = merged_df[merged_df['frame_num'] == frame_num]
        for track_id in current_df['track_id'].unique():
            track_data = current_df[current_df['track_id'] == track_id]
            if not track_data.empty:
                x = track_data['x'].values[0]
                y = track_data['y'].values[0]
                track_history[track_id]['x'].append(x)
                track_history[track_id]['y'].append(y)
                trail_lines[track_id].set_data(track_history[track_id]['x'], track_history[track_id]['y'])
        return trail_lines.values()
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    ani.save(output_path, writer="ffmpeg", fps=10)
    
    print(f"Animation saved to {output_path}")
    return ani


def analyze_new_participant(json_file_path, participant_id, model_path='trained_gmm_model.pkl'):
    """
    Complete analysis pipeline for new participant data
    
    Parameters:
    json_file_path: Path to new JSON file
    participant_id: ID for the new participant
    model_path: Path to trained model
    """
    # Process new data
    new_track_df, new_frame_df = json_to_df(json_file_path, participant_id)
    
    # Make predictions with UMAP
    predictions = predict_sperm_motility(new_track_df, model_path, include_umap=True)
    
    # Create UMAP visualization
    plot_umap_with_predictions(predictions, f"{participant_id}_umap.png")
    
    # Create trajectory visualization
    visualize_trajectories_with_predictions(
        new_frame_df, predictions, f"{participant_id}_trajectories.mp4"
    )
    
    # Print summary
    print(f"\nAnalysis complete for participant {participant_id}")
    print("Cluster distribution:")
    print(predictions['subtype_label'].value_counts())
    
    return predictions, new_frame_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sperm motility data using trained GMM model')
    parser.add_argument('json_file', help='Path to JSON file with sperm motility data')
    parser.add_argument('participant_id', help='Participant ID for the analysis')
    parser.add_argument('--model', default='trained_gmm_model.pkl', 
                       help='Path to trained model file (default: trained_gmm_model.pkl)')
    parser.add_argument('--output-dir', default='.', 
                       help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    print(f"Analyzing participant: {args.participant_id}")
    print(f"Input file: {args.json_file}")
    print(f"Model file: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Run the analysis
        predictions, frame_data = analyze_new_participant(
            json_file_path=args.json_file,
            participant_id=args.participant_id,
            model_path=args.model
        )
        
        # Save results to CSV
        output_csv = f"{args.output_dir}/{args.participant_id}_predictions.csv"
        predictions.to_csv(output_csv, index=False)
        print(f"\nPredictions saved to: {output_csv}")
        
        # Print detailed summary
        print(f"\nDetailed Results for {args.participant_id}:")
        print("=" * 50)
        print("Cluster Distribution:")
        cluster_counts = predictions['subtype_label'].value_counts()
        cluster_percentages = predictions['subtype_label'].value_counts(normalize=True) * 100
        
        for cluster, count in cluster_counts.items():
            percentage = cluster_percentages[cluster]
            print(f"  {cluster}: {count} tracks ({percentage:.1f}%)")
        
        print(f"\nTotal tracks analyzed: {len(predictions)}")
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()