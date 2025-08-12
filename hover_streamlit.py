import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
from full_pipeline import (
    json_to_df,
    predict_sperm_motility,
    plot_umap_with_predictions,
    overlay_trajectories_on_video,
    MODEL_PATH
)
from global_umap_utils import get_global_umap_comparison
import subprocess


def convert_to_h264(input_path: str, output_path: str):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


# UI Setup
st.set_page_config(page_title="Sperm Motility Analyzer", layout="wide")
st.title("🧬 Sperm Motility Classification & Trajectory Viewer")

# Clear results button in sidebar
if st.sidebar.button("🗑️ Clear Results"):
    for key in ['preds', 'frame_df', 'h264_path', 'preds_csv', 'participant_id']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Global variables for upload state
json_file = None
video_file = None
run_btn = False

# =====================
# 🔹 Tabbed Analysis Interface
# =====================
tab1, tab2 = st.tabs(["🌍 Global Explorer", "📊 Individual Analysis"])

with tab1:
    st.subheader("🌍 Global UMAP Comparison")
    
    # Load training data to get available participants
    from global_umap_utils import load_or_create_training_umap_data
    training_data = load_or_create_training_umap_data()
    
    if training_data is not None:
        # Get unique participants from training data
        if 'participant_id' in training_data.columns:
            available_participants = sorted(training_data['participant_id'].unique())
        else:
            # Extract participant IDs from track_ids
            available_participants = sorted(
                training_data['track_id'].str.split('_track_').str[0].unique()
            )
        
        # Participant selection dropdown
        st.markdown("**🔍 Select Participant to Highlight**")
        selected_participant = st.selectbox(
            "Choose a participant to highlight in the global UMAP:",
            options=available_participants,
            index=0 if available_participants else None,
            help="Select a participant to highlight their tracks in the global UMAP plot"
        )
        
        # Create global UMAP plot with selected participant
        with st.spinner("Creating global UMAP comparison..."):
            global_fig, comparison_stats = get_global_umap_comparison(None, participant_id=selected_participant)
        
        if comparison_stats:
            st.plotly_chart(global_fig, use_container_width=True)
            
            # Show comparison statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Overall Training Data Distribution**")
                st.dataframe(comparison_stats['training_distribution'])
            with col2:
                st.markdown(f"**📊 {selected_participant} Distribution (from training data)**")
                st.dataframe(comparison_stats['new_data_distribution'])
            
            st.info(f"💡 **Global UMAP Plot**: Training data points are small and transparent. {selected_participant}'s tracks are highlighted as large, red-outlined diamonds. The distribution shows how this participant's tracks were classified during training.")
        else:
            st.plotly_chart(global_fig, use_container_width=True)
            st.warning("⚠️ Could not load training data.")
        
        # Feature Analysis Section
        st.markdown("---")
        st.subheader("📊 Feature Distribution Analysis")
        
        with st.spinner("Analyzing feature distributions..."):
            from global_umap_utils import get_feature_analysis, create_single_feature_plot
            feature_stats, cutoffs = get_feature_analysis(training_data)
            
            # Display cutoff suggestions
            st.markdown("**🎯 GMM-Based Cutoff Ranges**")
            
            # Create tabs for different features
            feature_tabs = st.tabs(["ALH", "BCF", "LIN", "VCL", "VSL", "WOB", "MAD", "STR", "VAP"])
            
            for i, feature in enumerate(["ALH", "BCF", "LIN", "VCL", "VSL", "WOB", "MAD", "STR", "VAP"]):
                with feature_tabs[i]:
                    # Create two columns: chart on left, cutoffs on right
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create and display the individual feature plot
                        feature_fig = create_single_feature_plot(training_data, feature)
                        st.plotly_chart(feature_fig, use_container_width=True)
                    
                    with col2:
                        if feature in cutoffs:
                            st.markdown(f"**{feature} Cutoffs:**")
                            
                            # Get all cutoffs for this feature and sort them
                            feature_cutoffs = cutoffs[feature]
                            cutoff_values = []
                            for cutoff_name, cutoff_value in feature_cutoffs.items():
                                cutoff_values.append((cutoff_name, cutoff_value))
                            cutoff_values.sort(key=lambda x: x[1])  # Sort by value
                            
                            # Display in range format
                            if len(cutoff_values) >= 3:
                                st.markdown("**Classification Ranges:**")
                                st.write(f"- **Immotile**: {feature} < {cutoff_values[0][1]:.3f}")
                                st.write(f"- **Nonprogressive**: {cutoff_values[0][1]:.3f} ≤ {feature} < {cutoff_values[1][1]:.3f}")
                                st.write(f"- **Progressive**: {cutoff_values[1][1]:.3f} ≤ {feature} < {cutoff_values[2][1]:.3f}")
                                st.write(f"- **Vigorous**: {feature} ≥ {cutoff_values[2][1]:.3f}")
                            
                            st.markdown("**Raw Cutoff Values:**")
                            for cutoff_name, cutoff_value in cutoff_values:
                                st.write(f"- {cutoff_name}: {cutoff_value:.3f}")
                        else:
                            st.write("No cutoff data available for this feature.")
                        
                        # Show feature statistics
                        st.markdown("**📋 Feature Statistics by Cluster:**")
                        for subtype, stats in feature_stats.items():
                            if feature in stats:
                                stat = stats[feature]
                                st.write(f"**{subtype}**: mean={stat['mean']:.3f}, std={stat['std']:.3f}, q75={stat['q75']:.3f}")
            
            # Summary recommendations
            st.markdown("**💡 Recommendations for Identifying Sperm Types:**")
            st.markdown("""
            - **Hyperactivated/Vigorous**: High ALH, high BCF, lower LIN
            - **Progressive**: High VCL, high VSL, high LIN  
            - **Nonprogressive**: Lower VCL, lower LIN
            - **Immotile**: Very low VCL, very low ALH
            """)
    else:
        st.error("❌ Training data not found. Please ensure train_track_df.csv is available.")

with tab2:
    st.subheader("📊 Individual Analysis")
    
    # Upload section
    st.markdown("**📤 Upload Files**")
    col1, col2 = st.columns(2)
    with col1:
        json_file = st.file_uploader("Upload JSON", type=["json"])
    with col2:
        video_file = st.file_uploader("Upload Video", type=["mp4"])
    
    run_btn = st.button("▶️ Run Analysis", type="primary")
    
    # Process Inputs
    if run_btn and json_file and video_file:
        with st.spinner("Processing your data..."):
            # Save uploaded files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as jtf:
                jtf.write(json_file.read())
                json_path = jtf.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vtf:
                vtf.write(video_file.read())
                video_path = vtf.name

            participant_id = os.path.splitext(os.path.basename(json_file.name))[0]
            out_dir = os.path.join(tempfile.gettempdir(), f"{participant_id}_outputs")
            os.makedirs(out_dir, exist_ok=True)

            # STEP 1: Parse JSON
            track_df, frame_df = json_to_df(json_path, participant_id)

            # STEP 2: Predict Subtypes + UMAP
            preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=True)
            preds_csv = os.path.join(out_dir, f"{participant_id}_predictions.csv")
            preds.to_csv(preds_csv, index=False)

            # STEP 3: Static PNG (optional fallback)
            umap_png = os.path.join(out_dir, f"{participant_id}_umap.png")
            plot_umap_with_predictions(preds, umap_png)

            # STEP 4: Overlay all trajectories on video
            raw_overlay_path = os.path.join(out_dir, f"{participant_id}_raw_overlay.mp4")
            h264_path = os.path.join(out_dir, f"{participant_id}_overlay_h264.mp4")

            overlay_trajectories_on_video(
                frame_df=frame_df,
                track_df=preds,
                video_path=video_path,
                output_path=raw_overlay_path
            )
            convert_to_h264(raw_overlay_path, h264_path)
            
            # Store data in session state to persist between interactions
            st.session_state['preds'] = preds
            st.session_state['frame_df'] = frame_df
            st.session_state['h264_path'] = h264_path
            st.session_state['preds_csv'] = preds_csv
            st.session_state['participant_id'] = participant_id
    
    # Display Results (if data exists in session state)
    if 'preds' in st.session_state and 'frame_df' in st.session_state:
        preds = st.session_state['preds']
        frame_df = st.session_state['frame_df']
        h264_path = st.session_state['h264_path']
        preds_csv = st.session_state['preds_csv']
        participant_id = st.session_state['participant_id']
        
        st.success(f"✅ Analysis complete for {participant_id}")
        
        # Debug information
        with st.expander("🔧 Debug Info"):
            st.write("Session state keys:", list(st.session_state.keys()))
            st.write("Data shape:", preds.shape)
            st.write("Participant:", participant_id)
            st.write(f"**Columns:** {list(preds.columns)}")
            st.write(f"**Has umap_1:** {'umap_1' in preds.columns}")
            st.write(f"**Has umap_2:** {'umap_2' in preds.columns}")
            if 'umap_1' in preds.columns and 'umap_2' in preds.columns:
                st.write(f"**UMAP 1 range:** {preds['umap_1'].min():.3f} to {preds['umap_1'].max():.3f}")
                st.write(f"**UMAP 2 range:** {preds['umap_2'].min():.3f} to {preds['umap_2'].max():.3f}")
                st.write(f"**Number of points:** {len(preds)}")
                st.write(f"**Unique clusters:** {sorted(preds['cluster_id'].unique())}")
        
        # Create two columns for analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🗺️ Individual Data on Global UMAP**")
            
            if 'umap_1' in preds.columns and 'umap_2' in preds.columns:
                # Create a combined plot showing training data + new data
                from global_umap_utils import load_or_create_training_umap_data
                
                # Load training data for comparison
                training_data = load_or_create_training_umap_data()
                if training_data is not None:
                    # Create a combined plot showing both training and new data
                    import plotly.graph_objects as go
                    
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
                    for cluster_id in sorted(training_data['cluster_id'].unique()):
                        cluster_data = training_data[training_data['cluster_id'] == cluster_id]
                        subtype = cluster_data['subtype_label'].iloc[0]
                        
                        fig.add_trace(go.Scatter(
                            x=cluster_data['umap_1'],
                            y=cluster_data['umap_2'],
                            mode='markers',
                            marker=dict(
                                size=4,
                                color=cluster_colors.get(cluster_id, '#888'),
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
                    
                    # Add new participant data (larger, more prominent)
                    for cluster_id in sorted(preds['cluster_id'].unique()):
                        cluster_data = preds[preds['cluster_id'] == cluster_id]
                        subtype = cluster_data['subtype_label'].iloc[0]
                        
                        fig.add_trace(go.Scatter(
                            x=cluster_data['umap_1'],
                            y=cluster_data['umap_2'],
                            mode='markers',
                            marker=dict(
                                size=12,
                                color=cluster_colors.get(cluster_id, '#444'),
                                opacity=1.0,
                                line=dict(color='red', width=3),
                                symbol='diamond'
                            ),
                            name=f'Your Data: {subtype}',
                            showlegend=True,
                            hovertemplate='<b>Your Participant</b><br>' +
                                         'Track: %{text}<br>' +
                                         f'Subtype: {subtype}<br>' +
                                         'Cluster: %{customdata}<br>' +
                                         '<extra></extra>',
                            text=cluster_data['track_id'],
                            customdata=cluster_data['cluster_id']
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Global UMAP: Your Data vs Training Data - {participant_id}",
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **Global UMAP**: Training data points are small and transparent. Your participant's tracks are highlighted as large, red-outlined diamonds.")
                else:
                    # Fallback to individual UMAP if training data not available
                    fig = px.scatter(
                        preds,
                        x='umap_1',
                        y='umap_2',
                        color='subtype_label' if 'subtype_label' in preds else 'cluster_id',
                        hover_data=['track_id'],
                        title="Individual UMAP (Training data not available)",
                        width=400,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.warning("⚠️ Training data not available, showing individual UMAP only.")
            else:
                st.warning("UMAP coordinates not available.")
        
        with col2:
            st.markdown("**🎯 Track Trajectory Viewer**")
            
            # Create a dropdown for track selection
            available_tracks = sorted(preds['track_id'].unique())
            selected_track = st.selectbox(
                "Select a track:",
                available_tracks,
                format_func=lambda x: f"{x} ({preds[preds['track_id']==x]['subtype_label'].iloc[0]})"
            )
            
            if selected_track:
                st.success(f"🧬 Selected: `{selected_track}`")
                
                traj_df = frame_df[frame_df['track_id'] == selected_track].sort_values("frame_num")
                if not traj_df.empty:
                    fig_traj = px.line(
                        traj_df,
                        x='x',
                        y='y',
                        title=f'Trajectory: {selected_track}',
                        markers=True,
                        width=400,
                        height=400
                    )
                    st.plotly_chart(fig_traj, use_container_width=True)
                    
                    # Show track statistics in a more compact format
                    track_stats = preds[preds['track_id'] == selected_track].iloc[0]
                    st.markdown(f"""
                    **Track Info:**
                    - **Subtype:** {track_stats['subtype_label']}
                    - **Cluster:** {track_stats['cluster_id']}
                    - **Frames:** {len(traj_df)}
                    """)
                else:
                    st.warning("No trajectory data found.")
        
        # Cluster distribution summary
        st.markdown("**📊 Cluster Distribution Summary**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Subtype Distribution**")
            if "subtype_label" in preds.columns:
                subtype_counts = preds["subtype_label"].value_counts()
                st.dataframe(subtype_counts.rename("Count"))
            else:
                st.dataframe(preds["cluster_id"].value_counts().rename("Count"))
        with col2:
            st.markdown("**Summary Statistics**")
            st.markdown(f"""
            - **Total Tracks:** {len(preds)}
            - **Unique Subtypes:** {preds['subtype_label'].nunique() if 'subtype_label' in preds.columns else preds['cluster_id'].nunique()}
            - **Participant ID:** {participant_id}
            """)
        
        # Full trajectory video overlay
        st.markdown("**🎬 Full Trajectory Video Overlay**")
        if os.path.exists(h264_path):
            with open(h264_path, 'rb') as video_file:
                st.video(video_file.read())
        else:
            st.error("Video overlay not found.")
        
        # Download results
        st.markdown("**📥 Download Results**")
        if os.path.exists(preds_csv):
            with open(preds_csv, 'r') as f:
                st.download_button(
                    label="📊 Download Predictions CSV",
                    data=f.read(),
                    file_name=f"{participant_id}_predictions.csv",
                    mime="text/csv"
                )
    elif run_btn and (not json_file or not video_file):
        st.error("❌ Please upload both JSON and video files to run analysis.")
