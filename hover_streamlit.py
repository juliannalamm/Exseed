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
            st.markdown("**📊 Individual UMAP**")
            
            if 'umap_1' in preds.columns and 'umap_2' in preds.columns:
                # Check if we have meaningful UMAP coordinates
                umap_range_1 = preds['umap_1'].max() - preds['umap_1'].min()
                umap_range_2 = preds['umap_2'].max() - preds['umap_2'].min()
                
                if umap_range_1 > 0.1 and umap_range_2 > 0.1:
                    # Create the scatter plot
                    fig = px.scatter(
                        preds,
                        x='umap_1',
                        y='umap_2',
                        color='subtype_label' if 'subtype_label' in preds else 'cluster_id',
                        hover_data=['track_id'],
                        title="Individual UMAP",
                        width=400,
                        height=400
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a note about the plot
                    st.info("💡 Individual clustering pattern for your uploaded data.")
                else:
                    st.error(f"UMAP coordinates are too compressed (ranges: {umap_range_1:.3f}, {umap_range_2:.3f}).")
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
