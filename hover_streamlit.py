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

# Upload
st.sidebar.header("📤 Upload Files")
json_file = st.sidebar.file_uploader("Upload JSON", type=["json"])
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])
run_btn = st.sidebar.button("▶️ Run Analysis")

# Clear results button
if st.sidebar.button("🗑️ Clear Results"):
    for key in ['preds', 'frame_df', 'h264_path', 'preds_csv', 'participant_id']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

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

# =====================
# 🔹 Display Results (if data exists in session state)
# =====================

# Debug: Show session state info
with st.sidebar.expander("🔧 Debug Info"):
    st.write("Session state keys:", list(st.session_state.keys()))
    if 'preds' in st.session_state:
        st.write("Data shape:", st.session_state['preds'].shape)
    if 'participant_id' in st.session_state:
        st.write("Participant:", st.session_state['participant_id'])

if 'preds' in st.session_state and 'frame_df' in st.session_state:
    preds = st.session_state['preds']
    frame_df = st.session_state['frame_df']
    h264_path = st.session_state['h264_path']
    preds_csv = st.session_state['preds_csv']
    participant_id = st.session_state['participant_id']
    
    st.success(f"✅ Analysis complete for {participant_id}")
    
    # =====================
    # 🔹 UMAP Interactive Plot
    # =====================
    st.subheader("📊 Interactive UMAP Clustering")
    
    # Debug information
    with st.expander("🔍 Debug Info"):
        st.write(f"**Data shape:** {preds.shape}")
        st.write(f"**Columns:** {list(preds.columns)}")
        st.write(f"**Has umap_1:** {'umap_1' in preds.columns}")
        st.write(f"**Has umap_2:** {'umap_2' in preds.columns}")
        if 'umap_1' in preds.columns and 'umap_2' in preds.columns:
            st.write(f"**UMAP 1 range:** {preds['umap_1'].min():.3f} to {preds['umap_1'].max():.3f}")
            st.write(f"**UMAP 2 range:** {preds['umap_2'].min():.3f} to {preds['umap_2'].max():.3f}")
            st.write(f"**Number of points:** {len(preds)}")
            st.write(f"**Unique clusters:** {sorted(preds['cluster_id'].unique())}")
    
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
                title="UMAP Projection",
                width=800,
                height=600
            )
            
            # Display the plot using st.plotly_chart instead of plotly_events
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a note about the plot
            st.info("💡 The UMAP plot shows sperm tracks clustered by motility patterns. Each point represents a sperm track, colored by its predicted motility subtype.")
        else:
            st.error(f"UMAP coordinates are too compressed (ranges: {umap_range_1:.3f}, {umap_range_2:.3f}). This indicates an issue with the UMAP model.")
    else:
        st.warning("UMAP coordinates not available. Check if the model contains a UMAP model and if include_umap=True.")

    # =====================
    # 🔹 Show Trajectory for Selected Track
    # =====================
    st.subheader("🎯 Track Trajectory Viewer")
    
    # Create a dropdown for track selection
    available_tracks = sorted(preds['track_id'].unique())
    selected_track = st.selectbox(
        "Select a track to view its trajectory:",
        available_tracks,
        format_func=lambda x: f"{x} ({preds[preds['track_id']==x]['subtype_label'].iloc[0]})"
    )
    
    if selected_track:
        st.success(f"🧬 Selected Track: `{selected_track}`")
        
        traj_df = frame_df[frame_df['track_id'] == selected_track].sort_values("frame_num")
        if not traj_df.empty:
            fig_traj = px.line(
                traj_df,
                x='x',
                y='y',
                title=f'Sperm Trajectory for {selected_track}',
                markers=True,
                width=700,
                height=500
            )
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # Show track statistics
            track_stats = preds[preds['track_id'] == selected_track].iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Subtype", track_stats['subtype_label'])
            with col2:
                st.metric("Cluster ID", track_stats['cluster_id'])
            with col3:
                st.metric("Track Length", f"{len(traj_df)} frames")
        else:
            st.warning("No trajectory data found for that track.")

    # =====================
    # 🔹 Show Cluster Counts
    # =====================
    st.subheader("📈 Cluster Summary")
    if "subtype_label" in preds.columns:
        st.dataframe(preds["subtype_label"].value_counts().rename("Count"))
    else:
        st.dataframe(preds["cluster_id"].value_counts().rename("Count"))

    # =====================
    # 🔹 Show Video with All Trajectories
    # =====================
    st.subheader("🎥 Full Trajectory Overlay")
    st.video(h264_path)

    # =====================
    # 🔹 Download Results
    # =====================
    st.subheader("📥 Download")
    st.download_button("📥 Download Predictions CSV", data=open(preds_csv, "rb"), file_name="predictions.csv")
    st.download_button("📥 Download Overlayed Video", data=open(h264_path, "rb"), file_name="overlayed_trajectories.mp4")

elif run_btn and json_file and video_file:
    st.info("🔄 Processing... Please wait for the analysis to complete.")
else:
    st.info("⬅️ Please upload both a JSON and MP4 file, then click Run.")
