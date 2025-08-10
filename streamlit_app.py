import os
import io
import json
import base64
import tempfile

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from full_pipeline import json_to_df, predict_sperm_motility

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Sperm Motility Analysis",
    page_icon="🔬",
    layout="wide"
)

# ---------------- Helpers ----------------
def create_trajectory_image(track_frames, track_id, subtype_label):
    """Create a small trajectory plot and return as base64 image for tooltip"""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(track_frames['x'], track_frames['y'], 'r-', linewidth=1, alpha=0.8)
    ax.scatter(track_frames.iloc[0]['x'], track_frames.iloc[0]['y'],
               color='green', s=20, marker='o', zorder=3)
    ax.scatter(track_frames.iloc[-1]['x'], track_frames.iloc[-1]['y'],
               color='red', s=20, marker='x', zorder=3)
    ax.set_title(f'{track_id}\n({subtype_label})', fontsize=8)
    ax.set_xlabel('X', fontsize=6)
    ax.set_ylabel('Y', fontsize=6)
    ax.tick_params(labelsize=6)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# ---------------- UI ----------------
st.title("🔬 Sperm Motility Analysis Dashboard")
st.markdown("Upload JSON data and optionally a video to analyze sperm motility patterns")

st.sidebar.header("📁 Data Upload")

uploaded_json = st.sidebar.file_uploader(
    "Upload JSON file",
    type=['json'],
    help="Upload a JSON file with sperm motility data"
)

uploaded_video = st.sidebar.file_uploader(
    "Upload video (optional)",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
    help="Upload the corresponding video file for trajectory visualization."
)

participant_id = st.sidebar.text_input(
    "Participant ID",
    value="participant_001",
    help="Enter a unique identifier for this participant"
)

model_path = st.sidebar.text_input(
    "Model path",
    value="trained_gmm_model.pkl",
    help="Path to your trained GMM model"
)

# ---------------- Main analysis ----------------
if uploaded_json is not None:
    st.header("📊 Analysis Results")
    try:
        with st.spinner("Processing JSON data..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                tmp_file.write(uploaded_json.getvalue())
                tmp_path = tmp_file.name
            track_df, frame_df = json_to_df(tmp_path, participant_id)
            os.unlink(tmp_path)

        with st.spinner("Making predictions..."):
            predictions = predict_sperm_motility(track_df, model_path, include_umap=False)

        if predictions is None or len(predictions) == 0:
            st.warning("No predictions returned.")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tracks", len(predictions))
            with col2:
                st.metric("Progressive (%)", f"{(predictions['subtype_label'].eq('progressive').mean()*100):.1f}%")
            with col3:
                st.metric("Vigorous (%)", f"{(predictions['subtype_label'].eq('vigorous').mean()*100):.1f}%")
            with col4:
                st.metric("Immotile (%)", f"{(predictions['subtype_label'].eq('immotile').mean()*100):.1f}%")

            # Track selection + trajectory
            st.subheader("🎯 Track Trajectory Visualization")
            all_tracks = predictions['track_id'].tolist()
            default_track = st.session_state.get('selected_track', all_tracks[0])
            default_index = all_tracks.index(default_track) if default_track in all_tracks else 0

            selected_track = st.selectbox(
                "Select a track to visualize:",
                options=all_tracks,
                index=default_index,
                help="Choose a track to see its trajectory"
            )
            st.session_state['selected_track'] = selected_track

            if selected_track:
                track_frames = frame_df[frame_df['track_id'] == selected_track].sort_values('frame_num')
                if len(track_frames) == 0:
                    st.warning("No frame data for the selected track.")
                else:
                    track_pred = predictions.loc[predictions['track_id'] == selected_track].iloc[0]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Track ID", selected_track)
                    with col2:
                        st.metric("Subtype", track_pred['subtype_label'])
                    with col3:
                        conf_key = f"P_{track_pred['subtype_label']}"
                        if conf_key in track_pred:
                            st.metric("Confidence", f"{track_pred[conf_key]:.3f}")

                    fig_traj = go.Figure()
                    fig_traj.add_trace(go.Scatter(
                        x=track_frames['x'], y=track_frames['y'],
                        mode='lines+markers', name='Trajectory',
                        line=dict(color='red', width=2), marker=dict(size=4)
                    ))
                    fig_traj.add_trace(go.Scatter(
                        x=[track_frames.iloc[0]['x']], y=[track_frames.iloc[0]['y']],
                        mode='markers', name='Start',
                        marker=dict(color='green', size=10, symbol='circle')
                    ))
                    fig_traj.add_trace(go.Scatter(
                        x=[track_frames.iloc[-1]['x']], y=[track_frames.iloc[-1]['y']],
                        mode='markers', name='End',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    fig_traj.update_layout(
                        title=f"Trajectory for {selected_track} ({track_pred['subtype_label']})",
                        xaxis_title="X Position", yaxis_title="Y Position",
                        width=800, height=600, legend=dict(orientation="h")
                    )
                    fig_traj.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_traj, use_container_width=True)

            # Detailed results
            st.subheader("📋 Detailed Results")
            display_cols = ['track_id', 'subtype_label', 'VCL', 'VSL', 'LIN', 'ALH', 'BCF']
            prob_cols = [c for c in predictions.columns if c.startswith('P_')]
            show_cols = [c for c in display_cols + prob_cols if c in predictions.columns]
            st.dataframe(predictions[show_cols].round(3), use_container_width=True)

            st.download_button(
                label="📥 Download Results (CSV)",
                data=predictions.to_csv(index=False),
                file_name=f"{participant_id}_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        st.exception(e)
else:
    st.info("👈 Please upload a JSON file to begin analysis")
    st.subheader("📖 How to use:")
    st.markdown("""
    1. **Upload JSON file** - Your sperm motility data in JSON format  
    2. **Upload video (optional)** - Corresponding video file for visualization  
    3. **Enter Participant ID** - Unique identifier for this analysis  
    4. **Set Model Path** - Path to your trained GMM model  
    5. **View Results** - Trajectory visualization and results table  
    """)

# ---------------- Video section (after JSON if/else) ----------------
st.subheader("🎥 Video Analysis")
if uploaded_video is None:
    st.info("No video uploaded. Upload a video file in the sidebar to see it alongside the trajectory analysis.")
else:
    try:
        st.write(f"**Video file:** {uploaded_video.name}")
        st.write(f"**File size:** {uploaded_video.size / 1024 / 1024:.2f} MB")
        st.write(f"**File type:** {uploaded_video.type}")

        # Cleanup previous temp file
        prev_path = st.session_state.get("tmp_video_path")
        if prev_path and os.path.exists(prev_path):
            try:
                os.remove(prev_path)
            except OSError:
                pass

        # Persist with correct suffix and display
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_video:
            uploaded_video.seek(0)
            tmp_video.write(uploaded_video.getbuffer())
            tmp_video_path = tmp_video.name

        st.session_state["tmp_video_path"] = tmp_video_path
        st.video(tmp_video_path, start_time=0)

        st.info("- Trajectory plot uses the same coordinate system (y is inverted).  \n- Green dot = start, Red X = end.")
    except Exception as e:
        st.error(f"Error displaying video: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("🔬 Sperm Motility Analysis Dashboard | Built with Streamlit")
