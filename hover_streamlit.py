import streamlit as st
import os
import tempfile
import pandas as pd
from full_pipeline import (
    json_to_df,
    predict_sperm_motility,
    plot_umap_with_predictions,
    overlay_trajectories_on_video,
    MODEL_PATH
)
import subprocess

# -----------------------------
# Utility: Convert to H.264
# -----------------------------
def convert_to_h264(input_path: str, output_path: str):
    """
    Convert MP4 to H.264-encoded MP4 using FFmpeg for browser playback.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Sperm Motility Analyzer", layout="wide")
st.title("🧬 Sperm Motility Classification & Trajectory Viewer")

# Uploads
st.sidebar.header("📤 Upload Files")
json_file = st.sidebar.file_uploader("Upload JSON", type=["json"])
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])
run_btn = st.sidebar.button("▶️ Run Analysis")

# -----------------------------
# Process Inputs
# -----------------------------
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

        # STEP 2: Predict Motility Subtypes
        preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=True)
        preds_csv = os.path.join(out_dir, f"{participant_id}_predictions.csv")
        preds.to_csv(preds_csv, index=False)

        # STEP 3: Generate UMAP Plot
        umap_png = os.path.join(out_dir, f"{participant_id}_umap.png")
        plot_umap_with_predictions(preds, umap_png)

        # STEP 4: Overlay trajectories on video
        raw_overlay_path = os.path.join(out_dir, f"{participant_id}_raw_overlay.mp4")
        h264_path = os.path.join(out_dir, f"{participant_id}_overlay_h264.mp4")

        overlay_trajectories_on_video(
            frame_df=frame_df,
            track_df=preds,
            video_path=video_path,
            output_path=raw_overlay_path
        )

        # STEP 5: Convert for Streamlit-friendly playback
        convert_to_h264(raw_overlay_path, h264_path)

    # -----------------------------
    # Show Results
    # -----------------------------
    st.success("✅ Analysis complete!")

    st.subheader("📊 UMAP Clustering")
    st.image(umap_png, use_column_width=True)

    st.subheader("📈 Cluster Summary")
    if "subtype_label" in preds.columns:
        st.dataframe(preds["subtype_label"].value_counts().rename("Count"))
    else:
        st.dataframe(preds["cluster_id"].value_counts().rename("Count"))

    st.subheader("🎥 Trajectory Overlay")
    st.video(h264_path)

    st.subheader("📥 Download Results")
    st.download_button("Download Predictions CSV", data=open(preds_csv, "rb"), file_name="predictions.csv")
    st.download_button("Download Trajectory Video (H.264)", data=open(h264_path, "rb"), file_name="overlayed_trajectories.mp4")

else:
    st.info("⬅️ Please upload both a JSON and MP4 file, then click Run.")
