import os
import json
import tempfile
import subprocess
from typing import Dict, Tuple, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="Sperm Motility Explorer for ExSeed", page_icon="🎥", layout="wide")
st.title("Sperm Motility Explorer for ExSeed")

# ---------------- Sidebar ----------------
st.sidebar.header("📁 Upload")
uploaded_video = st.sidebar.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
    help="Any common video format; will be converted to H.264 for browser playback."
)
uploaded_json = st.sidebar.file_uploader(
    "Upload tracks JSON",
    type=["json"],
    help="JSON with Tracks -> Track -> 'Frame N': [x, y] and optional 'Framerate (FPS)'."
)

st.sidebar.header("⚙️ Overlay options")
trail_len = st.sidebar.slider("Trail length (frames)", 0, 200, 40, step=5)
draw_ids = st.sidebar.checkbox("Draw track IDs", True)
point_radius = st.sidebar.slider("Point radius", 1, 8, 3)
line_thickness = st.sidebar.slider("Line thickness", 1, 6, 2)
coords_normalized = st.sidebar.checkbox("JSON coordinates are normalized (0–1)", False)
frame_base = st.sidebar.selectbox("Frame numbering in JSON", ["0-indexed", "1-indexed"], index=0)
limit_seconds = st.sidebar.number_input("Limit to first N seconds (0 = full video)", min_value=0, value=0, step=1)

run = st.sidebar.button("Overlay trajectories")

# ---------------- Helpers ----------------
def persist_temp(upload, default_suffix: str) -> str:
    suffix = os.path.splitext(upload.name)[1] or default_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload.seek(0)
        tmp.write(upload.getbuffer())
        return tmp.name

def convert_to_h264(in_path: str) -> str:
    """Convert to H.264/AAC MP4 to ensure browser playback."""
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-acodec", "aac", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def parse_tracks_json(json_path: str, frame_base_1: bool) -> pd.DataFrame:
    """
    Returns frame_df with columns: frame_num (int), track_id (str), x (float), y (float).
    Assumes schema:
      {
        "Tracks": [
          {
            "Track": {"Frame 0": [x, y], "Frame 1": [x, y], ...},
            "Motility Parameters": {...}  # optional
          }, ...
        ],
        "Framerate (FPS)": 50.0  # optional
      }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows: List[Dict] = []
    for i, t in enumerate(data.get("Tracks", [])):
        tid = f"track_{i}"
        track_coords = (t or {}).get("Track", {}) or {}
        for k, xy in track_coords.items():
            # k is like "Frame 12"
            try:
                fnum = int(str(k).strip().split()[-1])
            except Exception:
                continue
            if frame_base_1:
                fnum -= 1  # shift to 0-based
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            x, y = float(xy[0]), float(xy[1])
            rows.append({"frame_num": fnum, "track_id": tid, "x": x, "y": y})

    frame_df = pd.DataFrame(rows)
    if frame_df.empty:
        raise ValueError("Parsed JSON produced no frame rows. Check schema/keys.")
    return frame_df.sort_values(["frame_num", "track_id"]).reset_index(drop=True)

def track_color(i: int) -> Tuple[int, int, int]:
    """Distinct BGR colors."""
    palette = [
        (40, 39, 214), (44, 160, 44), (189, 103, 148), (180, 119, 31),
        (0, 215, 255), (255, 144, 30), (128, 0, 255), (128, 128, 128),
        (0, 140, 255), (50, 205, 50), (147, 20, 255), (0, 69, 255),
    ]
    return palette[i % len(palette)]

def build_track_styles(frame_df: pd.DataFrame) -> Dict[str, Dict]:
    styles = {}
    for i, tid in enumerate(sorted(frame_df["track_id"].unique())):
        styles[tid] = {"color": track_color(i)}
    return styles

def overlay_trajectories(
    video_path: str,
    frame_df: pd.DataFrame,
    styles: Dict[str, Dict],
    trail_len: int,
    draw_ids: bool,
    point_radius: int,
    line_thickness: int,
    coords_normalized: bool,
    limit_seconds: int,
) -> bytes:
    """
    Draw trajectories onto the video and return MP4 bytes.
    Expects frame_df columns: frame_num, track_id, x, y.
    If coords_normalized=True, scales x,y by (width,height).
    """
    req = {"frame_num", "track_id", "x", "y"}
    if not req.issubset(frame_df.columns):
        missing = req - set(frame_df.columns)
        raise ValueError(f"frame_df missing required columns: {missing}")

    fdf = frame_df.copy()
    fdf["frame_num"] = fdf["frame_num"].astype(int)
    fdf["track_id"] = fdf["track_id"].astype(str)

    # group by frame for fast lookup
    by_frame = dict(tuple(fdf.groupby("frame_num")))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = total_frames if limit_seconds <= 0 else min(total_frames, int(round(fps * limit_seconds)))

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_tmp, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter.")

    history: Dict[str, List[Tuple[int, int]]] = {}

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        rows = by_frame.get(frame_idx, None)
        if rows is not None:
            for _, r in rows.iterrows():
                tid = r["track_id"]
                x, y = float(r["x"]), float(r["y"])
                if coords_normalized:
                    x *= width
                    y *= height

                # Skip obviously out of bounds
                if x < -5 or y < -5 or x > width + 5 or y > height + 5:
                    continue

                color = styles.get(tid, {"color": (255, 255, 255)})["color"]

                pts = history.setdefault(tid, [])
                pts.append((int(round(x)), int(round(y))))
                if trail_len > 0 and len(pts) > trail_len:
                    history[tid] = pts[-trail_len:]
                    pts = history[tid]

                # Trail
                if trail_len > 0 and len(pts) >= 2:
                    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=line_thickness)

                # Current point
                cv2.circle(frame, (int(round(x)), int(round(y))), point_radius, color, thickness=-1)

                # Label
                if draw_ids:
                    cv2.putText(frame, tid, (int(round(x)) + 6, int(round(y)) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Read bytes and clean
    with open(out_tmp, "rb") as f:
        data = f.read()
    try:
        os.remove(out_tmp)
    except OSError:
        pass
    return data

# ---------------- Main UI flow ----------------
converted_video_path = None
if uploaded_video:
    st.subheader("🎬 Video preview")
    try:
        raw_path = persist_temp(uploaded_video, default_suffix=".mp4")
        converted_video_path = convert_to_h264(raw_path)
        st.video(converted_video_path, start_time=0)
        # keep files around during session; we won't delete immediately
    except Exception as e:
        st.error(f"Video conversion failed: {e}")

else:
    st.info("Upload a video to begin.")

if uploaded_video and uploaded_json and run:
    try:
        with st.spinner("Parsing tracks JSON..."):
            json_path = persist_temp(uploaded_json, default_suffix=".json")
            frame_df = parse_tracks_json(json_path, frame_base_1=(frame_base == "1-indexed"))
            os.remove(json_path)

        st.write("Frame data (head):")
        st.dataframe(frame_df.head(), use_container_width=True)

        with st.spinner("Building per-track styles..."):
            styles = build_track_styles(frame_df)

        with st.spinner("Overlaying trajectories on video..."):
            out_bytes = overlay_trajectories(
                video_path=converted_video_path or raw_path,
                frame_df=frame_df,
                styles=styles,
                trail_len=trail_len,
                draw_ids=draw_ids,
                point_radius=point_radius,
                line_thickness=line_thickness,
                coords_normalized=coords_normalized,
                limit_seconds=limit_seconds,
            )

        st.success("Overlay complete.")
        st.subheader("Result")
        st.video(out_bytes)
        st.download_button("Download overlay video", out_bytes, file_name="overlay_trajectories.mp4", mime="video/mp4")

    except Exception as e:
        st.error(f"Failed to overlay trajectories: {e}")
        st.exception(e)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("🎥 Video Display App | Built with Streamlit")
