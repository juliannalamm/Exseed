#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import plotly.express as px 


# =========================
# Config (edit if needed)
# =========================
MODEL_PATH = "trained_gmm_model.pkl"  # expected keys: gmm, scaler, cluster_to_label, features, (optional) umap_model
TRAIL_LEN = 40
LINE_THICKNESS = 2
POINT_RADIUS = 3
SHOW_IDS = True
LIMIT_SECONDS = 0     # 0 = full video
AUTO_DETECT_FRAME_BASE = True  # if min frame == 1 and no 0 seen, we shift to 0-based automatically


# =========================
# 1) JSON → DataFrames
# =========================
def json_to_df(json_file_path: str, participant_id: str):
    """
    Convert your JSON file to:
      - track_df: one row per track with motility params
      - frame_df: frame-level coords with columns [participant_id, track_id, frame_num, x, y]
    Expected JSON keys:
      data["Tracks"][i]["Motility Parameters"]  (ALH, BCF, LIN, MAD, STR, VAP, VCL, VSL, WOB, "Track Length (frames)")
      data["Tracks"][i]["Track"] -> { "Frame N": [x, y], ... }
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    if "Tracks" not in data:
        raise ValueError("JSON missing 'Tracks' key.")

    track_rows, frame_rows = [], []

    for i, track in enumerate(data["Tracks"]):
        mot = track.get("Motility Parameters", {})
        tid = f"{participant_id}_track_{i}"

        # track-level row
        track_rows.append({
            'participant_id': participant_id,
            'track_id': tid,
            'ALH': mot.get('ALH'),
            'BCF': mot.get('BCF'),
            'LIN': mot.get('LIN'),
            'MAD': mot.get('MAD'),
            'STR': mot.get('STR'),
            'track_length_frames': mot.get('Track Length (frames)'),
            'VAP': mot.get('VAP'),
            'VCL': mot.get('VCL'),
            'VSL': mot.get('VSL'),
            'WOB': mot.get('WOB'),
        })

        # frame-level rows
        coords = track.get('Track', {}) or {}
        for frame_name, xy in coords.items():
            try:
                frame_num = int(str(frame_name).replace('Frame ', '').strip())
            except Exception:
                continue
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            frame_rows.append({
                'participant_id': participant_id,
                'track_id': tid,
                'frame_num': frame_num,
                'x': xy[0],
                'y': xy[1],
            })

    track_df = pd.DataFrame(track_rows)
    frame_df = pd.DataFrame(frame_rows)
    if frame_df.empty:
        raise ValueError("Parsed JSON produced no frame rows—check schema/keys.")
    frame_df = frame_df.sort_values(["frame_num", "track_id"]).reset_index(drop=True)
    return track_df, frame_df


# =========================
# 2) GMM Prediction (+UMAP)
# =========================
def predict_sperm_motility(new_data: pd.DataFrame,
                           model_path: str = MODEL_PATH,
                           include_umap: bool = True) -> pd.DataFrame:
    """
    Predict sperm motility subtypes (cluster_id → subtype_label) with optional UMAP embedding.
    The model pickle should contain: {'gmm','scaler','cluster_to_label','features', 'umap_model' (optional)}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    gmm = model_data['gmm']
    scaler = model_data['scaler']
    cluster_to_label = model_data['cluster_to_label']
    features = model_data['features']
    umap_model = model_data.get('umap_model', None)

    new_df = new_data.copy()
    missing = set(features) - set(new_df.columns)
    if missing:
        raise ValueError(f"Missing features in new_data: {missing}")

    X_scaled = scaler.transform(new_df[features])
    cluster_labels = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    result_df = new_df.copy()
    result_df['cluster_id'] = cluster_labels
    result_df['subtype_label'] = result_df['cluster_id'].map(cluster_to_label)

    for i in range(gmm.n_components):
        result_df[f'P_cluster_{i}'] = probabilities[:, i]
        if i in cluster_to_label:
            result_df[f'P_{cluster_to_label[i]}'] = probabilities[:, i]

    if include_umap:
        # First, try to get UMAP coordinates from existing training data
        try:
            import pandas as pd
            if os.path.exists("train_track_df.csv"):
                train_df = pd.read_csv("train_track_df.csv")
                
                # Check if this participant exists in training data
                participant_id = new_data_df['participant_id'].iloc[0] if 'participant_id' in new_data_df.columns else None
                
                if participant_id and participant_id in train_df['participant_id'].values:
                    # Get existing UMAP coordinates for this participant
                    existing_data = train_df[train_df['participant_id'] == participant_id]
                    
                    # Match tracks by track_id
                    merged_data = new_data_df.merge(
                        existing_data[['track_id', 'umap_1', 'umap_2']], 
                        on='track_id', 
                        how='left'
                    )
                    
                    # Use existing coordinates where available
                    if not merged_data['umap_1'].isna().all():
                        result_df['umap_1'] = merged_data['umap_1']
                        result_df['umap_2'] = merged_data['umap_2']
                        print(f"✅ Using existing UMAP coordinates for {participant_id} from training data")
                        return result_df
                    else:
                        print(f"⚠️ Participant {participant_id} found in training data but no matching track_ids")
                else:
                    print(f"ℹ️ Participant {participant_id} not found in training data, using UMAP model")
        except Exception as e:
            print(f"⚠️ Error accessing training data: {e}")
        
        # Fallback to UMAP model - use SCALED features to match notebook
        # Define X_scaled here so it's available in all cases
        X_scaled = scaler.transform(new_data[features])
        
        if umap_model is not None:
            try:
                # Use scaled features for UMAP to match the notebook training
                umap_embedding = umap_model.transform(X_scaled)
                result_df['umap_1'] = umap_embedding[:, 0]
                result_df['umap_2'] = umap_embedding[:, 1]
                
                print("Using pre-trained UMAP model with raw features for consistent coordinate system")
                
                # Check the embedding ranges for debugging
                umap_range_1 = result_df['umap_1'].max() - result_df['umap_1'].min()
                umap_range_2 = result_df['umap_2'].max() - result_df['umap_2'].min()
                print(f"UMAP ranges for new data: X={umap_range_1:.3f}, Y={umap_range_2:.3f}")
                print(f"UMAP coordinate range: X=[{result_df['umap_1'].min():.3f}, {result_df['umap_1'].max():.3f}], Y=[{result_df['umap_2'].min():.3f}, {result_df['umap_2'].max():.3f}]")
                    
            except Exception as e:
                print(f"Error with pre-trained UMAP: {e}. Retraining on new data...")
                import umap
                new_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
                new_embedding = new_umap.fit_transform(X_scaled)
                result_df['umap_1'] = new_embedding[:, 0]
                result_df['umap_2'] = new_embedding[:, 1]
        else:
            # No pre-trained UMAP model, train new one
            print("No pre-trained UMAP model found, training new one...")
            import umap
            new_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            new_embedding = new_umap.fit_transform(X_scaled)
            result_df['umap_1'] = new_embedding[:, 0]
            result_df['umap_2'] = new_embedding[:, 1]

    return result_df


# =========================
# 3) UMAP Plot
# =========================
def plot_umap_with_predictions(predictions_df: pd.DataFrame, output_path: str):
    """
    Save a static UMAP scatter colored by cluster_id.
    Expected columns: ['umap_1','umap_2','cluster_id'].
    """
    if 'umap_1' not in predictions_df.columns or 'umap_2' not in predictions_df.columns:
        print("UMAP coordinates not found (include_umap=True + model has umap_model). Skipping UMAP plot.")
        return

    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple'}
    cluster_to_label = {0: 'nonprogressive', 1: 'immotile', 2: 'vigorous', 3: 'progressive'}

    plt.figure(figsize=(10, 8))
    for cid in predictions_df['cluster_id'].unique():
        sub = predictions_df[predictions_df['cluster_id'] == cid]
        plt.scatter(sub['umap_1'], sub['umap_2'],
                    c=colors.get(cid, 'gray'),
                    label=cluster_to_label.get(cid, f'cluster {cid}'),
                    alpha=0.7, s=50)

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Projection of New Sperm Motility Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UMAP plot saved to {output_path}")


# =========================
# 4) Overlay Trajectories on Video (OpenCV)
# =========================
def overlay_trajectories_on_video(
    frame_df: pd.DataFrame,
    track_df: pd.DataFrame,
    video_path: str,
    output_path: str,
    trail_len: int = TRAIL_LEN,
    line_thickness: int = LINE_THICKNESS,
    point_radius: int = POINT_RADIUS,
    show_ids: bool = SHOW_IDS,
    limit_seconds: int = LIMIT_SECONDS,
) -> str:
    """
    Overlay trajectory trails on the actual video, colored by GMM cluster.

    frame_df: columns ['track_id','frame_num','x','y'] in pixel coords (origin top-left).
    track_df: includes ['track_id','cluster_id','subtype_label'] (from predict_sperm_motility).
    """
    # Cluster → BGR color mapping (OpenCV)
    # 0: vigorous (green), 1: immotile (blue), 2: vigorous (red), 3: progressive (purple)
    cluster_colors_bgr = {
        0: (255, 0, 0),      # blue
        1: (0, 0, 255),      # red
        2: (0,100,0),      # red
        3: (0, 90, 178),    # orange
    }

    fdf = frame_df.copy()

    # Auto-detect frame base if requested (shift 1-based → 0-based)
    if AUTO_DETECT_FRAME_BASE:
        # If there is no frame 0 but min frame is 1 or greater, shift down by 1
        unique_frames = set(int(v) for v in fdf['frame_num'])
        if 0 not in unique_frames and min(unique_frames) == 1:
            fdf['frame_num'] = fdf['frame_num'].astype(int) - 1
    else:
        fdf['frame_num'] = fdf['frame_num'].astype(int)

    by_frame = dict(tuple(fdf.groupby('frame_num', sort=True)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = total_frames if limit_seconds <= 0 else min(total_frames, int(round(fps * limit_seconds)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Streamlit/browser-friendly; re-encode to H.264 later if needed
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter for output.")

    history = {}  # {track_id: [(x,y), ...]}

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # Update with any points at this frame
        rows = by_frame.get(frame_idx, None)
        if rows is not None:
            for _, r in rows.iterrows():
                tid = r['track_id']
                x, y = float(r['x']), float(r['y'])

                # Skip wildly out-of-bounds coords (JSON/video mismatch guard)
                if x < -5 or y < -5 or x > width + 5 or y > height + 5:
                    continue

                pts = history.setdefault(tid, [])
                pts.append((int(round(x)), int(round(y))))
                if trail_len > 0 and len(pts) > trail_len:
                    history[tid] = pts[-trail_len:]
                    pts = history[tid]

        # Draw trails + current points for all active tracks
        for tid, pts in history.items():
            row = track_df.loc[track_df['track_id'] == tid]
            if not row.empty and pd.notna(row.iloc[0]['cluster_id']):
                cluster = int(row.iloc[0]['cluster_id'])
                color = cluster_colors_bgr.get(cluster, (255, 255, 255))
                subtype = str(row.iloc[0]['subtype_label']) if 'subtype_label' in row.columns else ''
            else:
                color = (255, 255, 255)
                subtype = ''

            # Trail
            if trail_len > 0 and len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=line_thickness)

            # Current point + label
            cx, cy = pts[-1]
            cv2.circle(frame, (cx, cy), point_radius, color, thickness=-1)
            if show_ids:
                label = subtype if subtype else ""
                if label:
                    cv2.putText(frame, label, (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


# =========================
# 5) Main (only two args)
# =========================
def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <json_file> <video_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    video_path = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        sys.exit(1)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    participant_id = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.join(os.path.dirname(json_path), f"{participant_id}_outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Participant: {participant_id}")
    print("Parsing JSON → DataFrames…")
    track_df, frame_df = json_to_df(json_path, participant_id)

    print("Predicting clusters/subtypes with GMM…")
    preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=True)

    # Save predictions
    preds_csv = os.path.join(out_dir, f"{participant_id}_predictions.csv")
    preds.to_csv(preds_csv, index=False)
    print(f"Predictions CSV: {preds_csv}")

    # UMAP plot (only if umap_model was in the pickle)
    umap_png = os.path.join(out_dir, f"{participant_id}_umap.png")
    plot_umap_with_predictions(preds, umap_png)

    print("Overlaying trajectories on video…")
    overlay_mp4 = os.path.join(out_dir, f"{participant_id}_trajectories_overlay.mp4")
    overlay_trajectories_on_video(
        frame_df=frame_df,
        track_df=preds,
        video_path=video_path,
        output_path=overlay_mp4,
        trail_len=TRAIL_LEN,
        line_thickness=LINE_THICKNESS,
        point_radius=POINT_RADIUS,
        show_ids=SHOW_IDS,
        limit_seconds=LIMIT_SECONDS,
    )

    print("\n=== Summary ===")
    print(f"UMAP PNG: {umap_png if os.path.exists(umap_png) else '(skipped)'}")
    print(f"Overlay MP4: {overlay_mp4}")
    print("Cluster distribution:")
    if 'subtype_label' in preds.columns:
        print(preds['subtype_label'].value_counts())
    else:
        print(preds['cluster_id'].value_counts())


if __name__ == "__main__":
    # Guard: OpenCV available?
    if not hasattr(cv2, "VideoCapture"):
        raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")
    main()
