# Shared data and utilities for the Dash app components
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple

# ---------- Data Source Configuration ----------
# Check if we're in container (files in /app) or local development (files in parent)
def _get_data_path() -> Path:
    """Determine the correct data path for current environment.

    Prefer local cwd if data is present; otherwise fall back to parent.
    This supports both container (WORKDIR=/app) and local runs.
    """
    # Prefer Felipe data if present
    if Path("felipe_data/fid_level_data.csv").exists():
        return Path(".")
    if Path("../felipe_data/fid_level_data.csv").exists():
        return Path("..")

    # Back-compat with older CSV
    if Path("kmeans_results.csv").exists():
        return Path(".")
    if Path("../kmeans_results.csv").exists():
        return Path("..")

    # Default to cwd
    return Path(".")

DATA_PATH = _get_data_path()

def _csv_uri() -> str:
    """Resolve the CSV path/URI.

    Priority:
      1) CSV_URI env var (supports local paths, HTTP(S), or gs:// if gcsfs is installed)
      2) Default to DATA_PATH / "felipe_data" / "fid_level_data.csv"
    """
    csv_env = os.getenv("CSV_URI")
    if csv_env and csv_env.strip():
        return csv_env.strip()
    return str(DATA_PATH / "felipe_data" / "fid_level_data.csv")

def _trajectory_csv() -> str:
    """Get trajectory CSV path"""
    return str(DATA_PATH / "felipe_data" / "trajectory.csv")

def _parquet_glob() -> str:
    # matches participant=<ID>/frames.parquet (unused for Felipe data)
    return str(DATA_PATH / "parquet_data" / "participant=*/frames.parquet")

def _participant_parquet(participant_id: str) -> str:
    # Unused for Felipe data
    return str(DATA_PATH / "parquet_data" / f"participant={participant_id}" / "frames.parquet")

# ---------- FOV / view settings ----------
FOV_QUANTILE   = 0.95   # fixed "Compare" FOV = p95 of half-spans (change to 0.99 if you still see clipping)
MIN_VIEW_HALF  = 10.0   # smallest half-range so tiny tracks remain visible
AUTO_PAD       = 1.10   # padding for auto-fit (10% extra so tips don't touch the frame)
# ----------------------------------------

# ---------- Load points (t-SNE/UMAP + kinematic metrics) ----------
print(f">>> DATA SOURCE: {_csv_uri()}")
KINEMATIC_FEATURES = ["ALH", "BCF", "LIN", "MAD", "STR", "VAP", "VCL", "VSL", "WOB"]
# Additional axis features for P/E plot
AXIS_FEATURES = ["P_axis_byls", "E_axis_byls", "entropy"]
# Felipe data uses 'fid' for track identifier
BASE_COLUMNS = ["tsne_1", "tsne_2", "fid", "subtype_label"]
try:
    _df = pd.read_csv(_csv_uri())
    # Rename fid to track_id and fid to participant_id for compatibility
    if "fid" in _df.columns:
        _df["track_id"] = _df["fid"].astype(str)
        _df["participant_id"] = _df["fid"].astype(str)
    wanted = [c for c in BASE_COLUMNS + KINEMATIC_FEATURES + AXIS_FEATURES + ["track_id", "participant_id"] if c in _df.columns]
    if not wanted:
        raise ValueError("No expected columns found in CSV")
    POINTS = _df[wanted].copy()
    # ensure numeric metrics
    for c in KINEMATIC_FEATURES + AXIS_FEATURES:
        if c in POINTS.columns:
            POINTS[c] = pd.to_numeric(POINTS[c], errors="coerce")
    print(f">>> LOADED {len(POINTS)} points successfully | cols: {list(POINTS.columns)}")
except Exception as e:
    print(f">>> ERROR loading data: {e}")
    POINTS = pd.DataFrame(columns=BASE_COLUMNS + KINEMATIC_FEATURES + ["track_id", "participant_id"])

# ---------- Precompute per-track centers & spans; compute fixed FOV ----------
def build_track_index():
    """
    Returns:
      idx_df (pandas): [participant_id, track_id, cx, cy, half_span]
      view_half_fixed (float): fixed half-range for 'Compare' mode from quantile
      view_half_max   (float): global max half-range (useful if you want 'No-clip fixed')
    """
    try:
        # Load trajectory CSV (Felipe data)
        traj_df = pd.read_csv(_trajectory_csv())
        
        # Group by fid to compute centers and spans
        grouped = traj_df.groupby('fid').agg({
            'x': ['min', 'max'],
            'y': ['min', 'max']
        })
        
        # Calculate centers and half_spans
        centers = []
        for fid in grouped.index:
            xmin, xmax = grouped.loc[fid, ('x', 'min')], grouped.loc[fid, ('x', 'max')]
            ymin, ymax = grouped.loc[fid, ('y', 'min')], grouped.loc[fid, ('y', 'max')]
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            half_span = max(xmax - xmin, ymax - ymin) / 2.0
            centers.append({
                'participant_id': str(fid),
                'track_id': str(fid),
                'cx': cx,
                'cy': cy,
                'half_span': half_span
            })
        
        df = pd.DataFrame(centers)
        
        if df.empty:
            return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)
        
        hs = df["half_span"].to_numpy()
        hs = hs[np.isfinite(hs)]
        if hs.size == 0:
            return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)
        
        view_half_fixed = max(float(np.quantile(hs, FOV_QUANTILE)), MIN_VIEW_HALF)
        view_half_max = max(float(np.max(hs)), MIN_VIEW_HALF)
        
        return df, view_half_fixed, view_half_max
    except Exception as e:
        print(f">>> ERROR building track index: {e}")
        df = pd.DataFrame(columns=["participant_id","track_id","cx","cy","half_span"])
        return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)

# Build track index
TRACK_IDX, VIEW_HALF_FIXED, VIEW_HALF_MAX = build_track_index()

# Fast lookups
CENTER_LOOKUP = {(r.participant_id, r.track_id): (float(r.cx), float(r.cy))
                 for r in TRACK_IDX.itertuples(index=False)}
HALF_LOOKUP   = {(r.participant_id, r.track_id): float(r.half_span)
                 for r in TRACK_IDX.itertuples(index=False)}

# ---------- Data access ----------
# Background loading for trajectory data to avoid blocking hover interactions
_TRAJ_DF = None
_TRAJECTORY_INDEX = None
_TRAJECTORY_LOADING = False
_TRAJECTORY_LOADED = False

def _load_trajectory_data():
    """Load and index trajectory data in background."""
    global _TRAJ_DF, _TRAJECTORY_INDEX, _TRAJECTORY_LOADING, _TRAJECTORY_LOADED
    
    if _TRAJECTORY_LOADING or _TRAJECTORY_LOADED:
        return  # Already loading or loaded
    
    _TRAJECTORY_LOADING = True
    print(">>> Loading trajectory data in background...")
    
    try:
        _TRAJ_DF = pd.read_csv(_trajectory_csv())
        print(f">>> LOADED {len(_TRAJ_DF)} trajectory records successfully")
        
        # Pre-index trajectories by fid for instant lookups
        _TRAJECTORY_INDEX = {}
        for fid, group in _TRAJ_DF.groupby('fid'):
            # Sort by frame_number and select relevant columns
            traj = group.sort_values('frame_number')[['frame_number', 'x', 'y']].copy()
            traj = traj.rename(columns={'frame_number': 'frame_num'})
            _TRAJECTORY_INDEX[fid] = traj.reset_index(drop=True)
        
        print(f">>> INDEXED {len(_TRAJECTORY_INDEX)} trajectories for fast lookup")
        _TRAJECTORY_LOADED = True
        
    except Exception as e:
        print(f">>> ERROR loading trajectory data: {e}")
        _TRAJ_DF = pd.DataFrame(columns=["fid", "frame_number", "x", "y"])
        _TRAJECTORY_INDEX = {}
    finally:
        _TRAJECTORY_LOADING = False

def get_trajectory(track_id: str, participant_id: str) -> pd.DataFrame:
    """Fetch a single track's frames with fallback for loading state."""
    # Start background loading if not already started
    if not _TRAJECTORY_LOADING and not _TRAJECTORY_LOADED:
        _load_trajectory_data()
    
    # If still loading, return empty trajectory with loading message
    if _TRAJECTORY_LOADING:
        return pd.DataFrame(columns=["frame_num", "x", "y"])
    
    try:
        # Convert track_id to fid (Felipe data uses fid)
        fid = int(track_id) if track_id.isdigit() else track_id
        
        # Direct lookup from pre-indexed data
        if _TRAJECTORY_INDEX and fid in _TRAJECTORY_INDEX:
            return _TRAJECTORY_INDEX[fid].copy()
        else:
            return pd.DataFrame(columns=["frame_num", "x", "y"])
        
    except Exception as e:
        print(f"Error reading trajectory data for {participant_id}/{track_id}: {e}")
        return pd.DataFrame(columns=["frame_num", "x", "y"])
