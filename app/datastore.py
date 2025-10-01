# Shared data and utilities for the Dash app components
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple
import time
import pickle

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

def _points_parquet() -> Path:
    return DATA_PATH / "felipe_data" / "fid_level_data.parquet"

def _trajectory_csv() -> str:
    """Get trajectory CSV path"""
    return str(DATA_PATH / "felipe_data" / "trajectory.csv")

def _trajectory_index_parquet() -> Path:
    """Preferred precomputed trajectory index parquet file."""
    return DATA_PATH / "felipe_data" / "trajectory_index.parquet"

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
    # Prefer Parquet if available for faster startup
    if _points_parquet().exists():
        print(f">>> POINTS: using Parquet {_points_parquet()}")
        _df = pd.read_parquet(_points_parquet())
    else:
        try:
            print(f">>> POINTS: using CSV (pyarrow) {_csv_uri()}")
            _df = pd.read_csv(_csv_uri(), engine="pyarrow")
        except Exception:
            print(f">>> POINTS: using CSV (default engine) {_csv_uri()}")
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
        # Load trajectories (prefer precomputed Parquet index if present)
        if _trajectory_index_parquet().exists():
            print(f">>> TRAJ: using Parquet index {_trajectory_index_parquet()}")
            traj_df = pd.read_parquet(_trajectory_index_parquet())
            if 'frame_number' in traj_df.columns and 'frame_num' not in traj_df.columns:
                traj_df = traj_df.rename(columns={'frame_number': 'frame_num'})
        else:
            try:
                print(f">>> TRAJ: using CSV (pyarrow) {_trajectory_csv()}")
                traj_df = pd.read_csv(_trajectory_csv(), engine="pyarrow")
            except Exception:
                print(f">>> TRAJ: using CSV (default engine) {_trajectory_csv()}")
                traj_df = pd.read_csv(_trajectory_csv())
            if 'frame_number' in traj_df.columns and 'frame_num' not in traj_df.columns:
                traj_df = traj_df.rename(columns={'frame_number': 'frame_num'})
        
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
# Load all trajectory data once and build an in-memory index for instant lookups
try:
    t0_total = time.perf_counter()
    # Prefer binary index, then parquet, then CSV
    pkl = DATA_PATH / "felipe_data" / "trajectory_index.pkl"
    if pkl.exists():
        print(f">>> TRAJ INDEX: using Binary Index {pkl}")
        t0 = time.perf_counter()
        with open(pkl, 'rb') as f:
            _TRAJECTORY_INDEX = pickle.load(f)
        t1 = time.perf_counter()
        _TRAJ_DF = pd.DataFrame(columns=["fid","frame_num","x","y"])  # minimal placeholder
        print(f">>> TRAJ INDEX (pkl) loaded in {t1-t0:.2f}s | tracks={len(_TRAJECTORY_INDEX)}")
    else:
        if _trajectory_index_parquet().exists():
            print(f">>> TRAJ INDEX: building from Parquet {_trajectory_index_parquet()}")
            t0 = time.perf_counter()
            _TRAJ_DF = pd.read_parquet(_trajectory_index_parquet())
        else:
            try:
                print(f">>> TRAJ INDEX: building from CSV (pyarrow) {_trajectory_csv()}")
                _TRAJ_DF = pd.read_csv(_trajectory_csv(), engine="pyarrow")
            except Exception:
                print(f">>> TRAJ INDEX: building from CSV (default engine) {_trajectory_csv()}")
                _TRAJ_DF = pd.read_csv(_trajectory_csv())
        if 'frame_number' in _TRAJ_DF.columns and 'frame_num' not in _TRAJ_DF.columns:
            _TRAJ_DF = _TRAJ_DF.rename(columns={'frame_number': 'frame_num'})
        _TRAJECTORY_INDEX = {}
        for fid, grp in _TRAJ_DF.groupby('fid'):
            _TRAJECTORY_INDEX[fid] = grp.sort_values('frame_num')[["frame_num", 'x', 'y']].reset_index(drop=True)
        t1 = time.perf_counter()
        print(f">>> TRAJ INDEX built in {t1-t0:.2f}s | tracks={len(_TRAJECTORY_INDEX)} | rows={len(_TRAJ_DF)}")
    print(f">>> TRAJ TOTAL init took {time.perf_counter()-t0_total:.2f}s")
except Exception as e:
    print(f">>> ERROR loading trajectory data: {e}")
    _TRAJ_DF = pd.DataFrame(columns=["fid", "frame_num", "x", "y"])
    _TRAJECTORY_INDEX = {}

def get_trajectory(track_id: str, participant_id: str) -> pd.DataFrame:
    """Fetch a single track's frames using prebuilt index for O(1) lookup."""
    try:
        # Convert track_id to fid (Felipe data uses fid)
        fid = int(track_id) if isinstance(track_id, str) and track_id.isdigit() else track_id
        # Direct index lookup first
        if fid in _TRAJECTORY_INDEX:
            val = _TRAJECTORY_INDEX[fid]
            # If binary index provides ndarray, convert to DataFrame
            if isinstance(val, pd.DataFrame):
                return val.copy()
            else:
                # Expect shape (N,3): frame_num, x, y
                return pd.DataFrame(val, columns=["frame_num", "x", "y"]).copy()
        # Fallback to filtering if for some reason not in index
        if not _TRAJ_DF.empty:
            traj = _TRAJ_DF[_TRAJ_DF['fid'] == fid].copy()
            if traj.empty:
                return pd.DataFrame(columns=["frame_num", "x", "y"])
            if 'frame_number' in traj.columns:
                traj = traj.rename(columns={'frame_number': 'frame_num'})
            traj = traj.sort_values('frame_num')[['frame_num', 'x', 'y']]
            return traj.reset_index(drop=True)
        return pd.DataFrame(columns=["frame_num", "x", "y"])
    except Exception as e:
        print(f"Error reading trajectory data for {participant_id}/{track_id}: {e}")
        return pd.DataFrame(columns=["frame_num", "x", "y"])
 