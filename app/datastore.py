# Shared data and utilities for the Dash app components
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# ---------- Data Source Configuration ----------
# Check if we're in container (files in /app) or local development (files in parent)
def _get_data_path() -> Path:
    """Determine the correct data path for current environment."""
    if Path("tsne_and_umap_k5.csv").exists():
        return Path(".")  # Container
    else:
        return Path("..")  # Local development

DATA_PATH = _get_data_path()

def _csv_uri() -> str:
    return str(DATA_PATH / "tsne_and_umap_k5.csv")

def _parquet_glob() -> str:
    # matches participant=<ID>/frames.parquet
    return str(DATA_PATH / "parquet_data" / "participant=*/frames.parquet")

def _participant_parquet(participant_id: str) -> str:
    return str(DATA_PATH / "parquet_data" / f"participant={participant_id}" / "frames.parquet")

# ---------- FOV / view settings ----------
FOV_QUANTILE   = 0.95   # fixed "Compare" FOV = p95 of half-spans (change to 0.99 if you still see clipping)
MIN_VIEW_HALF  = 10.0   # smallest half-range so tiny tracks remain visible
AUTO_PAD       = 1.10   # padding for auto-fit (10% extra so tips don't touch the frame)
# ----------------------------------------

# ---------- Load UMAP points ----------
print(f">>> DATA SOURCE: {_csv_uri()}")
try:
    POINTS = pd.read_csv(_csv_uri())[["umap_1", "umap_2", "tsne_1", "tsne_2", "track_id", "participant_id", "subtype_label", 
                                      "P_rapid_progressive", "P_immotile", "P_nonprogressive", "P_progressive", "P_cluster_4"]]
    print(f">>> LOADED {len(POINTS)} points successfully")
except Exception as e:
    print(f">>> ERROR loading data: {e}")
    # Fallback to empty dataframe
    POINTS = pd.DataFrame(columns=["umap_1", "umap_2", "tsne_1", "tsne_2", "track_id", "participant_id", "subtype_label",
                                   "P_rapid_progressive", "P_immotile", "P_nonprogressive", "P_progressive", "P_cluster_4"])

# ---------- Precompute per-track centers & spans; compute fixed FOV ----------
def build_track_index():
    """
    Returns:
      idx_df (pandas): [participant_id, track_id, cx, cy, half_span]
      view_half_fixed (float): fixed half-range for 'Compare' mode from quantile
      view_half_max   (float): global max half-range (useful if you want 'No-clip fixed')
    """
    pattern = _parquet_glob()
    
    lf = pl.scan_parquet(pattern)

    agg = (
        lf.group_by(["participant_id", "track_id"])
          .agg([
              pl.col("x").min().alias("xmin"),
              pl.col("x").max().alias("xmax"),
              pl.col("y").min().alias("ymin"),
              pl.col("y").max().alias("ymax"),
          ])
          .with_columns([
              ((pl.col("xmin") + pl.col("xmax")) / 2.0).alias("cx"),
              ((pl.col("ymin") + pl.col("ymax")) / 2.0).alias("cy"),
              (pl.max_horizontal(pl.col("xmax") - pl.col("xmin"),
                                 pl.col("ymax") - pl.col("ymin")) / 2.0).alias("half_span"),
          ])
          .select(["participant_id", "track_id", "cx", "cy", "half_span"])
    )

    tracks = agg.collect()
    if tracks.is_empty():
        df = pd.DataFrame(columns=["participant_id","track_id","cx","cy","half_span"])
        return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)

    df = tracks.to_pandas()
    hs = df["half_span"].to_numpy()
    hs = hs[np.isfinite(hs)]
    if hs.size == 0:
        return df, float(MIN_VIEW_HALF), float(MIN_VIEW_HALF)

    view_half_fixed = max(float(np.quantile(hs, FOV_QUANTILE)), MIN_VIEW_HALF)
    view_half_max   = max(float(hs.max()), MIN_VIEW_HALF)
    return df, view_half_fixed, view_half_max

# Build track index
TRACK_IDX, VIEW_HALF_FIXED, VIEW_HALF_MAX = build_track_index()

# Fast lookups
CENTER_LOOKUP = {(r.participant_id, r.track_id): (float(r.cx), float(r.cy))
                 for r in TRACK_IDX.itertuples(index=False)}
HALF_LOOKUP   = {(r.participant_id, r.track_id): float(r.half_span)
                 for r in TRACK_IDX.itertuples(index=False)}

# ---------- Data access ----------
def get_trajectory(track_id: str, participant_id: str) -> pd.DataFrame:
    """Fetch a single track's frames from the participant Parquet."""
    p = _participant_parquet(participant_id)
    try:
        df = pl.read_parquet(p)
        out = (
            df.filter(pl.col("track_id") == track_id)
              .sort("frame_num")
              .select(["frame_num", "x", "y"])
              .to_pandas()
        )
        return out
    except Exception as e:
        print(f"Error reading trajectory data for {participant_id}/{track_id}: {e}")
        return pd.DataFrame(columns=["frame_num", "x", "y"])
