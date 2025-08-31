# Shared data and utilities for the Dash app components
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# ---------- Hardcoded paths ----------
POINTS_CSV  = Path("train_track_df.csv")     # must contain: umap_1, umap_2, track_id, participant_id, subtype_label
FRAMES_ROOT = Path("parquet_data")           # partitions: participant=<ID>/frames.parquet
# -------------------------------------

# ---------- FOV / view settings ----------
FOV_QUANTILE   = 0.95   # fixed "Compare" FOV = p95 of half-spans (change to 0.99 if you still see clipping)
MIN_VIEW_HALF  = 10.0   # smallest half-range so tiny tracks remain visible
AUTO_PAD       = 1.10   # padding for auto-fit (10% extra so tips don't touch the frame)
# ----------------------------------------

# ---------- Load UMAP points ----------
POINTS = pd.read_csv(POINTS_CSV)[["umap_1", "umap_2", "track_id", "participant_id", "subtype_label"]]

# ---------- Precompute per-track centers & spans; compute fixed FOV ----------
def build_track_index(frames_root: Path):
    """
    Returns:
      idx_df (pandas): [participant_id, track_id, cx, cy, half_span]
      view_half_fixed (float): fixed half-range for 'Compare' mode from quantile
      view_half_max   (float): global max half-range (useful if you want 'No-clip fixed')
    """
    pattern = str(frames_root / "participant=*/frames.parquet")
    
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

# Use local paths for now
TRACK_IDX, VIEW_HALF_FIXED, VIEW_HALF_MAX = build_track_index(FRAMES_ROOT)

# Fast lookups
CENTER_LOOKUP = {(r.participant_id, r.track_id): (float(r.cx), float(r.cy))
                 for r in TRACK_IDX.itertuples(index=False)}
HALF_LOOKUP   = {(r.participant_id, r.track_id): float(r.half_span)
                 for r in TRACK_IDX.itertuples(index=False)}

# ---------- Data access ----------
def get_trajectory(track_id: str, participant_id: str) -> pd.DataFrame:
    """Fetch a single track's frames from the participant Parquet."""
    p = FRAMES_ROOT / f"participant={participant_id}" / "frames.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["frame_num", "x", "y"])
    df = pl.read_parquet(p)
    out = (
        df.filter(pl.col("track_id") == track_id)
          .sort("frame_num")
          .select(["frame_num", "x", "y"])
          .to_pandas()
    )
    return out
