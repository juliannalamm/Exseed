# scripts/parquet_convert.py
from pathlib import Path
import json
import pandas as pd
import polars as pl

# ====== HARD-CODED PATHS ======
JSON_DIR = Path("../combined_views_tracks")   # input JSONs (one per participant)
OUT_DIR  = Path("../parquet_data")            # output folder
WRITE_TRACK_METRICS = True
# ==============================

def json_to_df(json_file_path: str, participant_id: str):
    """
    Convert JSON to:
      - track_df: one row per track with motility params
      - frame_df: frame-level coords [participant_id, track_id, frame_num, x, y]
    Assumes schema:
      data["Tracks"][i]["Motility Parameters"]  (ALH, BCF, LIN, MAD, STR, VAP, VCL, VSL, WOB, "Track Length (frames)")
      data["Tracks"][i]["Track"] -> { "Frame N": [x, y], ... }  (N is an integer)
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)

    tracks = data.get("Tracks", [])
    track_rows, frame_rows = [], []

    for i, track in enumerate(tracks):
        mot = track.get("Motility Parameters", {}) or {}
        tid = f"{participant_id}_track_{i}"

        # track-level row (always keep)
        track_rows.append({
            "participant_id": participant_id,
            "track_id": tid,
            "ALH": mot.get("ALH"),
            "BCF": mot.get("BCF"),
            "LIN": mot.get("LIN"),
            "MAD": mot.get("MAD"),
            "STR": mot.get("STR"),
            "track_length_frames": mot.get("Track Length (frames)"),
            "VAP": mot.get("VAP"),
            "VCL": mot.get("VCL"),
            "VSL": mot.get("VSL"),
            "WOB": mot.get("WOB"),
        })

        # frame-level rows (assumes dict: "Frame N": [x, y])
        coords = track.get("Track", {}) or {}
        for frame_name, xy in coords.items():
            # expects "Frame N"
            frame_num = int(str(frame_name).replace("Frame", "").strip())
            frame_rows.append({
                "participant_id": participant_id,
                "track_id": tid,
                "frame_num": frame_num,
                "x": xy[0],
                "y": xy[1],
            })

    track_df = pd.DataFrame(track_rows)
    frame_df = pd.DataFrame(frame_rows)
    if not frame_df.empty:
        frame_df = frame_df.sort_values(["track_id", "frame_num"]).reset_index(drop=True)
    return track_df, frame_df

def run_once(json_dir: Path, out_dir: Path, write_track_metrics: bool = True):
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON_DIR not found: {json_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tracks = []
    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in {json_dir}")

    for jp in json_paths:
        # Accept "P01.json" or "P01_combined.json" → participant_id "P01"
        participant_id = jp.stem.replace("_combined", "")
        print(f"[parse] {jp.name} -> participant={participant_id}")

        track_df, frame_df = json_to_df(str(jp), participant_id)

        if frame_df.empty:
            print(f"  [skip] no frame rows parsed for {participant_id}; no Parquet written.")
        else:
            part_path = out_dir / f"participant={participant_id}"
            part_path.mkdir(parents=True, exist_ok=True)
            pl.from_pandas(frame_df).write_parquet(part_path / "frames.parquet")
            print(f"  wrote {part_path/'frames.parquet'} (rows={len(frame_df)})")

        if write_track_metrics and not track_df.empty:
            all_tracks.append(track_df)

    if write_track_metrics and all_tracks:
        combined = pd.concat(all_tracks, ignore_index=True)
        combined_out = out_dir.parent / "track_metrics_from_json.parquet"
        combined.to_parquet(combined_out, index=False)
        print(f"[ok] wrote {combined_out} (rows={len(combined)})")

    print("[done] Partitioned Parquet frames ready.")

if __name__ == "__main__":
    run_once(JSON_DIR, OUT_DIR, WRITE_TRACK_METRICS)
