#!/usr/bin/env python3
"""
Safely merge extra columns (experiment_media, umap_1, umap_2) from
felipe_data/fid_level_w_drug.csv into the existing
felipe_data/fid_level_data.parquet without changing any app wiring.

Steps:
- Load existing parquet (baseline used by the app)
- Load new CSV with drug columns
- Validate counts and keys
- Left-merge on fid with only the desired new columns
- Backup original parquet (.parquet.bak) and overwrite parquet
"""

import pandas as pd
from pathlib import Path
import shutil


BASE_PARQUET = Path("felipe_data/fid_level_data.parquet")
EXTRA_CSV = Path("felipe_data/fid_level_w_drug.csv")

EXTRA_COLS = ["experiment_media", "umap_1", "umap_2"]


def main() -> None:
    if not BASE_PARQUET.exists():
        raise SystemExit(f"Missing base parquet: {BASE_PARQUET}")
    if not EXTRA_CSV.exists():
        raise SystemExit(f"Missing extra CSV: {EXTRA_CSV}")

    print("📖 Loading base parquet...")
    base_df = pd.read_parquet(BASE_PARQUET)
    if "fid" not in base_df.columns:
        raise SystemExit("Base parquet must contain 'fid' column")

    print("📖 Loading extra CSV...")
    try:
        extra_df = pd.read_csv(EXTRA_CSV, engine="pyarrow")
    except Exception:
        extra_df = pd.read_csv(EXTRA_CSV)

    if "fid" not in extra_df.columns:
        raise SystemExit("Extra CSV must contain 'fid' column")

    # Reduce extra to only desired columns (plus key)
    present_extra_cols = [c for c in EXTRA_COLS if c in extra_df.columns]
    missing = [c for c in EXTRA_COLS if c not in extra_df.columns]
    if missing:
        print(f"⚠️ Missing in extra CSV (will be skipped): {missing}")
    extra_reduced = extra_df[["fid"] + present_extra_cols].copy()

    # Ensure types compatible for join
    base_df["fid"] = pd.to_numeric(base_df["fid"], errors="coerce")
    extra_reduced["fid"] = pd.to_numeric(extra_reduced["fid"], errors="coerce")

    # Optional sanity checks
    print(f"🔍 Base rows: {len(base_df)} | Extra rows: {len(extra_reduced)}")
    base_fids = set(base_df["fid"].dropna().unique())
    extra_fids = set(extra_reduced["fid"].dropna().unique())
    missing_in_extra = len(base_fids - extra_fids)
    if missing_in_extra:
        print(f"⚠️ {missing_in_extra} base fids not present in extra; those rows won't get new values")

    # Avoid duplicate suffixes: drop any existing columns we will replace
    for col in present_extra_cols:
        if col in base_df.columns:
            print(f"ℹ️ Column already exists in base: {col} — will be overwritten from extra")

    # Merge: left join to preserve exact base set/order
    merged = base_df.merge(extra_reduced, on="fid", how="left", suffixes=("", "_extra"))

    # If both base and extra had a column, prefer the extra and drop original
    for col in present_extra_cols:
        extra_name = col + "_extra"
        if extra_name in merged.columns:
            merged[col] = merged[extra_name]
            merged.drop(columns=[extra_name], inplace=True)

    # Backup and write
    backup = BASE_PARQUET.with_suffix(".parquet.bak")
    print(f"🗂️  Backing up original parquet to {backup}")
    shutil.copy2(BASE_PARQUET, backup)

    print(f"💾 Writing merged parquet to {BASE_PARQUET}")
    merged.to_parquet(BASE_PARQUET, index=False)

    # Report
    print("✅ Done. Columns now include:")
    have_cols = [c for c in EXTRA_COLS if c in merged.columns]
    for c in have_cols:
        print(f"  - {c} ({merged[c].dtype})")


if __name__ == "__main__":
    main()


