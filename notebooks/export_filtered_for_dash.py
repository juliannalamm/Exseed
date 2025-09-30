"""
Export ONLY participant 158d356b data for Dash app.
This creates a minimal dataset for the comparison feature.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Output directory
OUTPUT_DIR = Path("../app/dash_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Only export this participant
TARGET_PARTICIPANT = "158d356b"

print("="*70)
print("EXPORTING FILTERED DATA FOR DASH APP")
print(f"Target participant: {TARGET_PARTICIPANT}")
print("="*70)

# ============================================================================
# Check required variables exist
# ============================================================================
required_vars = {
    'train_out_k5': 'tracks dataframe with GMM posteriors',
    'train_frame_df': 'frame-by-frame trajectory data',
    'patient_fp_enriched': 'patient-level fingerprints'
}

missing_vars = []
for var_name, description in required_vars.items():
    if var_name not in globals():
        missing_vars.append(f"  - {var_name}: {description}")

if missing_vars:
    print("\n❌ ERROR: Missing required variables!")
    print("\nPlease run these cells first:")
    print("  1. Cell that creates train_out_k5 (GMM clustering)")
    print("  2. Cell that creates train_frame_df (trajectory frames)")
    print("  3. Cell that creates patient_fp_enriched (patient fingerprints)")
    print("\nMissing variables:")
    for var in missing_vars:
        print(var)
    print("\nTip: Make sure you've run all analysis cells up to the")
    print("     point where these variables are created!")
    print("="*70)
    raise NameError(f"Required variables not found: {', '.join([v.split(':')[0].strip('  - ') for v in missing_vars])}")

print(f"✓ All required variables found")

# ============================================================================
# 1. Export tracks data (FILTERED)
# ============================================================================
print(f"\n[1/4] Filtering tracks data for {TARGET_PARTICIPANT}...")

# Filter to only the target participant
tracks_filtered = train_out_k5[train_out_k5['participant_id'] == TARGET_PARTICIPANT].copy()

if len(tracks_filtered) == 0:
    print(f"  ⚠️  WARNING: No tracks found for participant {TARGET_PARTICIPANT}")
    print(f"  Available participants: {train_out_k5['participant_id'].unique()[:10]}")
else:
    # Select columns
    track_cols = [
        'participant_id', 'track_id',
        # GMM posteriors
        'P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 
        'P_immotile', 'P_erratic',
        # Computed axes
        'progressivity', 'erraticity', 'entropy',
        # CASA features
        'ALH', 'BCF', 'LIN', 'MAD', 'STR', 'VAP', 'VCL', 'VSL', 'WOB',
    ]
    
    # Add optional columns if they exist
    optional_cols = ['P_axis_byls', 'E_axis_byls', 'subtype_label', 'cluster_id', 'track_length_frames']
    for col in optional_cols:
        if col in tracks_filtered.columns:
            track_cols.append(col)
    
    tracks_export = tracks_filtered[[c for c in track_cols if c in tracks_filtered.columns]].copy()
    
    # Save
    tracks_path = OUTPUT_DIR / "tracks_data.parquet"
    tracks_export.to_parquet(tracks_path, index=False)
    
    print(f"  ✓ Saved {len(tracks_export):,} tracks to: {tracks_path}")
    print(f"    Original size: {len(train_out_k5):,} tracks")
    print(f"    Filtered size: {len(tracks_export):,} tracks")
    print(f"    Reduction: {(1 - len(tracks_export)/len(train_out_k5))*100:.1f}%")

# ============================================================================
# 2. Export frames data (FILTERED)
# ============================================================================
print(f"\n[2/4] Filtering frames data for {TARGET_PARTICIPANT}...")

frames_filtered = train_frame_df[train_frame_df['participant_id'] == TARGET_PARTICIPANT].copy()

if len(frames_filtered) > 0:
    frame_cols = ['participant_id', 'track_id', 'frame_num', 'x', 'y']
    frames_export = frames_filtered[frame_cols].copy()
    
    # Save
    frames_path = OUTPUT_DIR / "frames_data.parquet"
    frames_export.to_parquet(frames_path, index=False)
    
    print(f"  ✓ Saved {len(frames_export):,} frames to: {frames_path}")
    print(f"    Original size: {len(train_frame_df):,} frames")
    print(f"    Filtered size: {len(frames_export):,} frames")
    print(f"    Reduction: {(1 - len(frames_export)/len(train_frame_df))*100:.1f}%")
else:
    print(f"  ⚠️  No frames found for {TARGET_PARTICIPANT}")

# ============================================================================
# 3. Export patient data (FILTERED)
# ============================================================================
print(f"\n[3/4] Filtering patient data for {TARGET_PARTICIPANT}...")

patient_filtered = patient_fp_enriched[patient_fp_enriched['participant_id'] == TARGET_PARTICIPANT].copy()

if len(patient_filtered) > 0:
    # All columns for this one participant
    patients_export = patient_filtered.copy()
    
    # Save
    patients_path = OUTPUT_DIR / "patient_data.parquet"
    patients_export.to_parquet(patients_path, index=False)
    
    print(f"  ✓ Saved {len(patients_export):,} patient(s) to: {patients_path}")
    print(f"    Original size: {len(patient_fp_enriched):,} patients")
    print(f"    Filtered size: {len(patients_export):,} patient(s)")
    print(f"    Columns: {len(patients_export.columns)}")
else:
    print(f"  ⚠️  Participant {TARGET_PARTICIPANT} not found in patient_fp_enriched")
    print(f"  Available: {patient_fp_enriched['participant_id'].head(10).tolist()}")

# ============================================================================
# 4. Export archetype configuration
# ============================================================================
print(f"\n[4/4] Creating archetype configuration...")

config = {
    "archetypes": {
        "A": {
            "participant_id": TARGET_PARTICIPANT,
            "title": "Clean Progressive High P, Low E",
            "description": "Representative participant with high progressivity and low erraticity"
        }
    },
    "radar_order": ['progressive', 'rapid_progressive', 'nonprogressive', 'immotile', 'erratic'],
    "casa_columns": [c for c in patient_fp_enriched.columns if c.startswith('CASA_') and c.endswith('_mean')],
    "plot_defaults": {
        "n_tracks_bottom": 120,
        "mark_endpoints": False,
        "invert_y": True,
        "rng_seed": 42
    },
    "note": f"Filtered export containing ONLY participant {TARGET_PARTICIPANT}"
}

config_path = OUTPUT_DIR / "archetype_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"  ✓ Saved config to: {config_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("EXPORT COMPLETE!")
print("="*70)

# Calculate file sizes
import os

def get_size_mb(path):
    if path.exists():
        return os.path.getsize(path) / (1024 * 1024)
    return 0

tracks_size = get_size_mb(OUTPUT_DIR / "tracks_data.parquet")
frames_size = get_size_mb(OUTPUT_DIR / "frames_data.parquet")
patient_size = get_size_mb(OUTPUT_DIR / "patient_data.parquet")
config_size = get_size_mb(OUTPUT_DIR / "archetype_config.json")
total_size = tracks_size + frames_size + patient_size + config_size

print(f"\nExported files to: {OUTPUT_DIR.absolute()}")
print(f"\nFile sizes:")
print(f"  tracks_data.parquet:   {tracks_size:>6.2f} MB")
print(f"  frames_data.parquet:   {frames_size:>6.2f} MB")
print(f"  patient_data.parquet:  {patient_size:>6.2f} MB")
print(f"  archetype_config.json: {config_size:>6.2f} MB")
print(f"  {'─' * 35}")
print(f"  TOTAL:                 {total_size:>6.2f} MB")

print(f"\n✨ Minimal dataset ready for Docker image!")
print(f"   Contains only participant {TARGET_PARTICIPANT}")
print("="*70)
