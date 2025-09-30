"""
Export data from GMM analysis for Dash app visualization.

This script exports three main datasets:
1. tracks_data.parquet - per-sperm/track-level data with GMM posteriors
2. frames_data.parquet - frame-by-frame trajectory coordinates
3. patient_data.parquet - patient-level fingerprints with CASA metrics
4. archetype_config.json - metadata about selected archetype patients
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

# ============================================================================
# Load the data (assumes you've already run the notebook up to Cell 40+)
# ============================================================================

print("Loading data from notebook variables...")
print("Note: Run this script from within the notebook using %run or exec()")

# Required columns for each dataset
TRACK_COLS_REQUIRED = [
    'participant_id', 'track_id',
    # GMM posteriors
    'P_progressive', 'P_rapid_progressive', 'P_nonprogressive', 
    'P_immotile', 'P_erratic',
    # Computed axes
    'progressivity', 'erraticity', 'entropy',
    # Optional: byls axes if available
    'P_axis_byls', 'E_axis_byls',
    # CASA features (for reference)
    'ALH', 'BCF', 'LIN', 'MAD', 'STR', 'VAP', 'VCL', 'VSL', 'WOB',
]

FRAME_COLS_REQUIRED = [
    'participant_id', 'track_id', 'frame_num', 'x', 'y'
]

PATIENT_COLS_REQUIRED = [
    'participant_id',
    # GMM cluster percentages
    'pct_progressive', 'pct_rapid_progressive', 'pct_nonprogressive',
    'pct_immotile', 'pct_erratic',
    # Mean axes
    'mean_P_byls', 'mean_E_byls', 'mean_entropy',
    # CASA metrics
    'CASA_ALH_mean', 'CASA_BCF_mean', 'CASA_LIN_mean', 'CASA_MAD_mean',
    'CASA_STR_mean', 'CASA_VAP_mean', 'CASA_VCL_mean', 'CASA_VSL_mean',
    'CASA_WOB_mean',
    # Optional: CASA diversity if available
    'H_casa_norm', 'N_eff_casa',
    # Track count
    'n_tracks',
]

def export_datasets(train_out_k5, train_frame_df, patient_fp_enriched, 
                   archetype_pids=None, archetype_info=None):
    """
    Export the three main datasets needed for Dash visualization.
    
    Parameters
    ----------
    train_out_k5 : pd.DataFrame
        Per-track data with GMM posteriors and computed axes
    train_frame_df : pd.DataFrame
        Frame-by-frame trajectory data
    patient_fp_enriched : pd.DataFrame
        Patient-level fingerprints with CASA metrics
    archetype_pids : dict, optional
        Dictionary mapping archetype names to participant IDs
        Example: {'A': 'pid_123', 'B': 'pid_456', ...}
    archetype_info : dict, optional
        Dictionary with archetype titles and descriptions
        Example: {'A': {'title': 'High P, Low E', 'description': '...'}, ...}
    """
    
    print("\n" + "="*70)
    print("EXPORTING DATA FOR DASH APP")
    print("="*70)
    
    # ========================================================================
    # 1. Export tracks data
    # ========================================================================
    print("\n[1/4] Exporting tracks data...")
    
    # Select available columns
    available_track_cols = [c for c in TRACK_COLS_REQUIRED if c in train_out_k5.columns]
    missing_track_cols = [c for c in TRACK_COLS_REQUIRED if c not in train_out_k5.columns]
    
    if missing_track_cols:
        print(f"  ⚠️  Warning: Missing columns in train_out_k5: {missing_track_cols}")
    
    tracks_export = train_out_k5[available_track_cols].copy()
    
    # Add any additional useful columns
    extra_cols = ['subtype_label', 'cluster_id', 'track_length_frames']
    for col in extra_cols:
        if col in train_out_k5.columns and col not in tracks_export.columns:
            tracks_export[col] = train_out_k5[col]
    
    # Save as parquet (more efficient than CSV)
    tracks_path = OUTPUT_DIR / "tracks_data.parquet"
    tracks_export.to_parquet(tracks_path, index=False)
    print(f"  ✓ Saved {len(tracks_export):,} tracks to: {tracks_path}")
    print(f"    Columns: {list(tracks_export.columns)}")
    
    # ========================================================================
    # 2. Export frames data
    # ========================================================================
    print("\n[2/4] Exporting frames data...")
    
    available_frame_cols = [c for c in FRAME_COLS_REQUIRED if c in train_frame_df.columns]
    missing_frame_cols = [c for c in FRAME_COLS_REQUIRED if c not in train_frame_df.columns]
    
    if missing_frame_cols:
        print(f"  ⚠️  Warning: Missing columns in train_frame_df: {missing_frame_cols}")
        return
    
    frames_export = train_frame_df[available_frame_cols].copy()
    
    # Save as parquet
    frames_path = OUTPUT_DIR / "frames_data.parquet"
    frames_export.to_parquet(frames_path, index=False)
    print(f"  ✓ Saved {len(frames_export):,} frames to: {frames_path}")
    print(f"    Unique tracks: {frames_export['track_id'].nunique():,}")
    
    # ========================================================================
    # 3. Export patient data
    # ========================================================================
    print("\n[3/4] Exporting patient data...")
    
    available_patient_cols = [c for c in PATIENT_COLS_REQUIRED if c in patient_fp_enriched.columns]
    missing_patient_cols = [c for c in PATIENT_COLS_REQUIRED if c not in patient_fp_enriched.columns]
    
    if missing_patient_cols:
        print(f"  ⚠️  Warning: Missing columns in patient_fp_enriched: {missing_patient_cols}")
    
    patients_export = patient_fp_enriched[available_patient_cols].copy()
    
    # Add any additional useful columns
    extra_patient_cols = ['std_P_byls', 'std_E_byls', 'H_gmm_norm', 'N_eff_gmm']
    for col in extra_patient_cols:
        if col in patient_fp_enriched.columns and col not in patients_export.columns:
            patients_export[col] = patient_fp_enriched[col]
    
    # Save as parquet
    patients_path = OUTPUT_DIR / "patient_data.parquet"
    patients_export.to_parquet(patients_path, index=False)
    print(f"  ✓ Saved {len(patients_export):,} patients to: {patients_path}")
    print(f"    Columns: {list(patients_export.columns)}")
    
    # ========================================================================
    # 4. Export archetype configuration
    # ========================================================================
    print("\n[4/4] Exporting archetype configuration...")
    
    if archetype_pids is not None:
        config = {
            "archetypes": {},
            "radar_order": ['progressive', 'rapid_progressive', 'nonprogressive', 'immotile', 'erratic'],
            "casa_columns": [c for c in patient_fp_enriched.columns if c.startswith('CASA_') and c.endswith('_mean')],
            "plot_defaults": {
                "n_tracks_bottom": 120,
                "mark_endpoints": False,
                "invert_y": True,
                "rng_seed": 42
            }
        }
        
        for archetype_name, pid in archetype_pids.items():
            archetype_data = {
                "participant_id": pid,
                "title": archetype_info.get(archetype_name, {}).get('title', f'Archetype {archetype_name}') if archetype_info else f'Archetype {archetype_name}',
            }
            
            # Add description if available
            if archetype_info and archetype_name in archetype_info:
                if 'description' in archetype_info[archetype_name]:
                    archetype_data['description'] = archetype_info[archetype_name]['description']
            
            config['archetypes'][archetype_name] = archetype_data
        
        config_path = OUTPUT_DIR / "archetype_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Saved archetype config to: {config_path}")
        print(f"    Archetypes: {list(config['archetypes'].keys())}")
    else:
        print("  ⚠️  No archetype PIDs provided, skipping config export")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("EXPORT COMPLETE!")
    print("="*70)
    print(f"\nExported files to: {OUTPUT_DIR.absolute()}")
    print("\nTo use in your Dash app:")
    print("  1. Copy the dash_data/ folder to your app/ directory")
    print("  2. Load the data:")
    print("     tracks_df = pd.read_parquet('dash_data/tracks_data.parquet')")
    print("     frames_df = pd.read_parquet('dash_data/frames_data.parquet')")
    print("     patient_df = pd.read_parquet('dash_data/patient_data.parquet')")
    print("     with open('dash_data/archetype_config.json') as f:")
    print("         config = json.load(f)")
    print("\n" + "="*70)


# ============================================================================
# Example usage (to be run from notebook)
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nTo use this script, run it from your notebook:")
    print("\n  # In a notebook cell:")
    print("  %run export_for_dash.py")
    print("  ")
    print("  # Then export your data:")
    print("  archetype_pids = {")
    print("      'A': patient_A,  # High P, Low E")
    print("      'B': pid_B,      # Mixed")
    print("      'C': pid_C,      # High E, Low P")
    print("      'D': pid_D,      # Immotile")
    print("  }")
    print("  ")
    print("  archetype_info = {")
    print("      'A': {'title': 'Clean Progressive High P, Low E'},")
    print("      'B': {'title': 'Heterogeneous Sample: Mixed P, Mixed E'},")
    print("      'C': {'title': 'Erratic and Fast: High E, Low P'},")
    print("      'D': {'title': 'Immotile: Low P, High E'},")
    print("  }")
    print("  ")
    print("  export_datasets(train_out_k5, train_frame_df, patient_fp_enriched,")
    print("                  archetype_pids=archetype_pids,")
    print("                  archetype_info=archetype_info)")
