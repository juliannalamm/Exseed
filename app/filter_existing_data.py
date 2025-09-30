"""
Filter existing exported data to only include participant 158d356b.
Run this from the app/ directory.
"""

import pandas as pd
import json
from pathlib import Path

# Configuration
DATA_DIR = Path("dash_data")
TARGET_PARTICIPANT = "b7f96273"

print("="*70)
print("FILTERING EXISTING DATA")
print(f"Target participant: {TARGET_PARTICIPANT}")
print("="*70)

# ============================================================================
# 1. Filter tracks data
# ============================================================================
print("\n[1/4] Filtering tracks data...")

tracks_path = DATA_DIR / "tracks_data.parquet"
if tracks_path.exists():
    tracks = pd.read_parquet(tracks_path)
    print(f"  Original: {len(tracks):,} tracks from {tracks['participant_id'].nunique()} participants")
    
    tracks_filtered = tracks[tracks['participant_id'] == TARGET_PARTICIPANT].copy()
    tracks_filtered.to_parquet(tracks_path, index=False)
    
    print(f"  Filtered: {len(tracks_filtered):,} tracks")
    print(f"  ✓ Saved to {tracks_path}")
    
    # Show file size
    size_mb = tracks_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
else:
    print(f"  ⚠️  File not found: {tracks_path}")

# ============================================================================
# 2. Filter frames data
# ============================================================================
print("\n[2/4] Filtering frames data...")

frames_path = DATA_DIR / "frames_data.parquet"
if frames_path.exists():
    frames = pd.read_parquet(frames_path)
    print(f"  Original: {len(frames):,} frames from {frames['participant_id'].nunique()} participants")
    
    frames_filtered = frames[frames['participant_id'] == TARGET_PARTICIPANT].copy()
    frames_filtered.to_parquet(frames_path, index=False)
    
    print(f"  Filtered: {len(frames_filtered):,} frames")
    print(f"  ✓ Saved to {frames_path}")
    
    size_mb = frames_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
else:
    print(f"  ⚠️  File not found: {frames_path}")

# ============================================================================
# 3. Filter patient data
# ============================================================================
print("\n[3/4] Filtering patient data...")

patient_path = DATA_DIR / "patient_data.parquet"
if patient_path.exists():
    patients = pd.read_parquet(patient_path)
    print(f"  Original: {len(patients):,} patients")
    
    patient_filtered = patients[patients['participant_id'] == TARGET_PARTICIPANT].copy()
    patient_filtered.to_parquet(patient_path, index=False)
    
    print(f"  Filtered: {len(patient_filtered):,} patient(s)")
    print(f"  ✓ Saved to {patient_path}")
    
    size_kb = patient_path.stat().st_size / 1024
    print(f"  File size: {size_kb:.2f} KB")
else:
    print(f"  ⚠️  File not found: {patient_path}")

# ============================================================================
# 4. Update config
# ============================================================================
print("\n[4/4] Updating config...")

config_path = DATA_DIR / "archetype_config.json"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update to only have one archetype
    config['archetypes'] = {
        'A': {
            'participant_id': TARGET_PARTICIPANT,
            'title': 'Clean Progressive High P, Low E',
            'description': 'Representative participant with high progressivity and low erraticity'
        }
    }
    config['note'] = f'Filtered dataset containing only participant {TARGET_PARTICIPANT}'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✓ Updated config to single archetype")
else:
    print(f"  ⚠️  File not found: {config_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("FILTERING COMPLETE!")
print("="*70)

# Calculate total size
total_size = 0
for file in ['tracks_data.parquet', 'frames_data.parquet', 'patient_data.parquet', 'archetype_config.json']:
    file_path = DATA_DIR / file
    if file_path.exists():
        total_size += file_path.stat().st_size

total_mb = total_size / (1024 * 1024)
print(f"\nTotal dataset size: {total_mb:.2f} MB")
print(f"Ready for Docker deployment! 🐳")
print("="*70)
