# Exporting Data for Dash App

This guide explains how to export your GMM analysis data for use in the Dash app.

## Quick Start

### Step 1: Run the export from your notebook

Add a new cell to your notebook (after you've run all the analysis cells) and run:

```python
%run export_for_dash.py

# Update these with your actual participant IDs
archetype_pids = {
    'A': '158d356b',  # High P, Low E
    'B': pid_B,       # Mixed (use the variable from your notebook)
    'C': pid_C,       # High E, Low P  
    'D': pid_D,       # Immotile
}

archetype_info = {
    'A': {'title': 'Clean Progressive High P, Low E'},
    'B': {'title': 'Heterogeneous Sample: Mixed P, Mixed E'},
    'C': {'title': 'Erratic and Fast: High E, Low P'},
    'D': {'title': 'Immotile: Low P, High E'},
}

export_datasets(
    train_out_k5, 
    train_frame_df, 
    patient_fp_enriched,
    archetype_pids=archetype_pids,
    archetype_info=archetype_info
)
```

### Step 2: Exported Files

The script will create `app/dash_data/` with these files:

- **tracks_data.parquet** - Per-track data with GMM posteriors (~30K tracks)
  - Columns: participant_id, track_id, P_progressive, P_rapid_progressive, etc.
  - Computed axes: progressivity, erraticity, entropy
  - CASA features: ALH, VCL, LIN, etc.

- **frames_data.parquet** - Frame-by-frame trajectory coordinates (~1.5M frames)
  - Columns: participant_id, track_id, frame_num, x, y

- **patient_data.parquet** - Patient-level fingerprints (305 patients)
  - GMM cluster percentages: pct_progressive, pct_rapid_progressive, etc.
  - Mean axes: mean_P_byls, mean_E_byls, mean_entropy
  - CASA metrics: CASA_ALH_mean, CASA_VCL_mean, etc.

- **archetype_config.json** - Metadata about archetype patients
  - Participant IDs for each archetype
  - Titles and descriptions
  - Default plot settings

## Using in Dash App

### Load the data in your Dash app:

```python
from app.archetype_data_loader import ArchetypeDataLoader

# Initialize loader
loader = ArchetypeDataLoader("app/dash_data")

# List available archetypes
archetypes = loader.get_archetype_list()
# ['A', 'B', 'C', 'D']

# Get all data for an archetype
tracks, frames, patient, info = loader.get_archetype_data('A')

# Get just the tracks for a participant
tracks = loader.get_patient_tracks('158d356b')

# Pick diverse tracks for plotting
selected_track_ids = loader.pick_diverse_tracks(tracks, n_total=120, rng_seed=42)

# Get CASA scales for radar normalization
casa_scales = loader.get_casa_scales()
```

## Data Organization

The exported data follows this structure:

```
ExSeed/
├── app/
│   ├── dash_data/              # Exported data (gitignored)
│   │   ├── tracks_data.parquet
│   │   ├── frames_data.parquet
│   │   ├── patient_data.parquet
│   │   └── archetype_config.json
│   ├── archetype_data_loader.py  # Helper to load data
│   └── dash_app.py              # Your Dash app
└── notebooks/
    ├── export_for_dash.py       # Export script
    └── gmm_train_test_split_k5 copy.ipynb
```

## Tips

1. **Data size**: Parquet files are compressed and efficient. The full dataset should be ~50-100 MB.

2. **Git**: The `dash_data/` folder should be in `.gitignore` (add `app/dash_data/` to your `.gitignore`)

3. **Updating data**: Re-run the export cell whenever you:
   - Change the GMM model
   - Select different archetype patients
   - Update computed axes

4. **Custom archetypes**: You can add more archetypes by extending the `archetype_pids` dictionary

## Troubleshooting

**Missing variables**: Make sure you've run all cells in your notebook up to where `train_out_k5`, `train_frame_df`, and `patient_fp_enriched` are defined.

**Missing columns**: The export script will warn you about missing columns but will export what's available.

**File not found**: Make sure the export script creates the `app/dash_data/` directory automatically.
