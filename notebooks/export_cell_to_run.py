# ============================================================================
# Add this cell to your notebook to export data for Dash app
# ============================================================================

%run export_for_dash.py

# Define your archetype participants (update these variable names based on your notebook)
archetype_pids = {
    'A': '158d356b',  # or patient_A if you have that variable
    'B': pid_B,       # Update with the actual participant ID or variable
    'C': pid_C,       # Update with the actual participant ID or variable
    'D': pid_D,       # Update with the actual participant ID or variable
}

# Define archetype metadata
archetype_info = {
    'A': {
        'title': 'Clean Progressive High P, Low E',
        'description': 'Progressive sperm with low erraticity'
    },
    'B': {
        'title': 'Heterogeneous Sample: Mixed P, Mixed E',
        'description': 'Mixed population with diverse motility patterns'
    },
    'C': {
        'title': 'Erratic and Fast: High E, Low P',
        'description': 'Erratic sperm with high lateral movement'
    },
    'D': {
        'title': 'Immotile: Low P, High E',
        'description': 'Low motility sperm'
    },
}

# Export the data
export_datasets(
    train_out_k5, 
    train_frame_df, 
    patient_fp_enriched,
    archetype_pids=archetype_pids,
    archetype_info=archetype_info
)
