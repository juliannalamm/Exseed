# Run Filtered Export for Participant 158d356b

## Quick Start

Add this cell to your notebook and run it:

```python
# ============================================================================
# Export ONLY participant 158d356b for Dash app (minimal dataset)
# ============================================================================

%run export_filtered_for_dash.py
```

That's it! This will:
- ✅ Export only participant 158d356b's data
- ✅ Reduce dataset from ~7 MB to < 1 MB
- ✅ Include tracks, frames, and patient summary
- ✅ Create minimal archetype config

## What Gets Exported

### Before (All Participants)
- tracks_data.parquet: ~2.6 MB (30,895 tracks)
- frames_data.parquet: ~4.2 MB (1.5M frames)
- patient_data.parquet: ~73 KB (305 patients)
- **Total: ~6.9 MB**

### After (Only 158d356b)
- tracks_data.parquet: ~10-20 KB (one participant's tracks)
- frames_data.parquet: ~50-100 KB (one participant's frames)
- patient_data.parquet: ~1 KB (1 patient)
- **Total: < 200 KB** 💾

## File Reduction
Expected reduction: **~97% smaller!**

Perfect for Docker images! 🐳
