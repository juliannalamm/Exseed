#!/usr/bin/env python3
"""
Pre-calculate all data for the 3 sample patients to enable instant loading.
This script processes all the data once and saves it as JSON files.
"""

import os
import json
import pandas as pd
import numpy as np
from full_pipeline import json_to_df, predict_sperm_motility, overlay_trajectories_on_video, MODEL_PATH
import subprocess
import tempfile

def convert_to_h264(input_path: str, output_path: str):
    """Convert video to H.264 format for web compatibility."""
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def precalculate_patient_data(participant_id: str, json_path: str, video_path: str):
    """Pre-calculate all data for a patient."""
    print(f"🔄 Processing {participant_id}...")
    
    # Parse data
    track_df, frame_df = json_to_df(json_path, participant_id)
    
    # Predict subtypes
    preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=False)
    
    # Calculate percentages
    subtype_counts = preds['subtype_label'].value_counts()
    total_tracks = len(preds)
    percentages = (subtype_counts / total_tracks * 100).round(1)
    
    # Calculate feature averages by subtype
    feature_averages = {}
    for subtype in ['Progressive', 'Vigorous', 'Nonprogressive', 'Immotile']:
        if subtype in preds['subtype_label'].values:
            subtype_data = preds[preds['subtype_label'] == subtype]
            feature_averages[subtype] = {
                'VCL': subtype_data['VCL'].mean(),
                'VSL': subtype_data['VSL'].mean(), 
                'VAP': subtype_data['VAP'].mean(),
                'LIN': subtype_data['LIN'].mean(),
                'STR': subtype_data['STR'].mean(),
                'WOB': subtype_data['WOB'].mean(),
                'ALH': subtype_data['ALH'].mean(),
                'BCF': subtype_data['BCF'].mean(),
                'MAD': subtype_data['MAD'].mean()
            }
        else:
            feature_averages[subtype] = {key: 0 for key in ['VCL', 'VSL', 'VAP', 'LIN', 'STR', 'WOB', 'ALH', 'BCF', 'MAD']}
    
    # Generate video overlay
    out_dir = os.path.join("sample_data_for_streamlit", "pregenerated")
    os.makedirs(out_dir, exist_ok=True)
    
    raw_overlay_path = os.path.join(out_dir, f"{participant_id}_raw_overlay.mp4")
    h264_path = os.path.join(out_dir, f"{participant_id}_overlay_h264.mp4")
    
    overlay_trajectories_on_video(
        frame_df=frame_df,
        track_df=preds,
        video_path=video_path,
        output_path=raw_overlay_path
    )
    
    # Convert to H.264
    convert_to_h264(raw_overlay_path, h264_path)
    
    # Clean up raw file
    os.remove(raw_overlay_path)
    
    # Create data summary
    patient_data = {
        'participant_id': participant_id,
        'total_tracks': total_tracks,
        'percentages': percentages.to_dict(),
        'feature_averages': feature_averages,
        'video_path': h264_path,
        'raw_data': {
            'track_df': track_df.to_dict('records'),
            'frame_df': frame_df.to_dict('records'),
            'preds': preds.to_dict('records')
        }
    }
    
    # Save as JSON
    output_file = os.path.join(out_dir, f"{participant_id}_complete_data.json")
    with open(output_file, 'w') as f:
        json.dump(patient_data, f, indent=2)
    
    print(f"✅ Generated {output_file}")
    return patient_data

def main():
    """Pre-calculate all data for all sample participants."""
    sample_data_dir = "sample_data_for_streamlit"
    
    participants = {
        "767ef2ec_v2": {
            "json": "767ef2ec_v2_tracks_with_mot_params.json",
            "video": "767ef2ec_v2_raw_video.mp4"
        },
        "bcb2b5c7_v1": {
            "json": "bcb2b5c7_v1_tracks_with_mot_params.json", 
            "video": "bcb2b5c7_v1_raw_video.mp4"
        },
        "ef5f3e74_v1": {
            "json": "ef5f3e74_v1_tracks_with_mot_params.json",
            "video": "ef5f3e74_v1_raw_video.mp4"
        }
    }
    
    print("🚀 Pre-calculating all data for instant loading...")
    
    all_data = {}
    
    for participant_id, files in participants.items():
        json_path = os.path.join(sample_data_dir, files["json"])
        video_path = os.path.join(sample_data_dir, files["video"])
        
        if os.path.exists(json_path) and os.path.exists(video_path):
            patient_data = precalculate_patient_data(participant_id, json_path, video_path)
            all_data[participant_id] = patient_data
        else:
            print(f"❌ Missing files for {participant_id}")
    
    # Save master data file
    master_file = os.path.join(sample_data_dir, "pregenerated", "all_patient_data.json")
    with open(master_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"🎉 All data pre-calculated! Master file: {master_file}")
    print(f"📊 Processed {len(all_data)} patients")

if __name__ == "__main__":
    main()
