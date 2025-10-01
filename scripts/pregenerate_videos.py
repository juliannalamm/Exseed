#!/usr/bin/env python3
"""
Pre-generate video overlays for sample data to speed up initial loading.
Run this script to create optimized videos for the sample participants.
"""

import os
import tempfile
from full_pipeline import json_to_df, predict_sperm_motility, overlay_trajectories_on_video, MODEL_PATH
import subprocess

def convert_to_h264(input_path: str, output_path: str):
    """Convert video to H.264 format for web compatibility."""
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def pregenerate_video(participant_id: str, json_path: str, video_path: str):
    """Pre-generate video overlay for a participant."""
    print(f"🔄 Processing {participant_id}...")
    
    # Parse data
    track_df, frame_df = json_to_df(json_path, participant_id)
    
    # Predict subtypes
    preds = predict_sperm_motility(track_df, model_path=MODEL_PATH, include_umap=False)
    
    # Create output directory
    out_dir = os.path.join("sample_data_for_streamlit", "pregenerated")
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate video overlay
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
    
    print(f"✅ Generated {h264_path}")

def main():
    """Pre-generate videos for all sample participants."""
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
    
    print("🚀 Pre-generating video overlays for faster loading...")
    
    for participant_id, files in participants.items():
        json_path = os.path.join(sample_data_dir, files["json"])
        video_path = os.path.join(sample_data_dir, files["video"])
        
        if os.path.exists(json_path) and os.path.exists(video_path):
            pregenerate_video(participant_id, json_path, video_path)
        else:
            print(f"❌ Missing files for {participant_id}")
    
    print("🎉 All videos pre-generated!")

if __name__ == "__main__":
    main()
