#!/usr/bin/env python3
"""
Script to update parquet files with new CSV data including drug columns and UMAP data.
"""

import pandas as pd
import numpy as np
import os

def update_parquet_files():
    """Update the parquet files with new CSV data"""
    
    print("🔄 Starting parquet file update...")
    
    # Read the new CSV files
    print("📖 Reading fid_level_w_drug.csv...")
    fid_data = pd.read_csv('felipe_data/fid_level_w_drug.csv')
    
    print("📖 Reading frame_level_w_drug.csv...")
    frame_data = pd.read_csv('felipe_data/frame_level_w_drug.csv')
    
    print(f"✅ Loaded {len(fid_data)} fid records and {len(frame_data)} frame records")
    
    # Check what new columns we have
    print("\n📊 New columns in fid_level_w_drug.csv:")
    new_fid_cols = ['experiment_media', 'umap_1', 'umap_2', 'drug_cluster']
    for col in new_fid_cols:
        if col in fid_data.columns:
            print(f"  ✅ {col}: {fid_data[col].dtype}")
        else:
            print(f"  ❌ {col}: Not found")
    
    print("\n📊 New columns in frame_level_w_drug.csv:")
    new_frame_cols = ['experiment_media']
    for col in new_frame_cols:
        if col in frame_data.columns:
            print(f"  ✅ {col}: {frame_data[col].dtype}")
        else:
            print(f"  ❌ {col}: Not found")
    
    # Create the updated patient_data.parquet (from fid_level_w_drug)
    print("\n🔄 Creating updated patient_data.parquet...")
    
    # Select the columns we need for the patient data
    patient_columns = [
        'fid', 'subtype_label', 'is_hyperactive_mouse', 'tsne_1', 'tsne_2', 
        'P_axis_byls', 'E_axis_byls', 'entropy', 'experiment_media', 'umap_1', 'umap_2'
    ]
    
    # Filter to only include columns that exist
    available_patient_cols = [col for col in patient_columns if col in fid_data.columns]
    patient_data = fid_data[available_patient_cols].copy()
    
    print(f"📊 Patient data shape: {patient_data.shape}")
    print(f"📊 Columns: {list(patient_data.columns)}")
    
    # Save patient data (overwrite the existing fid_level_data.parquet)
    patient_data.to_parquet('felipe_data/fid_level_data.parquet', index=False)
    print("✅ Saved fid_level_data.parquet")
    
    # Create the updated trajectory data (from frame_level_w_drug)
    print("\n🔄 Creating updated trajectory data...")
    
    # Select the columns we need for the trajectory data
    trajectory_columns = [
        'fid', 'frame_number', 'x', 'y', 'experiment_media'
    ]
    
    # Filter to only include columns that exist
    available_trajectory_cols = [col for col in trajectory_columns if col in frame_data.columns]
    trajectory_data = frame_data[available_trajectory_cols].copy()
    
    print(f"📊 Trajectory data shape: {trajectory_data.shape}")
    print(f"📊 Columns: {list(trajectory_data.columns)}")
    
    # Save trajectory data (overwrite the existing trajectory_index.parquet)
    trajectory_data.to_parquet('felipe_data/trajectory_index.parquet', index=False)
    print("✅ Saved trajectory_index.parquet")
    
    # Show sample data
    print("\n📊 Sample patient data:")
    print(patient_data.head())
    
    print("\n📊 Sample trajectory data:")
    print(trajectory_data.head())
    
    # Check for any missing values
    print("\n🔍 Data quality check:")
    print("Patient data missing values:")
    print(patient_data.isnull().sum())
    
    print("\nTrajectory data missing values:")
    print(trajectory_data.isnull().sum())
    
    print("\n🎉 Parquet files updated successfully!")
    print("📁 Updated files:")
    print("  - felipe_data/fid_level_data.parquet")
    print("  - felipe_data/trajectory_index.parquet")

if __name__ == "__main__":
    update_parquet_files()
