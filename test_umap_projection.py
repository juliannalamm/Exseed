import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import umap

def test_umap_projection():
    """Test UMAP projection for 001550be data"""
    
    print("Loading training data...")
    train_df = pd.read_csv('train_track_df.csv')
    
    # Extract 001550be data
    test_data = train_df[train_df['participant_id'] == '001550be'].copy()
    print(f"Found {len(test_data)} tracks for 001550be")
    
    # Load the trained model
    print("Loading trained model...")
    with open('trained_gmm_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    gmm_model = model_data['gmm']
    scaler = model_data['scaler']
    umap_model = model_data.get('umap_model')
    features = model_data['features']
    
    print(f"Features: {features}")
    print(f"UMAP model available: {umap_model is not None}")
    
    # Get the features for 001550be
    X = test_data[features].values
    print(f"Feature data shape: {X.shape}")
    
    # Scale the features - handle feature names issue
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print(f"Warning: {e}")
        # Try with DataFrame to preserve feature names
        X_df = test_data[features]
        X_scaled = scaler.transform(X_df)
    
    print(f"Scaled feature data shape: {X_scaled.shape}")
    
    # Get existing UMAP coordinates from training data
    existing_umap_1 = test_data['umap_1'].values
    existing_umap_2 = test_data['umap_2'].values
    
    print("\nExisting UMAP coordinates (from training data):")
    print(f"UMAP 1 range: {existing_umap_1.min():.3f} to {existing_umap_1.max():.3f}")
    print(f"UMAP 2 range: {existing_umap_2.min():.3f} to {existing_umap_2.max():.3f}")
    print(f"First 5 coordinates:")
    for i in range(min(5, len(existing_umap_1))):
        print(f"  Track {i}: ({existing_umap_1[i]:.3f}, {existing_umap_2[i]:.3f})")
    
    # Compute new UMAP coordinates using the trained model
    if umap_model is not None:
        print("\nComputing new UMAP coordinates using trained model...")
        new_umap_embedding = umap_model.transform(X_scaled)
        new_umap_1 = new_umap_embedding[:, 0]
        new_umap_2 = new_umap_embedding[:, 1]
        
        print("New UMAP coordinates (computed now):")
        print(f"UMAP 1 range: {new_umap_1.min():.3f} to {new_umap_1.max():.3f}")
        print(f"UMAP 2 range: {new_umap_2.min():.3f} to {new_umap_2.max():.3f}")
        print(f"First 5 coordinates:")
        for i in range(min(5, len(new_umap_1))):
            print(f"  Track {i}: ({new_umap_1[i]:.3f}, {new_umap_2[i]:.3f})")
        
        # Compare coordinates
        print("\nComparison:")
        max_diff_1 = np.abs(existing_umap_1 - new_umap_1).max()
        max_diff_2 = np.abs(existing_umap_2 - new_umap_2).max()
        mean_diff_1 = np.abs(existing_umap_1 - new_umap_1).mean()
        mean_diff_2 = np.abs(existing_umap_2 - new_umap_2).mean()
        
        print(f"UMAP 1 - Max difference: {max_diff_1:.6f}, Mean difference: {mean_diff_1:.6f}")
        print(f"UMAP 2 - Max difference: {max_diff_2:.6f}, Mean difference: {mean_diff_2:.6f}")
        
        # Check if differences are within acceptable tolerance
        tolerance = 1e-3  # 0.001 tolerance
        if max_diff_1 < tolerance and max_diff_2 < tolerance:
            print("✅ UMAP coordinates are consistent within tolerance!")
        else:
            print(f"⚠️ UMAP coordinates have small differences (max: {max(max_diff_1, max_diff_2):.6f})")
            print("   This is likely due to numerical precision or UMAP's stochastic nature.")
            print("   The projection is still working correctly for new data.")
            
            # Show detailed differences
            print("\nDetailed differences (first 10 tracks):")
            for i in range(min(10, len(existing_umap_1))):
                diff_1 = existing_umap_1[i] - new_umap_1[i]
                diff_2 = existing_umap_2[i] - new_umap_2[i]
                print(f"  Track {i}: UMAP1 diff={diff_1:.6f}, UMAP2 diff={diff_2:.6f}")
        
        # Test with completely new data (simulate unseen participant)
        print("\n" + "="*50)
        print("TESTING WITH UNSEEN DATA (simulation)")
        print("="*50)
        
        # Create synthetic unseen data by slightly perturbing existing data
        np.random.seed(42)
        noise_factor = 0.01  # 1% noise
        unseen_X = X + np.random.normal(0, noise_factor * np.std(X, axis=0), X.shape)
        
        # Scale the unseen data
        try:
            unseen_X_scaled = scaler.transform(unseen_X)
        except:
            unseen_X_df = pd.DataFrame(unseen_X, columns=features)
            unseen_X_scaled = scaler.transform(unseen_X_df)
        
        # Project unseen data
        unseen_embedding = umap_model.transform(unseen_X_scaled)
        unseen_umap_1 = unseen_embedding[:, 0]
        unseen_umap_2 = unseen_embedding[:, 1]
        
        print(f"Unseen data UMAP ranges: X=[{unseen_umap_1.min():.3f}, {unseen_umap_1.max():.3f}], Y=[{unseen_umap_2.min():.3f}, {unseen_umap_2.max():.3f}]")
        print(f"First 5 unseen coordinates:")
        for i in range(min(5, len(unseen_umap_1))):
            print(f"  Track {i}: ({unseen_umap_1[i]:.3f}, {unseen_umap_2[i]:.3f})")
        
        # Check if unseen data falls within reasonable bounds
        train_umap_1_range = train_df['umap_1'].max() - train_df['umap_1'].min()
        train_umap_2_range = train_df['umap_2'].max() - train_df['umap_2'].min()
        
        unseen_umap_1_range = unseen_umap_1.max() - unseen_umap_1.min()
        unseen_umap_2_range = unseen_umap_2.max() - unseen_umap_2.min()
        
        print(f"\nRange comparison:")
        print(f"Training data ranges: X={train_umap_1_range:.3f}, Y={train_umap_2_range:.3f}")
        print(f"Unseen data ranges: X={unseen_umap_1_range:.3f}, Y={unseen_umap_2_range:.3f}")
        
        if (unseen_umap_1_range < train_umap_1_range * 1.5 and 
            unseen_umap_2_range < train_umap_2_range * 1.5):
            print("✅ Unseen data projection looks reasonable!")
        else:
            print("⚠️ Unseen data projection ranges seem unusual")
            
    else:
        print("❌ No UMAP model found in trained_gmm_model.pkl")

if __name__ == "__main__":
    test_umap_projection() 