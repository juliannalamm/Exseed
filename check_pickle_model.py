import pickle
import pandas as pd
import numpy as np

def check_pickle_model():
    """Check what's actually in the pickle file"""
    
    print("Loading pickle file...")
    with open('trained_gmm_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print("Keys in pickle file:")
    for key in model_data.keys():
        print(f"  - {key}")
    
    print(f"\nUMAP model type: {type(model_data.get('umap_model'))}")
    
    if model_data.get('umap_model') is not None:
        umap_model = model_data['umap_model']
        print(f"UMAP model parameters:")
        print(f"  - n_neighbors: {umap_model.n_neighbors}")
        print(f"  - min_dist: {umap_model.min_dist}")
        print(f"  - n_components: {umap_model.n_components}")
        print(f"  - random_state: {umap_model.random_state}")
        print(f"  - metric: {umap_model.metric}")
        
        # Test the model on a small sample
        print(f"\nTesting UMAP model...")
        
        # Load a small sample of training data
        train_df = pd.read_csv('train_track_df.csv')
        sample_data = train_df.head(10)
        
        features = model_data['features']
        scaler = model_data['scaler']
        
        X_sample = sample_data[features].values
        X_sample_scaled = scaler.transform(X_sample)
        
        # Get existing coordinates
        existing_umap_1 = sample_data['umap_1'].values
        existing_umap_2 = sample_data['umap_2'].values
        
        # Compute new coordinates
        new_embedding = umap_model.transform(X_sample_scaled)
        new_umap_1 = new_embedding[:, 0]
        new_umap_2 = new_embedding[:, 1]
        
        print(f"Sample test results:")
        print(f"  Existing UMAP ranges: X=[{existing_umap_1.min():.3f}, {existing_umap_1.max():.3f}], Y=[{existing_umap_2.min():.3f}, {existing_umap_2.max():.3f}]")
        print(f"  New UMAP ranges: X=[{new_umap_1.min():.3f}, {new_umap_1.max():.3f}], Y=[{new_umap_2.min():.3f}, {new_umap_2.max():.3f}]")
        
        # Check if they match
        max_diff_1 = np.abs(existing_umap_1 - new_umap_1).max()
        max_diff_2 = np.abs(existing_umap_2 - new_umap_2).max()
        
        print(f"  Maximum differences: UMAP1={max_diff_1:.6f}, UMAP2={max_diff_2:.6f}")
        
        if max_diff_1 < 1e-6 and max_diff_2 < 1e-6:
            print("  ✅ UMAP model matches training data!")
        else:
            print("  ❌ UMAP model does NOT match training data!")
    else:
        print("❌ No UMAP model found in pickle file!")

if __name__ == "__main__":
    check_pickle_model() 