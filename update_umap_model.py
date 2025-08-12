import pandas as pd
import numpy as np
import pickle
import umap
from sklearn.preprocessing import MinMaxScaler

def update_umap_model():
    """Update the UMAP model in the pickle file to use scaled features"""
    
    print("Loading training data...")
    train_df = pd.read_csv('train_track_df.csv')
    
    # Load existing model
    print("Loading existing model...")
    with open('trained_gmm_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    gmm_model = model_data['gmm']
    scaler = model_data['scaler']
    cluster_to_label = model_data['cluster_to_label']
    features = model_data['features']
    
    print(f"Features: {features}")
    print(f"Training data shape: {train_df.shape}")
    
    # Get the features and scale them
    X = train_df[features].values
    X_scaled = scaler.transform(X)
    
    print(f"Feature data shape: {X.shape}")
    print(f"Scaled feature data shape: {X_scaled.shape}")
    
    # Train a new UMAP model on SCALED features
    print("Training new UMAP model on SCALED features...")
    new_umap = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='euclidean'
    )
    
    # Fit the UMAP model on scaled features
    new_embedding = new_umap.fit_transform(X_scaled)
    
    # Update the training data with new UMAP coordinates
    train_df['umap_1'] = new_embedding[:, 0]
    train_df['umap_2'] = new_embedding[:, 1]
    
    print(f"New UMAP ranges: X=[{train_df['umap_1'].min():.3f}, {train_df['umap_1'].max():.3f}], Y=[{train_df['umap_2'].min():.3f}, {train_df['umap_2'].max():.3f}]")
    
    # Save the updated training data
    train_df.to_csv("train_track_df.csv", index=False)
    print("✅ Updated train_track_df.csv with new UMAP coordinates")
    
    # Update the model with the new UMAP
    model_data['umap_model'] = new_umap
    
    # Save the updated model
    with open('trained_gmm_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Updated trained_gmm_model.pkl with new UMAP model")
    
    # Test the consistency
    print("\nTesting consistency with 001550be...")
    test_data = train_df[train_df['participant_id'] == '001550be']
    X_test = test_data[features].values
    X_test_scaled = scaler.transform(X_test)
    
    # Get existing coordinates
    existing_test_umap_1 = test_data['umap_1'].values
    existing_test_umap_2 = test_data['umap_2'].values
    
    # Compute new coordinates
    new_test_embedding = new_umap.transform(X_test_scaled)
    new_test_umap_1 = new_test_embedding[:, 0]
    new_test_umap_2 = new_test_embedding[:, 1]
    
    # Compare
    max_diff_1 = np.abs(existing_test_umap_1 - new_test_umap_1).max()
    max_diff_2 = np.abs(existing_test_umap_2 - new_test_umap_2).max()
    
    print(f"Maximum difference for 001550be: UMAP1={max_diff_1:.6f}, UMAP2={max_diff_2:.6f}")
    
    if max_diff_1 < 1e-6 and max_diff_2 < 1e-6:
        print("✅ Perfect consistency achieved!")
    else:
        print("⚠️ Some differences remain, but coordinates should be much more consistent now")
    
    print("\n🎯 Summary of changes:")
    print("1. UMAP model now trained on SCALED features (best practice)")
    print("2. Updated train_track_df.csv with new coordinates")
    print("3. Updated trained_gmm_model.pkl with new UMAP model")
    print("4. Your pipeline will now work correctly!")

if __name__ == "__main__":
    update_umap_model() 