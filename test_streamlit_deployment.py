#!/usr/bin/env python3
"""
Test script to verify Streamlit Cloud deployment readiness.
This script tests all critical components that might fail on Streamlit Cloud.
"""

import os
import sys
import traceback

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit")
    except Exception as e:
        print(f"❌ streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except Exception as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except Exception as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except Exception as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import umap
        print("✅ umap-learn")
    except Exception as e:
        print(f"❌ umap-learn: {e}")
        return False
    
    try:
        import plotly
        print("✅ plotly")
    except Exception as e:
        print(f"❌ plotly: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python")
    except Exception as e:
        print(f"❌ opencv-python: {e}")
        return False
    
    try:
        import joblib
        print("✅ joblib")
    except Exception as e:
        print(f"❌ joblib: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that the GMM model can be loaded."""
    print("\n🔍 Testing model loading...")
    
    try:
        from full_pipeline import load_model_robust
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_gmm_model.pkl")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        model_data = load_model_robust(model_path)
        
        # Check required keys
        required_keys = ['gmm', 'scaler', 'cluster_to_label', 'features']
        for key in required_keys:
            if key not in model_data:
                print(f"❌ Missing key in model: {key}")
                return False
        
        print("✅ Model loaded successfully")
        print(f"✅ Features: {model_data['features']}")
        print(f"✅ Clusters: {model_data['cluster_to_label']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_training_data():
    """Test that training data can be loaded."""
    print("\n🔍 Testing training data...")
    
    try:
        import pandas as pd
        
        if not os.path.exists("train_track_df.csv"):
            print("❌ train_track_df.csv not found")
            return False
        
        train_df = pd.read_csv("train_track_df.csv")
        print(f"✅ Training data loaded: {len(train_df)} rows")
        
        # Check for required columns
        required_cols = ['umap_1', 'umap_2', 'cluster_id', 'subtype_label']
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Training data loading failed: {e}")
        return False

def test_streamlit_functions():
    """Test that Streamlit-specific functions work."""
    print("\n🔍 Testing Streamlit functions...")
    
    try:
        from global_umap_utils import load_or_create_training_umap_data
        
        training_data = load_or_create_training_umap_data()
        
        if training_data is None:
            print("❌ Could not load training data for UMAP")
            return False
        
        print("✅ UMAP training data loaded")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit functions failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Streamlit Cloud deployment readiness...\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Training Data", test_training_data),
        ("Streamlit Functions", test_streamlit_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Your app should work on Streamlit Cloud.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the issues above before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 