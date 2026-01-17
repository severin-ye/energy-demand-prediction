"""
Simple Test Script - Verifying Core Modules
"""

import sys
sys.path.insert(0, '/home/severin/Codelib/YS')

import numpy as np

print("=" * 60)
print("Testing Core Modules")
print("=" * 60)

# Test 1: Data Preprocessor
print("\n[1] Testing Data Preprocessor...")
try:
    from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
    import pandas as pd
    
    # Create simulated data
    df = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=200, freq='15min'),
        'GlobalActivePower': np.random.randn(200) * 0.5 + 2.0,
        'Voltage': np.random.randn(200) * 5 + 240,
        'Kitchen': np.random.randn(200) * 0.3 + 0.5
    })
    
    preprocessor = EnergyDataPreprocessor(
        sequence_length=20,
        feature_cols=['GlobalActivePower', 'Voltage', 'Kitchen'],
        target_col='GlobalActivePower'
    )
    
    X, y = preprocessor.fit_transform(df)
    print(f"✓ Data preprocessing successful! X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"✗ Data preprocessing failed: {e}")

# Test 2: Prediction Model
print("\n[2] Testing Prediction Model...")
try:
    from src.models.predictor import ParallelCNNLSTMAttention, AttentionLayer
    
    model = ParallelCNNLSTMAttention(
        input_shape=(20, 3),
        cnn_filters=32,
        lstm_units=64
    )
    
    print(f"✓ Model construction successful! Parameters: {model.model.count_params():,}")
except Exception as e:
    print(f"✗ Model construction failed: {e}")

# Test 3: State Classifier
print("\n[3] Testing State Classifier...")
try:
    from src.models.state_classifier import SnStateClassifier
    
    data = np.random.randn(100) * 2 + 3
    classifier = SnStateClassifier(n_states=3)
    states = classifier.fit_predict(data)
    
    unique, counts = np.unique(states, return_counts=True)
    print(f"✓ State classification successful! Distribution: {dict(zip(unique, counts))}")
except Exception as e:
    print(f"✗ State classification failed: {e}")

# Test 4: Discretizer
print("\n[4] Testing Discretizer...")
try:
    from src.models.discretizer import QuantileDiscretizer
    
    X_cont = np.random.randn(50, 3) * 10 + 50
    discretizer = QuantileDiscretizer(n_bins=4)
    X_disc = discretizer.fit_transform(X_cont)
    
    print(f"✓ Discretization successful! Example: {X_disc[0]}")
except Exception as e:
    print(f"✗ Discretization failed: {e}")

# Test 5: DLP Clustering
print("\n[5] Testing DLP Clustering...")
try:
    from src.models.clustering import DLPClusterer, AttentionClusterer
    
    cam_data = np.random.rand(50, 15)
    cam_clusterer = DLPClusterer(n_clusters=3)
    cam_labels = cam_clusterer.fit_predict(cam_data)
    
    att_data = np.random.rand(50, 20)
    att_clusterer = AttentionClusterer(n_clusters=3)
    att_labels = att_clusterer.fit_predict(att_data)
    
    print(f"✓ DLP clustering successful! CAM Clusters: {np.unique(cam_labels)}, "
          f"Attention Types: {att_clusterer.cluster_names_}")
except Exception as e:
    print(f"✗ DLP clustering failed: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)