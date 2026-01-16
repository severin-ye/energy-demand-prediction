"""
简单测试脚本 - 验证核心模块
"""

import sys
sys.path.insert(0, '/home/severin/Codelib/YS')

import numpy as np

print("=" * 60)
print("测试核心模块")
print("=" * 60)

# 测试1: 数据预处理器
print("\n[1] 测试数据预处理器...")
try:
    from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
    import pandas as pd
    
    # 创建模拟数据
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
    print(f"✓ 数据预处理成功！X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"✗ 数据预处理失败: {e}")

# 测试2: 预测模型
print("\n[2] 测试预测模型...")
try:
    from src.models.predictor import ParallelCNNLSTMAttention, AttentionLayer
    
    model = ParallelCNNLSTMAttention(
        input_shape=(20, 3),
        cnn_filters=32,
        lstm_units=64
    )
    
    print(f"✓ 模型构建成功！参数量: {model.model.count_params():,}")
except Exception as e:
    print(f"✗ 模型构建失败: {e}")

# 测试3: 状态分类器
print("\n[3] 测试状态分类器...")
try:
    from src.models.state_classifier import SnStateClassifier
    
    data = np.random.randn(100) * 2 + 3
    classifier = SnStateClassifier(n_states=3)
    states = classifier.fit_predict(data)
    
    unique, counts = np.unique(states, return_counts=True)
    print(f"✓ 状态分类成功！分布: {dict(zip(unique, counts))}")
except Exception as e:
    print(f"✗ 状态分类失败: {e}")

# 测试4: 离散化器
print("\n[4] 测试离散化器...")
try:
    from src.models.discretizer import QuantileDiscretizer
    
    X_cont = np.random.randn(50, 3) * 10 + 50
    discretizer = QuantileDiscretizer(n_bins=4)
    X_disc = discretizer.fit_transform(X_cont)
    
    print(f"✓ 离散化成功！示例: {X_disc[0]}")
except Exception as e:
    print(f"✗ 离散化失败: {e}")

# 测试5: DLP聚类
print("\n[5] 测试DLP聚类...")
try:
    from src.models.clustering import DLPClusterer, AttentionClusterer
    
    cam_data = np.random.rand(50, 15)
    cam_clusterer = DLPClusterer(n_clusters=3)
    cam_labels = cam_clusterer.fit_predict(cam_data)
    
    att_data = np.random.rand(50, 20)
    att_clusterer = AttentionClusterer(n_clusters=3)
    att_labels = att_clusterer.fit_predict(att_data)
    
    print(f"✓ DLP聚类成功！CAM聚类: {np.unique(cam_labels)}, "
          f"Attention类型: {att_clusterer.cluster_names_}")
except Exception as e:
    print(f"✗ DLP聚类失败: {e}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
