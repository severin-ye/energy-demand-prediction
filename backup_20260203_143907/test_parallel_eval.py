#!/usr/bin/env python
"""测试并行模型评估"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_processing.uci_loader import load_uci_dataset
from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.models.predictor import ParallelCNNLSTMAttention
import json

# 加载配置
with open('configs/paper_config.json') as f:
    config = json.load(f)

# 加载数据
train_df, test_df = load_uci_dataset(use_splits=True)

# 预处理
preprocessor = EnergyDataPreprocessor(
    sequence_length=config['sequence_length'],
    feature_cols=config['feature_cols'],
    target_col=config['target_col']
)

X_train, y_train = preprocessor.fit_transform(train_df)
X_test, y_test = preprocessor.transform(test_df)

print(f"测试集: X={X_test.shape}, y={y_test.shape}")

# 加载已保存的模型
try:
    import tensorflow as tf
    model_keras = tf.keras.models.load_model('outputs/ablation/parallel-att/model.keras')
    print("\n成功加载模型")
    
    # 预测
    y_pred = model_keras.predict(X_test, verbose=0).flatten()
    
    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n评估结果（归一化空间）:")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.6f}")
    
    # 原始空间
    y_test_orig = preprocessor.inverse_transform_target(y_test)
    y_pred_orig = preprocessor.inverse_transform_target(y_pred)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    print(f"\n评估结果（原始空间）:")
    print(f"MAE:  {mae_orig:.4f} kW")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
