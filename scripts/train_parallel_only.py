#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只训练并行CNN-LSTM-Attention模型（快速验证）
"""

import sys
import os
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import random

# 固定随机种子（确保实验可重复）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.models.predictor import ParallelCNNLSTMAttention
from src.data_processing.uci_loader import load_uci_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载配置
config_path = project_root / 'configs' / 'paper_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

logger.info("=== 开始训练并行CNN-LSTM-Attention模型 ===")

# 1. 加载数据
logger.info("加载UCI数据集...")
train_df, test_df = load_uci_dataset(use_splits=True)

# 2. 预处理
logger.info("预处理数据...")
preprocessor = EnergyDataPreprocessor(
    sequence_length=config['sequence_length'],
    feature_cols=config['feature_cols'],
    target_col=config['target_col']
)

X_train, y_train = preprocessor.fit_transform(train_df)
X_test, y_test = preprocessor.transform(test_df)

logger.info(f"训练集: X={X_train.shape}, y={y_train.shape}")
logger.info(f"测试集: X={X_test.shape}, y={y_test.shape}")

# 3. 构建并行模型
logger.info("构建并行CNN-LSTM-Attention模型...")
model = ParallelCNNLSTMAttention(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    cnn_filters=64,  # 使用第一个卷积层的filter数
    lstm_units=config['lstm_units'],
    attention_units=config['attention_units'],
    dense_units=config['dense_units']
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
    loss='mse',
    metrics=['mae', 'mape']
)

model.model.summary()

# 4. 训练
logger.info("开始训练...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# 5. 评估（归一化空间）
logger.info("评估模型（归一化空间 [0,1]）...")
y_pred_norm = model.model.predict(X_test, verbose=0).flatten()

mae_norm = mean_absolute_error(y_test, y_pred_norm)
mse_norm = mean_squared_error(y_test, y_pred_norm)
rmse_norm = np.sqrt(mse_norm)
mape_norm = np.mean(np.abs((y_test - y_pred_norm) / (y_test + 1e-8))) * 100

logger.info(f"归一化空间指标:")
logger.info(f"  MAE:  {mae_norm:.6f}")
logger.info(f"  MSE:  {mse_norm:.8f}")
logger.info(f"  RMSE: {rmse_norm:.6f}")
logger.info(f"  MAPE: {mape_norm:.2f}%")

# 6. 评估（原始空间 kW）
logger.info("评估模型（原始空间 kW）...")
y_test_orig = preprocessor.inverse_transform_target(y_test)
y_pred_orig = preprocessor.inverse_transform_target(y_pred_norm)

mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
rmse_orig = np.sqrt(mse_orig)
mape_orig = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-8))) * 100

logger.info(f"原始空间指标:")
logger.info(f"  MAE:  {mae_orig:.4f} kW")
logger.info(f"  MSE:  {mse_orig:.4f} kW²")
logger.info(f"  RMSE: {rmse_orig:.4f} kW")
logger.info(f"  MAPE: {mape_orig:.2f}%")

# 7. 与论文基线对比
paper_baseline_mae = 0.03895  # 论文Table 3基线
paper_proposed_mae = 0.03628  # 论文Table 3提出方法
paper_improvement = (paper_baseline_mae - paper_proposed_mae) / paper_baseline_mae * 100

logger.info("\n=== 与论文对比 ===")
logger.info(f"论文基线MAE:  {paper_baseline_mae:.5f}")
logger.info(f"论文提出MAE:  {paper_proposed_mae:.5f}")
logger.info(f"论文提升率:   {paper_improvement:.2f}%")
logger.info(f"当前模型MAE:  {mae_norm:.5f}")

# 计算相对于论文基线的提升
if mae_norm < paper_baseline_mae:
    improvement = (paper_baseline_mae - mae_norm) / paper_baseline_mae * 100
    logger.info(f"相对论文基线提升: {improvement:.2f}%")
    if improvement >= 6.0:
        logger.info("✅ 达到预期性能（>6%提升）")
    else:
        logger.info(f"⚠️  未达到论文水平（期望>6%，实际{improvement:.2f}%）")
else:
    degradation = (mae_norm - paper_baseline_mae) / paper_baseline_mae * 100
    logger.info(f"❌ 性能下降: {degradation:.2f}%")

logger.info("\n训练完成！")
