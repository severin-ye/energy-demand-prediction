"""
快速验证脚本 - 使用小数据集验证完整流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("快速验证：使用小数据集测试完整流程")
print("=" * 80)

# 1. 加载配置
print("\n[步骤1] 加载配置")
with open('configs/paper_config.json', 'r') as f:
    config = json.load(f)

print(f"✅ 配置加载成功")
print(f"   序列长度: {config['sequence_length']}")
print(f"   LSTM单元: {config['lstm_units']}")

# 2. 加载数据（只用小样本）
print("\n[步骤2] 加载数据（小样本）")

try:
    train_df = pd.read_csv('data/uci/splits/train.csv')
    test_df = pd.read_csv('data/uci/splits/test.csv')
    
    # 只用前1000条训练，前200条测试
    train_df = train_df.head(1000)
    test_df = test_df.head(200)
    
    print(f"✅ 数据加载成功")
    print(f"   训练样本: {len(train_df)}")
    print(f"   测试样本: {len(test_df)}")
    
except Exception as e:
    logger.error(f"数据加载失败: {e}")
    sys.exit(1)

# 3. 数据预处理
print("\n[步骤3] 数据预处理")

from src.preprocessing.data_preprocessor import EnergyDataPreprocessor

preprocessor = EnergyDataPreprocessor(
    sequence_length=config['sequence_length'],
    feature_cols=config['feature_cols'],
    target_col='Global_active_power'
)

X_train, y_train = preprocessor.fit_transform(train_df)
X_test, y_test = preprocessor.transform(test_df)

print(f"✅ 预处理完成")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_test: {y_test.shape}")

# 4. 测试模型训练（3个epoch）
print("\n[步骤4] 快速训练测试（3个epoch）")

from src.models.predictor import ParallelCNNLSTMAttention
from src.models.baseline_models import SerialCNNLSTM

input_shape = (X_train.shape[1], X_train.shape[2])

# 4.1 并行模型
print("\n[4.1] 训练并行CNN-LSTM-Attention")
model_parallel = ParallelCNNLSTMAttention(
    input_shape=input_shape,
    cnn_filters=config['cnn_filters'][0],
    lstm_units=config['lstm_units'],
    attention_units=config['attention_units'],
    dense_units=config['dense_units']
)

model_parallel.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_parallel = model_parallel.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32,
    verbose=1
)

print(f"✅ 并行模型训练完成")

# 4.2 串联基线
print("\n[4.2] 训练串联CNN-LSTM（基线）")
model_serial = SerialCNNLSTM(
    input_shape=input_shape,
    cnn_filters=config['cnn_filters'][0],
    lstm_units=config['lstm_units'],
    dense_units=config['dense_units']
)

model_serial.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_serial = model_serial.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32,
    verbose=1
)

print(f"✅ 串联基线训练完成")

# 5. 评估对比
print("\n[步骤5] 评估对比")

y_pred_parallel = model_parallel.predict(X_test).flatten()
y_pred_serial = model_serial.predict(X_test).flatten()

def compute_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    
    # MAPE
    mask = y_true > 0.01
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    print(f"\n{name}:")
    print(f"   MAE:  {mae:.4f} kW")
    print(f"   RMSE: {rmse:.4f} kW")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'mse': mse, 'mape': mape}

metrics_parallel = compute_metrics(y_test, y_pred_parallel, "并行CNN-LSTM-Attention")
metrics_serial = compute_metrics(y_test, y_pred_serial, "串联CNN-LSTM（基线）")

# 计算提升
improvement_mae = (metrics_serial['mae'] - metrics_parallel['mae']) / metrics_serial['mae'] * 100
improvement_rmse = (metrics_serial['rmse'] - metrics_parallel['rmse']) / metrics_serial['rmse'] * 100

print(f"\n性能提升（相对基线）:")
print(f"   MAE提升:  {improvement_mae:+.2f}%")
print(f"   RMSE提升: {improvement_rmse:+.2f}%")

# 6. 测试状态分类
print("\n[步骤6] 测试状态分类")

from src.models.state_classifier import SnStateClassifier

classifier = SnStateClassifier(threshold=2.0)
classifier.fit(y_train)

states_parallel = classifier.predict(y_pred_parallel[:10])
states_serial = classifier.predict(y_pred_serial[:10])

print(f"✅ 状态分类完成")
print(f"\n前10个样本的状态预测（并行模型）:")
for i in range(min(10, len(states_parallel))):
    print(f"   样本{i}: 预测={y_pred_parallel[i]:.3f}kW, 真值={y_test[i]:.3f}kW, 状态={states_parallel[i]}")

# 7. 测试DLP提取
print("\n[步骤7] 测试深度学习参数提取")

cam = model_parallel.extract_cam(X_test[:5])
attention = model_parallel.extract_attention_weights(X_test[:5])

print(f"✅ DLP提取成功")
print(f"   CAM形状: {cam.shape}")
print(f"   Attention形状: {attention.shape}")

# 总结
print("\n" + "=" * 80)
print("快速验证总结")
print("=" * 80)

print(f"""
✅ 所有流程验证通过：
   1. 配置加载正常
   2. 数据预处理正常
   3. 并行模型训练成功（MAE: {metrics_parallel['mae']:.4f}）
   4. 串联基线训练成功（MAE: {metrics_serial['mae']:.4f}）
   5. 性能提升: MAE {improvement_mae:+.2f}%, RMSE {improvement_rmse:+.2f}%
   6. 状态分类正常
   7. DLP提取正常

📝 说明：
   - 这是使用小数据集（1000训练+200测试）的快速验证
   - 仅训练3个epoch用于测试流程
   - MAPE可能较高是因为样本少且训练不充分
   
🚀 下一步：
   如需完整实验，运行: 
   python scripts/run_ablation_study.py
   
   （注意：完整实验需要更长时间）
""")
