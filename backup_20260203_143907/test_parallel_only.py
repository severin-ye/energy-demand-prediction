"""
单独测试并行CNN-LSTM-Attention模型
确保验证集从训练集中划分，不使用测试集
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.models.predictor import ParallelCNNLSTMAttention
from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.data_processing.uci_loader import load_uci_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from datetime import datetime
import json

# 固定随机种子
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 创建输出目录
timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
output_dir = f"outputs/parallel_test/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("测试并行CNN-LSTM-Attention模型（方案1：无CNN压缩）")
print("=" * 80)

# 1. 加载数据
print("\n[1/5] 加载数据...")
train_df, test_df = load_uci_dataset()
print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

# 2. 数据预处理
print("\n[2/5] 数据预处理...")
preprocessor = EnergyDataPreprocessor(
    sequence_length=80,
    target_col='Global_active_power',
    feature_cols=[
        'Global_active_power',
        'Global_reactive_power', 
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]
)

# 处理训练集和测试集
X_train_full, y_train_full = preprocessor.fit_transform(train_df)
X_test, y_test = preprocessor.transform(test_df)

print(f"训练数据形状: X={X_train_full.shape}, y={y_train_full.shape}")
print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

# 3. 从训练集中划分验证集（20%）
print("\n[3/5] 划分验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

print(f"训练集: X={X_train.shape}, y={y_train.shape}")
print(f"验证集: X={X_val.shape}, y={y_val.shape}")
print(f"测试集: X={X_test.shape}, y={y_test.shape}")

# 4. 构建并训练模型
print("\n[4/5] 构建并训练并行模型...")
model = ParallelCNNLSTMAttention(
    input_shape=(80, 7),
    cnn_filters=64,
    lstm_units=128,
    attention_units=64,
    dense_units=[64, 32]
)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

print(f"模型参数量: {model.model.count_params():,}")

# 训练
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ],
    verbose=1
)

# 5. 评估模型
print("\n[5/5] 评估模型...")

# 验证集评估
y_val_pred = model.predict(X_val).flatten()
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = root_mean_squared_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# 测试集评估
y_test_pred = model.predict(X_test).flatten()
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n" + "=" * 80)
print("评估结果")
print("=" * 80)
print(f"\n验证集 (从训练集划分):")
print(f"  MAE:  {val_mae:.6f}")
print(f"  RMSE: {val_rmse:.6f}")
print(f"  MSE:  {val_mse:.8f}")

print(f"\n测试集 (完全独立数据):")
print(f"  MAE:  {test_mae:.6f}")
print(f"  RMSE: {test_rmse:.6f}")
print(f"  MSE:  {test_mse:.8f}")

# 6. 保存结果
print("\n[6/6] 保存模型和结果...")

# 保存模型
model_path = os.path.join(output_dir, "parallel_model.keras")
model.save(model_path)
print(f"✓ 模型已保存: {model_path}")

# 保存配置和结果
results = {
    "experiment": "Parallel CNN-LSTM-Attention (方案1: 无CNN压缩)",
    "timestamp": timestamp,
    "config": {
        "sequence_length": 80,
        "lstm_units": 128,
        "attention_units": 64,
        "cnn_filters": 64,
        "dense_units": [64, 32],
        "epochs": 50,
        "batch_size": 64,
        "seed": SEED,
        "validation_split": 0.2
    },
    "model_info": {
        "parameters": model.model.count_params(),
        "architecture": "Parallel (CNN从inputs + LSTM从inputs)"
    },
    "training": {
        "epochs_trained": len(history.history['loss']),
        "best_epoch": len(history.history['loss']) - 15,  # early stopping patience
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test)
    },
    "results": {
        "validation": {
            "mae": float(val_mae),
            "rmse": float(val_rmse),
            "mse": float(val_mse)
        },
        "test": {
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "mse": float(test_mse)
        }
    },
    "comparison": {
        "baseline_mae": 0.0309,
        "improvement_vs_baseline": f"{((0.0309 - val_mae) / 0.0309 * 100):.2f}%",
        "old_parallel_mae": 0.0349,
        "improvement_vs_old": f"{((0.0349 - val_mae) / 0.0349 * 100):.2f}%"
    }
}

results_path = os.path.join(output_dir, "results.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✓ 结果已保存: {results_path}")

# 保存训练历史
history_path = os.path.join(output_dir, "training_history.json")
history_data = {
    "loss": [float(x) for x in history.history['loss']],
    "mae": [float(x) for x in history.history['mae']],
    "val_loss": [float(x) for x in history.history['val_loss']],
    "val_mae": [float(x) for x in history.history['val_mae']]
}
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history_data, f, indent=2)
print(f"✓ 训练历史已保存: {history_path}")

print("\n" + "=" * 80)
print("训练完成！")
print(f"所有文件已保存到: {output_dir}")
print("=" * 80)
