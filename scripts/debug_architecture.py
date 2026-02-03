"""
调试脚本：检查并行和串行模型的架构差异
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from src.models.predictor import ParallelCNNLSTMAttention
from src.models.baseline_models import SerialCNNLSTM, SerialCNNLSTMAttention

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("模型架构对比分析")
print("=" * 80)

# 创建示例输入
input_shape = (80, 7)
sample_input = np.random.randn(1, 80, 7).astype(np.float32)

# 1. 串联CNN-LSTM (Baseline)
print("\n[1] 串联CNN-LSTM (Baseline)")
print("-" * 80)
serial = SerialCNNLSTM(input_shape=input_shape, cnn_filters=64, lstm_units=128, dense_units=[64, 32])
serial.model.summary()

# 测试维度变化
print("\n维度变化追踪:")
print(f"输入: {sample_input.shape}")

# 手动追踪CNN输出
from tensorflow.keras import layers
test_input = layers.Input(shape=input_shape)
x = layers.Conv1D(64, 3, padding='same', activation='relu')(test_input)
print(f"Conv1D(64) 后: {x.shape}")
x = layers.MaxPooling1D(2)(x)
print(f"MaxPool(2) 后: {x.shape}")
x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
print(f"Conv1D(128) 后: {x.shape}")
x = layers.MaxPooling1D(2)(x)
print(f"MaxPool(2) 后 (CNN输出): {x.shape}")

lstm_out = layers.LSTM(128, return_sequences=False)(x)
print(f"LSTM(128, return_sequences=False) 后: {lstm_out.shape}")

# 2. 串联CNN-LSTM-Attention
print("\n" + "=" * 80)
print("[2] 串联CNN-LSTM-Attention")
print("-" * 80)
serial_att = SerialCNNLSTMAttention(input_shape=input_shape, cnn_filters=64, lstm_units=128, attention_units=64, dense_units=[64, 32])
serial_att.model.summary()

print("\n维度变化追踪:")
print(f"输入: {sample_input.shape}")
print(f"CNN输出: (batch, 20, 128)")
lstm_out_seq = layers.LSTM(128, return_sequences=True)(x)
print(f"LSTM(128, return_sequences=True) 后: {lstm_out_seq.shape}")
print(f"Attention 后: (batch, 128)")

# 3. 并行CNN-LSTM-Attention
print("\n" + "=" * 80)
print("[3] 并行CNN-LSTM-Attention")
print("-" * 80)
parallel = ParallelCNNLSTMAttention(input_shape=input_shape, cnn_filters=64, lstm_units=128, attention_units=64, dense_units=[64, 32])
parallel.model.summary()

print("\n维度变化追踪:")
print(f"输入: {sample_input.shape}")
print("\nCNN分支:")
print(f"  CNN输出: (batch, 20, 128)")
print(f"  Flatten后: (batch, 2560)")
print("\nLSTM分支:")
print(f"  输入: (batch, 80, 7)  ← 注意！从原始输入")
lstm_from_raw = layers.LSTM(128, return_sequences=True)(test_input)
print(f"  LSTM(128, return_sequences=True) 后: {lstm_from_raw.shape}")
print(f"  Attention 后: (batch, 128)")
print("\n特征融合:")
print(f"  Concatenate([2560, 128]) = (batch, 2688)")

# 4. 关键差异分析
print("\n" + "=" * 80)
print("关键差异分析")
print("=" * 80)

print("\n🔍 **问题发现**：")
print("\n串联模型:")
print("  LSTM输入: CNN处理后的特征 (batch, 20, 128)")
print("  - 序列长度: 20 (已被CNN池化缩短)")
print("  - 特征维度: 128 (CNN提取的高级特征)")
print("  - LSTM学习: CNN特征之间的时序关系")

print("\n并行模型:")
print("  LSTM输入: 原始输入 (batch, 80, 7)")
print("  - 序列长度: 80 (完整的时间步)")
print("  - 特征维度: 7 (原始特征)")
print("  - LSTM学习: 原始数据的时序关系")

print("\n❌ **潜在问题**：")
print("1. LSTM处理的序列长度不同：")
print("   - 串联: 20步 (更容易学习短期依赖)")
print("   - 并行: 80步 (更难学习长期依赖)")

print("\n2. LSTM输入的信息密度不同：")
print("   - 串联: 128维高级特征 (CNN已提取模式)")
print("   - 并行: 7维原始特征 (需要LSTM自己提取)")

print("\n3. 参数利用效率：")
print("   - 串联LSTM: 在CNN特征基础上工作，更高效")
print("   - 并行LSTM: 需要从头学习，任务更重")

print("\n💡 **论文的真正意图可能是**：")
print("   并行结构应该让CNN和LSTM在**不同抽象层次**上工作，")
print("   而不是让它们处理**完全不同的输入长度和特征**。")
print("=" * 80)
