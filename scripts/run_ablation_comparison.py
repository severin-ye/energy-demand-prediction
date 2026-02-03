"""
消融实验脚本 - 加载已训练模型进行对比

实验组:
1. Serial CNN-LSTM (Baseline)
2. Serial CNN-LSTM-Attention
3. Parallel CNN-LSTM-Attention (论文方法 - 方案1)

工作流程:
1. 加载三个预训练模型
2. 在相同的测试集上评估
3. 生成对比报告

注意：
- 模型需要先用独立脚本训练好
- 确保使用相同的数据预处理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import tensorflow as tf
from tensorflow import keras

from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.data_processing.uci_loader import load_uci_dataset
from src.models.predictor import AttentionLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 固定随机种子
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("消融实验：加载预训练模型进行对比")
print("=" * 80)

# 1. 准备测试数据
print("\n[1/4] 准备测试数据...")
train_df, test_df = load_uci_dataset()
print(f"训练集大小: {len(train_df)} (用于fit预处理器)")
print(f"测试集大小: {len(test_df)}")

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

# Fit预处理器（使用训练集）
X_train_dummy, y_train_dummy = preprocessor.fit_transform(train_df)
print(f"预处理器已fit（使用训练集）")

# Transform测试集
X_test, y_test = preprocessor.transform(test_df)
print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

# 2. 加载模型
print("\n[2/4] 加载预训练模型...")

models = {}
model_paths = {
    "Serial CNN-LSTM (Baseline)": "outputs/models/serial_cnn_lstm/model.keras",
    "Serial CNN-LSTM-Attention": "outputs/models/serial_cnn_lstm_attention/model.keras",
    "Parallel CNN-LSTM-Attention": "outputs/models/parallel_cnn_lstm_attention/model.keras"
}

custom_objects = {'AttentionLayer': AttentionLayer}

for model_name, model_path in model_paths.items():
    if os.path.exists(model_path):
        try:
            models[model_name] = keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.info(f"✓ 已加载: {model_name}")
        except Exception as e:
            logger.error(f"✗ 加载失败 {model_name}: {e}")
    else:
        logger.warning(f"✗ 模型文件不存在: {model_path}")
        logger.warning(f"  请先运行相应的训练脚本")

if not models:
    print("\n❌ 没有找到任何模型！")
    print("请先运行训练脚本:")
    print("  - python scripts/train_serial_cnn_lstm.py")
    print("  - python scripts/train_serial_cnn_lstm_attention.py")
    print("  - python scripts/train_parallel_cnn_lstm_attention.py")
    sys.exit(1)

print(f"\n成功加载 {len(models)}/3 个模型")

# 3. 评估所有模型
print("\n[3/4] 评估所有模型...")

results = {}
for model_name, model in models.items():
    print(f"\n评估: {model_name}")
    
    # 预测
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    results[model_name] = {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "mape": mape,
        "params": model.count_params()
    }
    
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MSE:  {mse:.8f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  参数量: {model.count_params():,}")

# 4. 生成对比报告
print("\n[4/4] 生成对比报告...")

# 找到baseline
baseline_name = "Serial CNN-LSTM (Baseline)"
if baseline_name in results:
    baseline_mae = results[baseline_name]["mae"]
else:
    baseline_mae = None

# 创建对比表格
comparison_data = []
for model_name, metrics in results.items():
    row = {
        "模型": model_name,
        "MAE": f"{metrics['mae']:.6f}",
        "RMSE": f"{metrics['rmse']:.6f}",
        "MSE": f"{metrics['mse']:.8f}",
        "MAPE": f"{metrics['mape']:.2f}%",
        "参数量": f"{metrics['params']:,}"
    }
    
    if baseline_mae:
        improvement = ((baseline_mae - metrics['mae']) / baseline_mae) * 100
        row["vs Baseline"] = f"{improvement:+.2f}%"
    
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# 输出到控制台
print("\n" + "=" * 80)
print("消融实验结果对比")
print("=" * 80)
print(df_comparison.to_string(index=False))
print("=" * 80)

# 保存到文件
output_dir = "outputs/ablation"
os.makedirs(output_dir, exist_ok=True)

# 保存CSV
csv_path = os.path.join(output_dir, "ablation_comparison.csv")
df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✓ CSV已保存: {csv_path}")

# 保存JSON
json_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_samples": len(X_test),
    "models": results,
    "baseline": baseline_name if baseline_name in results else None
}

json_path = os.path.join(output_dir, "ablation_comparison.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_results, f, indent=2, ensure_ascii=False)
print(f"✓ JSON已保存: {json_path}")

# 生成Markdown报告
md_path = os.path.join(output_dir, "ABLATION_COMPARISON.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# 消融实验对比报告\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"测试样本数: {len(X_test):,}\n\n")
    f.write("## 实验结果\n\n")
    f.write(df_comparison.to_markdown(index=False))
    f.write("\n\n## 关键发现\n\n")
    
    if len(results) >= 3:
        serial_mae = results.get("Serial CNN-LSTM (Baseline)", {}).get("mae")
        serial_att_mae = results.get("Serial CNN-LSTM-Attention", {}).get("mae")
        parallel_mae = results.get("Parallel CNN-LSTM-Attention", {}).get("mae")
        
        if serial_mae and parallel_mae:
            improvement = ((serial_mae - parallel_mae) / serial_mae) * 100
            f.write(f"- **并行模型 vs 串行Baseline**: {improvement:+.2f}%\n")
        
        if serial_att_mae and parallel_mae:
            diff = ((serial_att_mae - parallel_mae) / serial_att_mae) * 100
            f.write(f"- **并行模型 vs 串行+Attention**: {diff:+.2f}%\n")
    
    f.write("\n## 论文对比\n\n")
    f.write("- 论文声称: 并行架构相比串行提升6.85% (单分辨率) / 34.84% (多分辨率)\n")
    f.write("- 本实验: 见上述对比结果\n")

print(f"✓ Markdown报告已保存: {md_path}")

print("\n" + "=" * 80)
print("消融实验完成！")
print("=" * 80)
