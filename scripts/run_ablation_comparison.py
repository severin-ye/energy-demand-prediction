"""
公平对比实验：训练3个模型
- S-CNN-LSTM (串联baseline)
- S-CNN-LSTM-Att (串联+Attention)
- P-CNN-LSTM-Att (并行+Attention，论文方法)

完全相同的训练条件，确保公平对比
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.models.ablation_models import SerialCNNLSTM, SerialCNNLSTMAttention, ParallelCNNLSTMAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 设置随机种子确保可重复性
SEED = 42
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    """加载并预处理数据"""
    logger.info("[1/6] 加载数据...")
    
    train_path = 'data/uci/splits/train.csv'
    test_path = 'data/uci/splits/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    
    return train_df, test_df


def prepare_sequences(train_df, test_df):
    """准备时间序列数据"""
    logger.info("\n[2/6] 数据预处理...")
    
    # 定义特征列
    feature_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    preprocessor = EnergyDataPreprocessor(
        sequence_length=80,
        feature_cols=feature_cols,
        target_col='Global_active_power'
    )
    
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    logger.info(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def split_validation(X_train, y_train, val_ratio=0.2):
    """划分验证集"""
    logger.info("\n[3/6] 划分验证集...")
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=SEED,
        shuffle=True
    )
    
    logger.info(f"训练集: X={X_train_split.shape}, y={y_train_split.shape}")
    logger.info(f"验证集: X={X_val.shape}, y={y_val.shape}")
    
    return X_train_split, X_val, y_train_split, y_val


def train_model(model_class, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """训练单个模型"""
    logger.info(f"\n{'='*80}")
    logger.info(f"训练模型: {model_name}")
    logger.info(f"{'='*80}")
    
    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = model_class(
        input_shape=input_shape,
        cnn_filters=64,
        lstm_units=128,
        attention_units=64,
        dense_units=[64, 32]
    )
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估
    logger.info("\n评估模型...")
    val_metrics = model.model.evaluate(X_val, y_val, verbose=0)
    test_metrics = model.model.evaluate(X_test, y_test, verbose=0)
    
    results = {
        'model_name': model_name,
        'val_loss': float(val_metrics[0]),
        'val_mae': float(val_metrics[1]),
        'val_mse': float(val_metrics[2]),
        'test_loss': float(test_metrics[0]),
        'test_mae': float(test_metrics[1]),
        'test_mse': float(test_metrics[2]),
        'params': int(model.model.count_params()),
        'epochs_trained': len(history.history['loss'])
    }
    
    logger.info(f"\n验证集:")
    logger.info(f"  Loss: {results['val_loss']:.6f}")
    logger.info(f"  MAE:  {results['val_mae']:.6f}")
    logger.info(f"  MSE:  {results['val_mse']:.6f}")
    
    logger.info(f"\n测试集:")
    logger.info(f"  Loss: {results['test_loss']:.6f}")
    logger.info(f"  MAE:  {results['test_mae']:.6f}")
    logger.info(f"  MSE:  {results['test_mse']:.6f}")
    
    # 保存模型
    timestamp = datetime.now().strftime("%y-%m-%d")
    output_dir = Path(f'outputs/ablation/{timestamp}/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_dir / 'model.keras'))
    
    # 保存结果
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存训练历史
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    logger.info(f"\n✓ 模型已保存: {output_dir}")
    
    return results


def generate_comparison_report(all_results):
    """生成对比报告"""
    logger.info(f"\n{'='*80}")
    logger.info("对比结果汇总")
    logger.info(f"{'='*80}\n")
    
    # 创建对比表格
    df = pd.DataFrame(all_results)
    
    # 按验证集MAE排序
    df = df.sort_values('val_mae')
    
    # 计算相对改进
    baseline_mae = df[df['model_name'] == 'S-CNN-LSTM']['val_mae'].values[0]
    df['improvement_vs_baseline'] = ((baseline_mae - df['val_mae']) / baseline_mae * 100)
    
    # 打印表格
    logger.info("验证集性能对比:")
    logger.info("-" * 80)
    for _, row in df.iterrows():
        logger.info(f"{row['model_name']:25s} | MAE: {row['val_mae']:.6f} | "
                   f"MSE: {row['val_mse']:.6f} | 参数: {row['params']:,} | "
                   f"改进: {row['improvement_vs_baseline']:+.2f}%")
    
    logger.info("\n测试集性能对比:")
    logger.info("-" * 80)
    for _, row in df.iterrows():
        logger.info(f"{row['model_name']:25s} | MAE: {row['test_mae']:.6f} | "
                   f"MSE: {row['test_mse']:.6f}")
    
    # 保存对比报告
    timestamp = datetime.now().strftime("%y-%m-%d")
    output_dir = Path(f'outputs/ablation/{timestamp}')
    
    df.to_csv(output_dir / 'comparison.csv', index=False)
    
    # 生成Markdown报告
    with open(output_dir / 'COMPARISON_REPORT.md', 'w') as f:
        f.write("# 公平对比实验报告\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**随机种子**: {SEED}\n\n")
        
        f.write("## 验证集性能\n\n")
        f.write("| 模型 | MAE | MSE | 参数量 | vs Baseline |\n")
        f.write("|------|-----|-----|--------|-------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['model_name']} | {row['val_mae']:.6f} | "
                   f"{row['val_mse']:.6f} | {row['params']:,} | "
                   f"{row['improvement_vs_baseline']:+.2f}% |\n")
        
        f.write("\n## 测试集性能\n\n")
        f.write("| 模型 | MAE | MSE |\n")
        f.write("|------|-----|-----|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['model_name']} | {row['test_mae']:.6f} | "
                   f"{row['test_mse']:.6f} |\n")
        
        f.write("\n## 分析\n\n")
        best_model = df.iloc[0]
        f.write(f"**最佳模型**: {best_model['model_name']}\n\n")
        f.write(f"- 验证集MAE: {best_model['val_mae']:.6f}\n")
        f.write(f"- 测试集MAE: {best_model['test_mae']:.6f}\n")
        f.write(f"- 相比baseline提升: {best_model['improvement_vs_baseline']:.2f}%\n")
        f.write(f"- 参数量: {best_model['params']:,}\n")
    
    logger.info(f"\n✓ 对比报告已保存: {output_dir}/COMPARISON_REPORT.md")
    
    return df


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("公平对比实验：S-CNN-LSTM vs S-CNN-LSTM-Att vs P-CNN-LSTM-Att")
    logger.info("="*80)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    
    # 准备序列
    X_train, y_train, X_test, y_test = prepare_sequences(train_df, test_df)
    
    # 划分验证集
    X_train_split, X_val, y_train_split, y_val = split_validation(X_train, y_train)
    
    # 定义要训练的模型
    models_to_train = [
        (SerialCNNLSTM, 'S-CNN-LSTM'),
        (SerialCNNLSTMAttention, 'S-CNN-LSTM-Att'),
        (ParallelCNNLSTMAttention, 'P-CNN-LSTM-Att')
    ]
    
    # 训练所有模型
    logger.info("\n[4/6] 开始训练所有模型...")
    all_results = []
    
    for i, (model_class, model_name) in enumerate(models_to_train, 1):
        logger.info(f"\n进度: {i}/{len(models_to_train)}")
        results = train_model(
            model_class, model_name,
            X_train_split, y_train_split,
            X_val, y_val,
            X_test, y_test
        )
        all_results.append(results)
    
    # 生成对比报告
    logger.info("\n[5/6] 生成对比报告...")
    comparison_df = generate_comparison_report(all_results)
    
    logger.info("\n[6/6] 实验完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()
