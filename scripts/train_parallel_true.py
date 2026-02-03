"""
训练真正的并行CNN-LSTM-Attention模型
严格按照论文Fig. 1实现：LSTM从原始输入开始
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.models.ablation_models import ParallelCNNLSTMAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 设置随机种子
SEED = 42
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("训练真正的并行CNN-LSTM-Attention模型（严格按照论文Fig. 1）")
    logger.info("="*80)
    
    # [1] 加载数据
    logger.info("\n[1/6] 加载数据...")
    train_df = pd.read_csv('data/uci/splits/train.csv')
    test_df = pd.read_csv('data/uci/splits/test.csv')
    logger.info(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")
    
    # [2] 数据预处理
    logger.info("\n[2/6] 数据预处理...")
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
    logger.info(f"训练数据: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"测试数据: X={X_test.shape}, y={y_test.shape}")
    
    # [3] 划分验证集
    logger.info("\n[3/6] 划分验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )
    logger.info(f"训练集: {X_train_split.shape}")
    logger.info(f"验证集: {X_val.shape}")
    
    # [4] 构建模型
    logger.info("\n[4/6] 构建并行模型...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = ParallelCNNLSTMAttention(
        input_shape=input_shape,
        cnn_filters=64,
        lstm_units=128,
        attention_units=64,
        dense_units=[64, 32]
    )
    
    # 编译
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # [5] 训练
    logger.info("\n[5/6] 开始训练...")
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
    
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # [6] 评估
    logger.info("\n[6/6] 评估模型...")
    val_metrics = model.model.evaluate(X_val, y_val, verbose=0)
    test_metrics = model.model.evaluate(X_test, y_test, verbose=0)
    
    results = {
        'model_name': 'P-CNN-LSTM-Att (True Parallel)',
        'architecture': 'LSTM from original input (80 steps)',
        'val_loss': float(val_metrics[0]),
        'val_mae': float(val_metrics[1]),
        'val_mse': float(val_metrics[2]),
        'test_loss': float(test_metrics[0]),
        'test_mae': float(test_metrics[1]),
        'test_mse': float(test_metrics[2]),
        'params': int(model.model.count_params()),
        'epochs_trained': len(history.history['loss'])
    }
    
    logger.info("\n" + "="*80)
    logger.info("最终结果")
    logger.info("="*80)
    logger.info(f"\n验证集:")
    logger.info(f"  MAE:  {results['val_mae']:.6f}")
    logger.info(f"  RMSE: {np.sqrt(results['val_mse']):.6f}")
    logger.info(f"  MSE:  {results['val_mse']:.6f}")
    
    logger.info(f"\n测试集:")
    logger.info(f"  MAE:  {results['test_mae']:.6f}")
    logger.info(f"  RMSE: {np.sqrt(results['test_mse']):.6f}")
    logger.info(f"  MSE:  {results['test_mse']:.6f}")
    
    logger.info(f"\n参数量: {results['params']:,}")
    logger.info(f"训练轮数: {results['epochs_trained']}")
    
    # 保存
    timestamp = datetime.now().strftime("%y-%m-%d")
    output_dir = Path(f'outputs/ablation/{timestamp}/P-CNN-LSTM-Att-TrueParallel')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_dir / 'model.keras'))
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    logger.info(f"\n✓ 模型已保存: {output_dir}")
    
    # 对比之前的结果
    logger.info("\n" + "="*80)
    logger.info("与之前消融实验结果对比")
    logger.info("="*80)
    
    prev_results = {
        'S-CNN-LSTM (baseline)': 0.030823,
        'S-CNN-LSTM-Att': 0.029874,
        'P-CNN-LSTM-Att (旧版-LSTM从CNN输出)': 0.035662
    }
    
    logger.info("\n验证集MAE对比:")
    for name, mae in prev_results.items():
        logger.info(f"  {name:40s}: {mae:.6f}")
    logger.info(f"  {'P-CNN-LSTM-Att (新版-LSTM从原始输入)':40s}: {results['val_mae']:.6f} ← 当前")
    
    # 计算改进
    baseline_mae = prev_results['S-CNN-LSTM (baseline)']
    improvement = (baseline_mae - results['val_mae']) / baseline_mae * 100
    
    logger.info(f"\n相比baseline改进: {improvement:+.2f}%")
    
    if results['val_mae'] < baseline_mae:
        logger.info("✅ 成功！并行模型超过了baseline！")
    else:
        logger.info("❌ 仍然比baseline差")
    
    logger.info("\n" + "="*80)
    logger.info("训练完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()
