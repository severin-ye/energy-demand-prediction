"""
完整训练脚本 - 支持合成数据和UCI真实数据集
"""

import sys
import os
import argparse
sys.path.append('/home/severin/Codelib/YS')

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from src.pipeline.train_pipeline import TrainPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def detect_data_type(data_path):
    """
    自动检测数据类型（合成数据 vs UCI数据）
    
    Returns:
        str: 'synthetic' 或 'uci'
    """
    if 'uci' in data_path.lower():
        return 'uci'
    elif 'synthetic' in data_path.lower():
        return 'synthetic'
    else:
        # 读取一小部分数据检查列名
        df = pd.read_csv(data_path, nrows=5)
        if 'Global_active_power' in df.columns:
            return 'uci'
        elif 'EDP' in df.columns:
            return 'synthetic'
        else:
            raise ValueError(f"无法识别数据类型，列名: {df.columns.tolist()}")


def prepare_uci_data(data_path):
    """
    准备UCI数据集用于训练
    
    UCI数据集特征：
    - Global_active_power, Global_reactive_power, Voltage, Global_intensity
    - Sub_metering_1, Sub_metering_2, Sub_metering_3
    - hour, day_of_week, month, is_weekend
    
    返回格式需要匹配模型输入
    """
    logger.info("加载UCI数据集...")
    df = pd.read_csv(data_path)
    
    # 选择用于训练的特征
    feature_cols = [
        'Global_reactive_power',  # 无功功率
        'Voltage',                # 电压
        'Global_intensity',       # 电流强度
    ]
    
    target_col = 'Global_active_power'  # 有功功率作为目标
    
    # 检查必要列是否存在
    required_cols = feature_cols + [target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"数据集缺少必要列: {missing}")
    
    # 准备数据
    prepared_df = df[[target_col] + feature_cols].copy()
    
    # 重命名目标列为EDP（兼容现有代码）
    prepared_df = prepared_df.rename(columns={target_col: 'EDP'})
    
    logger.info(f"✅ UCI数据准备完成: {prepared_df.shape}")
    logger.info(f"   特征列: {feature_cols}")
    logger.info(f"   目标列: EDP (原{target_col})")
    logger.info(f"   EDP范围: [{prepared_df['EDP'].min():.2f}, {prepared_df['EDP'].max():.2f}]")
    
    return prepared_df, feature_cols, 'EDP'


def prepare_synthetic_data(data_path):
    """准备合成数据集"""
    logger.info("加载合成数据集...")
    df = pd.read_csv(data_path)
    
    feature_cols = ['Temperature', 'Humidity', 'WindSpeed']
    target_col = 'EDP'
    
    logger.info(f"✅ 合成数据加载完成: {df.shape}")
    
    return df, feature_cols, target_col


def main():
    parser = argparse.ArgumentParser(description='训练能源预测模型')
    parser.add_argument(
        '--data',
        default='data/uci/splits/train.csv',
        help='训练数据路径（默认使用UCI训练集）'
    )
    parser.add_argument(
        '--data-type',
        choices=['auto', 'uci', 'synthetic'],
        default='auto',
        help='数据类型（auto自动检测）'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='输出目录（默认: outputs/training/YY-MM-DD）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='批次大小'
    )
    
    args = parser.parse_args()
    
    # 如果未指定输出目录，使用 outputs/training/日期/ 格式
    if args.output_dir is None:
        date_suffix = datetime.now().strftime('%y-%m-%d')
        args.output_dir = f'./outputs/training/{date_suffix}'
    
    logger.info("="*80)
    logger.info(" "*20 + "完整训练流水线" + " "*20)
    logger.info("="*80)
    
    start_time = time.time()
    
    # 1. 检测数据类型
    if args.data_type == 'auto':
        data_type = detect_data_type(args.data)
        logger.info(f"\n[自动检测] 数据类型: {data_type}")
    else:
        data_type = args.data_type
    
    # 2. 加载并准备数据
    logger.info(f"\n[步骤 1] 加载训练数据: {args.data}...")
    
    try:
        if data_type == 'uci':
            train_data, feature_cols, target_col = prepare_uci_data(args.data)
        else:
            train_data, feature_cols, target_col = prepare_synthetic_data(args.data)
            
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {args.data}")
        if data_type == 'uci':
            logger.info("请先运行: python scripts/split_uci_dataset.py")
        else:
            logger.info("请先运行: python scripts/prepare_data.py")
        return
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 2. 配置训练参数
    logger.info("\n[步骤 2] 配置训练参数...")
    
    config = {
        # 数据预处理
        'sequence_length': 20,
        'feature_cols': feature_cols,
        'target_col': target_col,
        
        # 预测模型
        'cnn_filters': [64, 32] if data_type == 'uci' else [32, 16],
        'lstm_units': 64 if data_type == 'uci' else 32,
        'attention_units': 25,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_split': 0.2,
        
        # 状态分类
        'n_states': 3,
        'state_names': ['Lower', 'Normal', 'Peak'],
        
        # 离散化
        'n_bins': 4,
        'bin_labels': ['Low', 'Medium', 'High', 'VeryHigh'],
        
        # DLP聚类
        'n_cam_clusters': 3,
        'n_attention_clusters': 3,
        
        # 关联规则
        'min_support': 0.05,
        'min_confidence': 0.6,
        'min_lift': 1.2,
        
        # 贝叶斯网络
        'bn_score_fn': 'bic',
        'bn_max_iter': 50,
        'bn_estimator': 'mle'
    }
    
    logger.info(f"数据类型: {data_type}")
    logger.info(f"训练配置: epochs={config['epochs']}, batch_size={config['batch_size']}")
    logger.info(f"模型规模: CNN={config['cnn_filters']}, LSTM={config['lstm_units']}")
    
    # 3. 创建训练流水线
    logger.info("\n[步骤 3] 创建训练流水线...")
    pipeline = TrainPipeline(
        config=config,
        output_dir=args.output_dir
    )
    
    # 4. 运行训练
    logger.info("\n[步骤 4] 开始训练...\n")
    
    try:
        results = pipeline.run(train_data)
        
        # 5. 输出结果
        logger.info("\n" + "="*80)
        logger.info(" "*30 + "训练完成！" + " "*30)
        logger.info("="*80)
        
        logger.info("\n训练结果摘要:")
        logger.info(f"  数据形状: {results['data_shapes']}")
        logger.info(f"  DLP形状: {results['dlp_shapes']}")
        logger.info(f"  状态分布: {results['state_distribution']}")
        logger.info(f"  聚类分布:")
        logger.info(f"    CAM: {results['cluster_distributions']['cam']}")
        logger.info(f"    Attention: {results['cluster_distributions']['attention']}")
        logger.info(f"  候选边数量: {len(results['candidate_edges'])}")
        logger.info(f"  贝叶斯网络边: {len(results['bn_edges'])}")
        
        # 计算训练时间
        elapsed_time = time.time() - start_time
        logger.info(f"\n总训练时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
        
        logger.info("\n输出文件:")
        logger.info(f"  模型: {args.output_dir}/models/")
        logger.info(f"  结果: {args.output_dir}/results/")
        logger.info(f"  配置: {args.output_dir}/config.json")
        return results
        
    except Exception as e:
        logger.error(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n" + "="*80)
        print("下一步：运行推理测试")
        print("  python scripts/run_inference.py")
        print("="*80)
