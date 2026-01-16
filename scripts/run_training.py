"""
完整训练脚本 - 使用合成数据集训练全部模型
"""

import sys
sys.path.append('/home/severin/Codelib/YS')

import pandas as pd
import numpy as np
import logging
import time
from src.pipeline.train_pipeline import TrainPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info(" "*20 + "完整训练流水线" + " "*20)
    logger.info("="*80)
    
    start_time = time.time()
    
    # 1. 加载数据
    logger.info("\n[步骤 1] 加载训练数据...")
    data_path = 'data/processed/synthetic_energy_data.csv'
    
    try:
        train_data = pd.read_csv(data_path)
        logger.info(f"✅ 数据加载成功: {train_data.shape}")
        logger.info(f"特征列: {list(train_data.columns)}")
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {data_path}")
        logger.info("请先运行: python scripts/prepare_data.py")
        return
    
    # 2. 配置训练参数
    logger.info("\n[步骤 2] 配置训练参数...")
    
    config = {
        # 数据预处理
        'sequence_length': 20,
        'feature_cols': ['Temperature', 'Humidity', 'WindSpeed'],
        'target_col': 'EDP',
        
        # 预测模型 - 减小规模加快训练
        'cnn_filters': [32, 16],  # 减小过滤器数量
        'lstm_units': 32,  # 减小LSTM单元
        'attention_units': 25,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 10,  # 仅训练10轮用于快速验证
        'batch_size': 32,
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
        'bn_max_iter': 50,  # 减小迭代次数
        'bn_estimator': 'mle'
    }
    
    logger.info(f"训练配置: epochs={config['epochs']}, batch_size={config['batch_size']}")
    
    # 3. 创建训练流水线
    logger.info("\n[步骤 3] 创建训练流水线...")
    pipeline = TrainPipeline(
        config=config,
        output_dir='./outputs/training_run_1'
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
        logger.info("  模型: outputs/training_run_1/models/")
        logger.info("  结果: outputs/training_run_1/results/")
        logger.info("  配置: outputs/training_run_1/config.json")
        
        logger.info("\n✅ 训练成功完成！")
        
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
