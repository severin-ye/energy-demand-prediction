"""
划分UCI数据集为训练集和测试集

用法:
    python scripts/split_uci_dataset.py --test-ratio 0.05
"""
import sys
import os
import argparse
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.uci_loader import UCIDataLoader
from src.data_processing.data_splitter import DataSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='划分UCI数据集为训练集和测试集'
    )
    parser.add_argument(
        '--input',
        default='data/uci/processed/uci_household_clean.csv',
        help='输入的预处理数据文件'
    )
    parser.add_argument(
        '--output-dir',
        default='data/uci/splits',
        help='输出目录'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.05,
        help='测试集比例（默认0.05即5%%）'
    )
    parser.add_argument(
        '--split-method',
        choices=['sequential', 'random'],
        default='sequential',
        help='划分方法：sequential（顺序划分）或random（随机划分）'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='随机种子（仅用于random方法）'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("="*70)
        logger.info("UCI数据集划分")
        logger.info("="*70)
        
        # 1. 加载数据
        logger.info(f"\n📂 加载数据: {args.input}")
        import pandas as pd
        df = pd.read_csv(args.input)
        logger.info(f"  样本数: {len(df):,}")
        logger.info(f"  特征数: {len(df.columns)}")
        
        # 显示时间范围
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        
        # 2. 划分数据集
        logger.info(f"\n✂️  划分数据集")
        logger.info(f"  方法: {args.split_method}")
        logger.info(f"  测试集比例: {args.test_ratio*100:.1f}%")
        
        splitter = DataSplitter(output_dir=args.output_dir)
        
        if args.split_method == 'sequential':
            train_df, test_df = splitter.split_sequential(df, test_ratio=args.test_ratio)
        else:
            train_df, test_df = splitter.split_random(
                df, 
                test_ratio=args.test_ratio,
                random_state=args.random_state
            )
        
        # 3. 保存划分
        logger.info(f"\n💾 保存划分后的数据集")
        paths = splitter.save_splits(train_df, test_df)
        
        # 4. 显示统计信息
        logger.info(f"\n📊 划分统计")
        info = splitter.get_split_info(train_df, test_df)
        
        logger.info(f"  总样本数: {info['total_samples']:,}")
        logger.info(f"  训练集: {info['train_samples']:,} ({info['train_ratio']*100:.2f}%)")
        logger.info(f"  测试集: {info['test_samples']:,} ({info['test_ratio']*100:.2f}%)")
        
        if 'train_time_range' in info:
            logger.info(f"\n  训练集时间范围:")
            logger.info(f"    {info['train_time_range']['start']} ~ {info['train_time_range']['end']}")
            logger.info(f"  测试集时间范围:")
            logger.info(f"    {info['test_time_range']['start']} ~ {info['test_time_range']['end']}")
        
        # 5. 数据质量检查
        logger.info(f"\n🔍 数据质量检查")
        
        train_missing = train_df.isnull().sum().sum()
        test_missing = test_df.isnull().sum().sum()
        
        logger.info(f"  训练集缺失值: {train_missing}")
        logger.info(f"  测试集缺失值: {test_missing}")
        
        # 检查目标变量
        if 'Global_active_power' in train_df.columns:
            logger.info(f"\n  目标变量 (Global_active_power):")
            logger.info(f"    训练集 - 均值: {train_df['Global_active_power'].mean():.3f}, "
                       f"标准差: {train_df['Global_active_power'].std():.3f}")
            logger.info(f"    测试集 - 均值: {test_df['Global_active_power'].mean():.3f}, "
                       f"标准差: {test_df['Global_active_power'].std():.3f}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ 划分完成！")
        logger.info("="*70)
        logger.info(f"训练集: {paths['train']}")
        logger.info(f"测试集: {paths['test']}")
        logger.info(f"\n可以使用训练集进行模型训练:")
        logger.info(f"  python scripts/run_training.py --data {paths['train']}")
        
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
