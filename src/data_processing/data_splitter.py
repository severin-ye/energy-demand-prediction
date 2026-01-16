"""
数据集划分器

功能：
1. 按比例划分训练集和测试集
2. 支持时间序列数据的顺序划分
3. 支持随机划分
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSplitter:
    """数据集划分器"""
    
    def __init__(self, output_dir='data/uci/splits'):
        """
        初始化划分器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_sequential(self, df, test_ratio=0.05):
        """
        顺序划分（用于时间序列数据）
        
        将数据按时间顺序划分，前面的作为训练集，后面的作为测试集
        
        Args:
            df: 数据集
            test_ratio: 测试集比例
            
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info(f"顺序划分数据集 (测试集比例: {test_ratio*100:.1f}%)")
        
        n_samples = len(df)
        n_test = int(n_samples * test_ratio)
        n_train = n_samples - n_test
        
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()
        
        logger.info(f"  训练集: {len(train_df):,} 样本 ({len(train_df)/n_samples*100:.1f}%)")
        logger.info(f"  测试集: {len(test_df):,} 样本 ({len(test_df)/n_samples*100:.1f}%)")
        
        return train_df, test_df
    
    def split_random(self, df, test_ratio=0.05, random_state=42):
        """
        随机划分
        
        Args:
            df: 数据集
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info(f"随机划分数据集 (测试集比例: {test_ratio*100:.1f}%)")
        
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_state
        )
        
        logger.info(f"  训练集: {len(train_df):,} 样本")
        logger.info(f"  测试集: {len(test_df):,} 样本")
        
        return train_df, test_df
    
    def save_splits(self, train_df, test_df, 
                    train_filename='train.csv', 
                    test_filename='test.csv'):
        """
        保存划分后的数据集
        
        Args:
            train_df: 训练集
            test_df: 测试集
            train_filename: 训练集文件名
            test_filename: 测试集文件名
            
        Returns:
            dict: 保存的文件路径
        """
        train_path = self.output_dir / train_filename
        test_path = self.output_dir / test_filename
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✅ 保存训练集: {train_path}")
        logger.info(f"✅ 保存测试集: {test_path}")
        
        return {
            'train': train_path,
            'test': test_path
        }
    
    def get_split_info(self, train_df, test_df):
        """
        获取划分信息统计
        
        Args:
            train_df: 训练集
            test_df: 测试集
            
        Returns:
            dict: 统计信息
        """
        total = len(train_df) + len(test_df)
        
        info = {
            'total_samples': total,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_ratio': len(train_df) / total,
            'test_ratio': len(test_df) / total,
        }
        
        # 如果有datetime列，添加时间信息
        if 'datetime' in train_df.columns:
            info['train_time_range'] = {
                'start': train_df['datetime'].min(),
                'end': train_df['datetime'].max()
            }
            info['test_time_range'] = {
                'start': test_df['datetime'].min(),
                'end': test_df['datetime'].max()
            }
        
        return info
