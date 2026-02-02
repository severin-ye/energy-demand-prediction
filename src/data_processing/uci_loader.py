"""
UCI数据集加载器

功能：
1. 下载UCI Household Electric Power Consumption数据集
2. 解析和清洗数据
3. 特征工程
"""
import os
import sys
import logging
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class UCIDataLoader:
    """UCI数据集加载和预处理"""
    
    DATASET_URL = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    DATASET_ID = 235
    
    def __init__(self, data_dir='data/uci'):
        """
        初始化加载器
        
        Args:
            data_dir: 数据根目录
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # 创建目录
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _download_with_progress(url, output_path):
        """带进度条的下载"""
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                downloaded = count * block_size / (1024 * 1024)
                total = total_size / (1024 * 1024)
                print(f"\r下载进度: {percent:.1f}% ({downloaded:.1f}/{total:.1f} MB)", 
                      end='', flush=True)
            else:
                downloaded = count * block_size / (1024 * 1024)
                print(f"\r已下载: {downloaded:.1f} MB", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(url, output_path, progress_hook)
            print()  # 换行
        except Exception as e:
            print()  # 换行
            raise
    
    def download(self, method='direct'):
        """
        下载UCI数据集
        
        Args:
            method: 'direct' 直接下载ZIP, 'api' 使用ucimlrepo包
            
        Returns:
            Path: 原始数据文件路径
        """
        txt_path = self.raw_dir / 'household_power_consumption.txt'
        
        # 如果已存在，跳过下载
        if txt_path.exists():
            logger.info(f"✅ 数据文件已存在: {txt_path}")
            return txt_path
        
        if method == 'direct':
            return self._download_direct()
        elif method == 'api':
            return self._download_api()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _download_direct(self):
        """直接下载ZIP文件"""
        logger.info("使用直接下载方式...")
        
        zip_path = self.raw_dir / 'uci_household.zip'
        txt_path = self.raw_dir / 'household_power_consumption.txt'
        
        # 下载
        logger.info(f"开始下载: {self.DATASET_URL}")
        logger.info("文件大小: ~126.8 MB，请耐心等待...")
        
        try:
            self._download_with_progress(self.DATASET_URL, str(zip_path))
            logger.info(f"✅ 下载完成: {zip_path}")
        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            logger.info("提示: 可以手动下载并放到指定位置")
            raise
        
        # 解压
        logger.info("解压文件...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        logger.info(f"✅ 解压完成")
        
        # 删除zip
        zip_path.unlink()
        logger.info("清理临时文件")
        
        return txt_path
    
    def _download_api(self):
        """使用ucimlrepo API下载"""
        logger.info("使用ucimlrepo Python API下载...")
        
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            logger.error("❌ ucimlrepo未安装")
            logger.info("安装命令: pip install ucimlrepo")
            sys.exit(1)
        
        # 获取数据集
        logger.info(f"正在从UCI仓库获取数据集 ID={self.DATASET_ID}...")
        dataset = fetch_ucirepo(id=self.DATASET_ID)
        
        logger.info("✅ 数据集下载成功")
        
        # 保存为CSV
        csv_path = self.raw_dir / 'uci_household_api.csv'
        X = dataset.data.features
        X.to_csv(csv_path, index=False)
        logger.info(f"✅ 保存数据: {csv_path}")
        
        return csv_path
    
    def load_raw(self, filepath=None):
        """
        加载原始数据
        
        Args:
            filepath: 数据文件路径，默认使用标准路径
            
        Returns:
            DataFrame: 加载的数据
        """
        if filepath is None:
            filepath = self.raw_dir / 'household_power_consumption.txt'
        
        logger.info(f"加载数据: {filepath}")
        
        # 读取数据（分号分隔，问号表示缺失值）
        df = pd.read_csv(
            filepath,
            sep=';',
            na_values=['?'],
            low_memory=False
        )
        
        # 解析日期时间
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H:%M:%S'
        )
        
        # 删除原始Date和Time列
        df = df.drop(['Date', 'Time'], axis=1)
        
        # 重新排序列（datetime放第一列）
        cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
        df = df[cols]
        
        logger.info(f"✅ 数据加载完成: {df.shape}")
        
        return df
    
    def preprocess(self, df, resample_freq='15T'):
        """
        预处理数据
        
        Args:
            df: 原始数据
            resample_freq: 重采样频率（默认15分钟）
            
        Returns:
            DataFrame: 预处理后的数据
        """
        logger.info("开始数据预处理...")
        
        df_clean = df.copy()
        
        # 1. 处理缺失值
        logger.info("1️⃣ 处理缺失值...")
        initial_missing = df_clean.isnull().sum().sum()
        
        # 前向填充和后向填充
        df_clean = df_clean.ffill().bfill()
        
        remaining_missing = df_clean.isnull().sum().sum()
        logger.info(f"  处理前: {initial_missing:,} 缺失值")
        logger.info(f"  处理后: {remaining_missing:,} 缺失值")
        
        # 2. 重采样
        if 'datetime' in df_clean.columns:
            logger.info(f"2️⃣ 重采样到{resample_freq}...")
            df_clean = df_clean.set_index('datetime')
            df_clean = df_clean.resample(resample_freq).mean()
            logger.info(f"  重采样后样本数: {len(df_clean):,}")
            df_clean = df_clean.reset_index()
        
        # 3. 特征工程
        logger.info("3️⃣ 特征工程...")
        if 'datetime' in df_clean.columns:
            df_clean['hour'] = df_clean['datetime'].dt.hour
            df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
            df_clean['month'] = df_clean['datetime'].dt.month
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            logger.info(f"  添加时间特征: hour, day_of_week, month, is_weekend")
        
        logger.info(f"✅ 预处理完成: {df_clean.shape}")
        
        return df_clean
    
    def save_processed(self, df, filename='uci_household_clean.csv'):
        """
        保存处理后的数据
        
        Args:
            df: 处理后的数据
            filename: 文件名
            
        Returns:
            Path: 保存的文件路径
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"✅ 保存清洗数据: {output_path}")
        logger.info(f"  最终形状: {df.shape}")
        
        return output_path
    
    def get_statistics(self, df):
        """获取数据统计信息"""
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': df.describe().to_dict()
        }
        
        if 'datetime' in df.columns:
            stats['time_range'] = {
                'start': df['datetime'].min(),
                'end': df['datetime'].max(),
                'duration_days': (df['datetime'].max() - df['datetime'].min()).days
            }
        
        return stats


def load_uci_dataset(use_splits=True):
    """
    便捷函数：加载UCI数据集
    
    Args:
        use_splits: 是否使用训练/测试分割（默认True）
                   如果True，返回训练集和测试集
                   如果False，返回完整处理后的数据集
    
    Returns:
        如果use_splits=True: (train_df, test_df)
        如果use_splits=False: df
    """
    loader = UCIDataLoader()
    
    if use_splits:
        # 检查是否存在分割后的数据
        train_path = loader.data_dir / 'splits' / 'train.csv'
        test_path = loader.data_dir / 'splits' / 'test.csv'
        
        if train_path.exists() and test_path.exists():
            logger.info(f"加载已分割的数据: {train_path}, {test_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # 转换datetime列
            if 'datetime' in train_df.columns:
                train_df['datetime'] = pd.to_datetime(train_df['datetime'])
                test_df['datetime'] = pd.to_datetime(test_df['datetime'])
            
            return train_df, test_df
        else:
            logger.warning("未找到分割数据，将加载完整数据集")
            logger.warning("请先运行 scripts/split_uci_dataset.py 进行数据分割")
            
    # 加载完整数据集
    processed_path = loader.processed_dir / 'uci_household_clean.csv'
    
    if processed_path.exists():
        logger.info(f"加载处理后的数据: {processed_path}")
        df = pd.read_csv(processed_path)
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        if use_splits:
            # 简单分割：80% 训练，20% 测试
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            return train_df, test_df
        else:
            return df
    else:
        raise FileNotFoundError(
            f"未找到处理后的数据集: {processed_path}\n"
            "请先运行以下命令下载和处理数据:\n"
            "  python scripts/download_uci_data.py\n"
            "  python scripts/split_uci_dataset.py"
        )
