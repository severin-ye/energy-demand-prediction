"""
数据预处理模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class EnergyDataPreprocessor:
    """
    能源数据预处理器
    
    功能:
        1. 数据清洗（缺失值处理、异常值处理）
        2. 时间特征提取
        3. 滑动窗口序列构造
        4. 特征标准化
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 feature_cols: List[str] = None,
                 target_col: str = 'GlobalActivePower'):
        """
        参数:
            sequence_length: 时间窗口长度
            feature_cols: 特征列名列表
            target_col: 目标变量列名
        """
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or []
        self.target_col = target_col
        
        self.scaler = StandardScaler()
        self.fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        步骤:
            1. 处理缺失值（前向+后向填充）
            2. 处理异常值（IQR截断）
        """
        logger.info("开始数据清洗...")
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 如果还有缺失值，用列均值填充
        if df.isnull().any().any():
            df = df.fillna(df.mean())
        
        # 处理异常值（IQR方法）
        for col in self.feature_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"数据清洗完成，样本数: {len(df)}")
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取时间特征
        
        特征:
            - Date: 日期（1-31）
            - Day: 星期（0-6）
            - Month: 月份（1-12）
            - Season: 季节（0-3）
            - Weekend: 是否周末（0/1）
        """
        logger.info("提取时间特征...")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            df['Day'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Season'] = ((df['Month'] % 12 + 3) // 3) % 4
            df['Weekend'] = (df['Day'] >= 5).astype(int)
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口序列
        
        输入:
            data: 特征数据 [样本数, 特征数]
            target: 目标数据 [样本数,]
        
        输出:
            X: [样本数, 序列长度, 特征数]
            y: [样本数,]
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        训练数据的完整预处理流程
        
        输入:
            df: 原始数据框
        
        输出:
            X: 序列特征
            y: 目标值
        """
        logger.info("开始数据预处理...")
        
        # 1. 清洗
        df = self.clean_data(df)
        
        # 2. 时间特征
        df = self.extract_time_features(df)
        
        # 3. 提取特征和目标
        features = df[self.feature_cols].values
        target = df[self.target_col].values
        
        # 4. 标准化
        features = self.scaler.fit_transform(features)
        self.fitted = True
        
        # 5. 创建序列
        X, y = self.create_sequences(features, target)
        
        logger.info(f"预处理完成，序列数: {len(X)}, 特征维度: {X.shape}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        测试数据的预处理流程（使用已拟合的scaler）
        """
        if not self.fitted:
            raise ValueError("请先调用 fit_transform() 拟合预处理器")
        
        # 清洗和特征提取
        df = self.clean_data(df)
        df = self.extract_time_features(df)
        
        # 提取特征
        features = df[self.feature_cols].values
        target = df[self.target_col].values
        
        # 标准化
        features = self.scaler.transform(features)
        
        # 创建序列
        X, y = self.create_sequences(features, target)
        
        return X, y
