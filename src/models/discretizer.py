"""
分位数离散化器
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import logging

logger = logging.getLogger(__name__)


class QuantileDiscretizer:
    """
    分位数离散化器
    
    将连续变量按分位数切分为4个等级: Low, Medium, High, VeryHigh
    """
    
    def __init__(self, n_bins=4, strategy='quantile'):
        """
        参数:
            n_bins: 离散化级数（默认4）
            strategy: 离散化策略（'quantile', 'uniform', 'kmeans'）
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.labels = ['Low', 'Medium', 'High', 'VeryHigh'][:n_bins]
        
        self.discretizers = {}
    
    def fit(self, X):
        """
        拟合离散化器
        
        输入:
            X: [样本数, 特征数] 或 DataFrame
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        logger.info(f"拟合离散化器，特征数: {X.shape[1]}")
        
        # 为每个特征创建离散化器
        for i, name in enumerate(feature_names):
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            
            discretizer.fit(X[:, i].reshape(-1, 1))
            self.discretizers[name] = discretizer
        
        logger.info("离散化器拟合完成")
        
        return self
    
    def transform(self, X):
        """
        转换为离散标签
        
        输入:
            X: [样本数, 特征数] 或 DataFrame
        
        输出:
            离散标签数组 [样本数, 特征数]，值为字符串标签
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X = X.values
        else:
            feature_names = list(self.discretizers.keys())
        
        result = np.empty(X.shape, dtype=object)
        
        for i, name in enumerate(feature_names):
            discretizer = self.discretizers[name]
            
            # 转换为bin索引
            bin_indices = discretizer.transform(X[:, i].reshape(-1, 1)).flatten().astype(int)
            
            # 映射到标签
            result[:, i] = [self.labels[idx] for idx in bin_indices]
        
        return result
    
    def fit_transform(self, X):
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_discrete):
        """
        反向转换（返回每个bin的中心值）
        
        输入:
            X_discrete: 离散标签数组
        
        输出:
            连续值数组
        """
        feature_names = list(self.discretizers.keys())
        result = np.zeros(X_discrete.shape)
        
        for i, name in enumerate(feature_names):
            discretizer = self.discretizers[name]
            
            # 将标签转换回bin索引
            label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            bin_indices = np.array([label_to_idx[label] for label in X_discrete[:, i]])
            
            # 获取bin边界
            bin_edges = discretizer.bin_edges_[0]
            
            # 计算每个bin的中心值
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            result[:, i] = bin_centers[bin_indices]
        
        return result


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    
    X = np.random.randn(1000, 5) * 10 + 50
    
    # 离散化
    discretizer = QuantileDiscretizer(n_bins=4)
    X_discrete = discretizer.fit_transform(X)
    
    print("原始数据样本:")
    print(X[:5])
    
    print("\n离散化后:")
    print(X_discrete[:5])
    
    # 统计每个特征的分布
    for i in range(X_discrete.shape[1]):
        unique, counts = np.unique(X_discrete[:, i], return_counts=True)
        print(f"\n特征 {i}: {dict(zip(unique, counts))}")
