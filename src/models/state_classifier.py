"""
Sn尺度状态分类器
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class SnStateClassifier:
    """
    基于Sn鲁棒尺度估计器的状态分类器
    
    Sn尺度估计器对异常值鲁棒，适合能源数据
    使用K-means将数据分为Peak/Normal/Lower三个状态
    """
    
    def __init__(self, n_states=3, state_names=None):
        """
        参数:
            n_states: 状态数量（默认3: Lower/Normal/Peak）
            state_names: 状态名称列表
        """
        self.n_states = n_states
        self.state_names = state_names or ['Lower', 'Normal', 'Peak']
        
        self.sn_scale_ = None
        self.median_ = None
        self.kmeans = KMeans(n_clusters=n_states, random_state=42)
        self.cluster_to_state_ = None
    
    def compute_sn_scale(self, data):
        """
        计算Sn尺度估计器
        
        Sn = c * median_i { median_j |x_i - x_j| }
        
        其中 c 是修正因子，使其在正态分布下无偏
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # 计算所有成对差异的中位数
        diffs = []
        for i in range(min(n, 1000)):  # 限制计算量
            diffs.append(np.median(np.abs(data[i] - data)))
        
        sn = np.median(diffs)
        
        # 修正因子（正态分布下）
        c = 1.1926
        
        return c * sn
    
    def fit(self, data):
        """
        拟合分类器
        
        步骤:
            1. 计算Sn尺度和中位数
            2. 使用K-means聚类
            3. 映射聚类标签到状态名称
        """
        data = np.asarray(data).flatten()
        
        logger.info("计算Sn尺度估计器...")
        self.sn_scale_ = self.compute_sn_scale(data)
        self.median_ = np.median(data)
        
        logger.info(f"中位数: {self.median_:.4f}, Sn尺度: {self.sn_scale_:.4f}")
        
        # 标准化后聚类
        data_normalized = (data - self.median_) / self.sn_scale_
        
        logger.info("K-means聚类...")
        self.kmeans.fit(data_normalized.reshape(-1, 1))
        
        # 映射聚类中心到状态
        centers = self.kmeans.cluster_centers_.flatten()
        center_order = np.argsort(centers)  # 从小到大排序
        
        self.cluster_to_state_ = {}
        for i, cluster_id in enumerate(center_order):
            self.cluster_to_state_[cluster_id] = self.state_names[i]
        
        logger.info(f"聚类中心: {centers}")
        logger.info(f"状态映射: {self.cluster_to_state_}")
        
        return self
    
    def predict(self, data):
        """
        预测状态
        
        输入:
            data: 能源需求值
        
        输出:
            状态标签数组
        """
        data = np.asarray(data).flatten()
        
        # 标准化
        data_normalized = (data - self.median_) / self.sn_scale_
        
        # 聚类预测
        cluster_labels = self.kmeans.predict(data_normalized.reshape(-1, 1))
        
        # 映射到状态
        state_labels = np.array([
            self.cluster_to_state_[label] for label in cluster_labels
        ])
        
        return state_labels
    
    def fit_predict(self, data):
        """拟合并预测"""
        self.fit(data)
        return self.predict(data)


# 使用示例
if __name__ == "__main__":
    # 模拟数据（包含异常值）
    np.random.seed(42)
    
    data = np.concatenate([
        np.random.normal(1.0, 0.3, 300),   # Lower
        np.random.normal(3.0, 0.5, 500),   # Normal
        np.random.normal(6.0, 0.8, 200),   # Peak
        np.array([15.0, 20.0, -2.0])       # 异常值
    ])
    
    # 分类
    classifier = SnStateClassifier()
    states = classifier.fit_predict(data)
    
    # 统计
    unique, counts = np.unique(states, return_counts=True)
    for state, count in zip(unique, counts):
        print(f"{state}: {count} samples")
