"""
DLP聚类模块（CAM和Attention权重聚类）
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DLPClusterer:
    """
    深度学习参数(DLP)聚类器 - CAM聚类
    
    对CAM值进行累积预处理后聚类
    """
    
    def __init__(self, n_clusters=3, dlp_type='CAM'):
        """
        参数:
            n_clusters: 聚类数量
            dlp_type: DLP类型标识
        """
        self.n_clusters = n_clusters
        self.dlp_type = dlp_type
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.cluster_names_ = [f'Type{i+1}' for i in range(n_clusters)]
    
    def cumulative_preprocess(self, cam_values):
        """
        累积预处理
        
        输入:
            cam_values: [样本数, 时间步]
        
        输出:
            累积CAM: [样本数, 时间步]
        """
        return np.cumsum(cam_values, axis=1)
    
    def fit(self, cam_values):
        """
        拟合聚类器
        
        输入:
            cam_values: CAM值 [样本数, 时间步]
        """
        logger.info(f"{self.dlp_type} 聚类拟合...")
        
        # 累积预处理
        cam_cumulative = self.cumulative_preprocess(cam_values)
        
        # 标准化
        cam_scaled = self.scaler.fit_transform(cam_cumulative)
        
        # K-means聚类
        self.kmeans.fit(cam_scaled)
        
        logger.info(f"{self.dlp_type} 聚类完成，聚类数: {self.n_clusters}")
        
        return self
    
    def predict(self, cam_values):
        """
        预测聚类标签
        
        输入:
            cam_values: CAM值 [样本数, 时间步]
        
        输出:
            聚类标签 [样本数,]
        """
        cam_cumulative = self.cumulative_preprocess(cam_values)
        cam_scaled = self.scaler.transform(cam_cumulative)
        
        labels = self.kmeans.predict(cam_scaled)
        
        return labels
    
    def fit_predict(self, cam_values):
        """拟合并预测"""
        self.fit(cam_values)
        return self.predict(cam_values)


class AttentionClusterer:
    """
    Attention权重聚类器
    
    识别Early/Late/Other时序关注模式
    """
    
    def __init__(self, n_clusters=3):
        """
        参数:
            n_clusters: 聚类数量
        """
        self.n_clusters = n_clusters
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.cluster_names_ = None
    
    def fit(self, attention_weights):
        """
        拟合聚类器
        
        输入:
            attention_weights: [样本数, 时间步]
        """
        logger.info("Attention权重聚类拟合...")
        
        # 标准化
        att_scaled = self.scaler.fit_transform(attention_weights)
        
        # K-means聚类
        self.kmeans.fit(att_scaled)
        
        # 分析聚类中心特征，命名聚类
        self._analyze_clusters(attention_weights)
        
        logger.info(f"Attention聚类完成，聚类: {self.cluster_names_}")
        
        return self
    
    def _analyze_clusters(self, attention_weights):
        """
        分析聚类中心特征，分配语义名称
        
        逻辑:
            - Early: 高权重集中在前1/3
            - Late: 高权重集中在后1/3
            - Other: 均匀分布或中间集中
        """
        centers = self.kmeans.cluster_centers_
        n_timesteps = centers.shape[1]
        
        early_third = n_timesteps // 3
        late_third = 2 * n_timesteps // 3
        
        cluster_types = []
        
        for center in centers:
            early_weight = np.sum(center[:early_third])
            late_weight = np.sum(center[late_third:])
            middle_weight = np.sum(center[early_third:late_third])
            
            total = early_weight + late_weight + middle_weight
            
            early_ratio = early_weight / total
            late_ratio = late_weight / total
            
            if early_ratio > 0.5:
                cluster_types.append('Early')
            elif late_ratio > 0.5:
                cluster_types.append('Late')
            else:
                cluster_types.append('Other')
        
        self.cluster_names_ = cluster_types
    
    def predict(self, attention_weights):
        """
        预测聚类标签
        
        输入:
            attention_weights: [样本数, 时间步]
        
        输出:
            聚类标签 [样本数,]
        """
        att_scaled = self.scaler.transform(attention_weights)
        labels = self.kmeans.predict(att_scaled)
        
        return labels
    
    def fit_predict(self, attention_weights):
        """拟合并预测"""
        self.fit(attention_weights)
        return self.predict(attention_weights)


# 使用示例
if __name__ == "__main__":
    np.random.seed(42)
    
    # 模拟CAM数据
    cam_data = np.random.rand(100, 20)
    
    # CAM聚类
    cam_clusterer = DLPClusterer(n_clusters=3)
    cam_labels = cam_clusterer.fit_predict(cam_data)
    
    print("CAM聚类结果:")
    unique, counts = np.unique(cam_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {cam_clusterer.cluster_names_[label]}: {count}")
    
    # 模拟Attention数据
    att_data = np.random.rand(100, 60)
    
    # Attention聚类
    att_clusterer = AttentionClusterer(n_clusters=3)
    att_labels = att_clusterer.fit_predict(att_data)
    
    print("\nAttention聚类结果:")
    unique, counts = np.unique(att_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {att_clusterer.cluster_names_[label]}: {count}")
