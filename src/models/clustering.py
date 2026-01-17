"""
DLP Clustering Module (CAM and Attention Weight Clustering)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DLPClusterer:
    """
    Deep Learning Parameters (DLP) Clusterer - CAM Clustering
    
    Performs clustering on CAM values after cumulative preprocessing.
    """
    
    def __init__(self, n_clusters=3, dlp_type='CAM'):
        """
        Parameters:
            n_clusters: Number of clusters
            dlp_type: DLP type identifier
        """
        self.n_clusters = n_clusters
        self.dlp_type = dlp_type
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.cluster_names_ = [f'Type{i+1}' for i in range(n_clusters)]
    
    def cumulative_preprocess(self, cam_values):
        """
        Cumulative Preprocessing
        
        Input:
            cam_values: [n_samples, n_timesteps]
        
        Output:
            Cumulative CAM: [n_samples, n_timesteps]
        """
        return np.cumsum(cam_values, axis=1)
    
    def fit(self, cam_values):
        """
        Fit Clusterer
        
        Input:
            cam_values: CAM values [n_samples, n_timesteps]
        """
        logger.info(f"{self.dlp_type} Clustering Fit...")
        
        # Cumulative preprocessing
        cam_cumulative = self.cumulative_preprocess(cam_values)
        
        # Standardization
        cam_scaled = self.scaler.fit_transform(cam_cumulative)
        
        # K-means clustering
        self.kmeans.fit(cam_scaled)
        
        logger.info(f"{self.dlp_type} clustering complete, n_clusters: {self.n_clusters}")
        
        return self
    
    def predict(self, cam_values):
        """
        Predict cluster labels
        
        Input:
            cam_values: CAM values [n_samples, n_timesteps]
        
        Output:
            Cluster labels [n_samples,]
        """
        cam_cumulative = self.cumulative_preprocess(cam_values)
        cam_scaled = self.scaler.transform(cam_cumulative)
        
        labels = self.kmeans.predict(cam_scaled)
        
        return labels
    
    def fit_predict(self, cam_values):
        """Fit and predict"""
        self.fit(cam_values)
        return self.predict(cam_values)


class AttentionClusterer:
    """
    Attention Weight Clusterer
    
    Identifies Early/Late/Other temporal focus patterns.
    """
    
    def __init__(self, n_clusters=3):
        """
        Parameters:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.cluster_names_ = None
    
    def fit(self, attention_weights):
        """
        Fit Clusterer
        
        Input:
            attention_weights: [n_samples, n_timesteps]
        """
        logger.info("Attention weight clustering fit...")
        
        # Standardization
        att_scaled = self.scaler.fit_transform(attention_weights)
        
        # K-means clustering
        self.kmeans.fit(att_scaled)
        
        # Analyze cluster center characteristics and name them
        self._analyze_clusters(attention_weights)
        
        logger.info(f"Attention clustering complete, clusters: {self.cluster_names_}")
        
        return self
    
    def _analyze_clusters(self, attention_weights):
        """
        Analyze cluster center features to assign semantic names
        
        Logic:
            - Early: High weights concentrated in the first 1/3
            - Late: High weights concentrated in the last 1/3
            - Other: Uniform distribution or center concentration
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
        Predict cluster labels
        
        Input:
            attention_weights: [n_samples, n_timesteps]
        
        Output:
            Cluster labels [n_samples,]
        """
        att_scaled = self.scaler.transform(attention_weights)
        labels = self.kmeans.predict(att_scaled)
        
        return labels
    
    def fit_predict(self, attention_weights):
        """Fit and predict"""
        self.fit(attention_weights)
        return self.predict(attention_weights)


# Usage Example
if __name__ == "__main__":
    np.random.seed(42)
    
    # Mock CAM data
    cam_data = np.random.rand(100, 20)
    
    # CAM Clustering
    cam_clusterer = DLPClusterer(n_clusters=3)
    cam_labels = cam_clusterer.fit_predict(cam_data)
    
    print("CAM Clustering Results:")
    unique, counts = np.unique(cam_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {cam_clusterer.cluster_names_[label]}: {count}")
    
    # Mock Attention data
    att_data = np.random.rand(100, 60)
    
    # Attention Clustering
    att_clusterer = AttentionClusterer(n_clusters=3)
    att_labels = att_clusterer.fit_predict(att_data)
    
    print("\nAttention Clustering Results:")
    unique, counts = np.unique(att_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {att_clusterer.cluster_names_[label]}: {count}")