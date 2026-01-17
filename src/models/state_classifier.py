"""
Sn Scale State Classifier
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class SnStateClassifier:
    """
    State Classifier based on the Sn robust scale estimator.
    
    The Sn estimator is robust to outliers, making it suitable for energy data.
    Uses K-means to partition data into three states: Peak/Normal/Lower.
    """
    
    def __init__(self, n_states=3, state_names=None):
        """
        Parameters:
            n_states: Number of states (default 3: Lower/Normal/Peak)
            state_names: List of names for the states
        """
        self.n_states = n_states
        self.state_names = state_names or ['Lower', 'Normal', 'Peak']
        
        self.sn_scale_ = None
        self.median_ = None
        self.kmeans = KMeans(n_clusters=n_states, random_state=42)
        self.cluster_to_state_ = None
    
    def compute_sn_scale(self, data):
        """
        Calculates the Sn scale estimator.
        
        Sn = c * median_i { median_j |x_i - x_j| }
        
        where c is a correction factor to make it unbiased under a normal distribution.
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Calculate the median of all pairwise differences
        diffs = []
        for i in range(min(n, 1000)):  # Cap calculation for efficiency
            diffs.append(np.median(np.abs(data[i] - data)))
        
        sn = np.median(diffs)
        
        # Correction factor (for normal distribution)
        c = 1.1926
        
        return c * sn
    
    def fit(self, data):
        """
        Fits the classifier.
        
        Steps:
            1. Calculate the Sn scale and median.
            2. Perform K-means clustering.
            3. Map cluster labels to state names.
        """
        data = np.asarray(data).flatten()
        
        logger.info("Computing Sn scale estimator...")
        self.sn_scale_ = self.compute_sn_scale(data)
        self.median_ = np.median(data)
        
        logger.info(f"Median: {self.median_:.4f}, Sn scale: {self.sn_scale_:.4f}")
        
        # Cluster after normalization
        data_normalized = (data - self.median_) / self.sn_scale_
        
        logger.info("Performing K-means clustering...")
        self.kmeans.fit(data_normalized.reshape(-1, 1))
        
        # Map cluster centers to states
        centers = self.kmeans.cluster_centers_.flatten()
        center_order = np.argsort(centers)  # Sort from lowest to highest
        
        self.cluster_to_state_ = {}
        for i, cluster_id in enumerate(center_order):
            self.cluster_to_state_[cluster_id] = self.state_names[i]
        
        logger.info(f"Cluster centers: {centers}")
        logger.info(f"State mapping: {self.cluster_to_state_}")
        
        return self
    
    def predict(self, data):
        """
        Predicts the states.
        
        Input:
            data: Energy demand values
        
        Output:
            Array of state labels
        """
        data = np.asarray(data).flatten()
        
        # Normalization
        data_normalized = (data - self.median_) / self.sn_scale_
        
        # Clustering prediction
        cluster_labels = self.kmeans.predict(data_normalized.reshape(-1, 1))
        
        # Map to states
        state_labels = np.array([
            self.cluster_to_state_[label] for label in cluster_labels
        ])
        
        return state_labels
    
    def fit_predict(self, data):
        """Fit and predict"""
        self.fit(data)
        return self.predict(data)


# Usage Example
if __name__ == "__main__":
    # Mock data (including outliers)
    np.random.seed(42)
    
    data = np.concatenate([
        np.random.normal(1.0, 0.3, 300),   # Lower
        np.random.normal(3.0, 0.5, 500),   # Normal
        np.random.normal(6.0, 0.8, 200),   # Peak
        np.array([15.0, 20.0, -2.0])       # Outliers
    ])
    
    # Classification
    classifier = SnStateClassifier()
    states = classifier.fit_predict(data)
    
    # Statistics
    unique, counts = np.unique(states, return_counts=True)
    for state, count in zip(unique, counts):
        print(f"{state}: {count} samples")