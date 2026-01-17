"""
Quantile Discretizer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import logging

logger = logging.getLogger(__name__)


class QuantileDiscretizer:
    """
    Quantile Discretizer
    
    Partitions continuous variables based on quantiles into 4 levels: Low, Medium, High, VeryHigh.
    """
    
    def __init__(self, n_bins=4, strategy='quantile'):
        """
        Parameters:
            n_bins: Number of discretization bins (default 4)
            strategy: Discretization strategy ('quantile', 'uniform', 'kmeans')
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.labels = ['Low', 'Medium', 'High', 'VeryHigh'][:n_bins]
        
        self.discretizers = {}
    
    def fit(self, X):
        """
        Fit the discretizer
        
        Input:
            X: [n_samples, n_features] or DataFrame
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        logger.info(f"Fitting discretizer, feature count: {X.shape[1]}")
        
        # Create a discretizer for each feature
        for i, name in enumerate(feature_names):
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            
            discretizer.fit(X[:, i].reshape(-1, 1))
            self.discretizers[name] = discretizer
        
        logger.info("Discretizer fit complete")
        
        return self
    
    def transform(self, X):
        """
        Transform to discrete labels
        
        Input:
            X: [n_samples, n_features] or DataFrame
        
        Output:
            Discrete label array [n_samples, n_features] with string labels as values
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X = X.values
        else:
            feature_names = list(self.discretizers.keys())
        
        result = np.empty(X.shape, dtype=object)
        
        for i, name in enumerate(feature_names):
            discretizer = self.discretizers[name]
            
            # Convert to bin indices
            bin_indices = discretizer.transform(X[:, i].reshape(-1, 1)).flatten().astype(int)
            
            # Map to labels
            result[:, i] = [self.labels[idx] for idx in bin_indices]
        
        return result
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_discrete):
        """
        Inverse transformation (returns the center value of each bin)
        
        Input:
            X_discrete: Discrete label array
        
        Output:
            Array of continuous values
        """
        feature_names = list(self.discretizers.keys())
        result = np.zeros(X_discrete.shape)
        
        for i, name in enumerate(feature_names):
            discretizer = self.discretizers[name]
            
            # Map labels back to bin indices
            label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            bin_indices = np.array([label_to_idx[label] for label in X_discrete[:, i]])
            
            # Get bin edges
            bin_edges = discretizer.bin_edges_[0]
            
            # Calculate the center value for each bin
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            result[:, i] = bin_centers[bin_indices]
        
        return result


# Usage Example
if __name__ == "__main__":
    # Mock data
    np.random.seed(42)
    
    X = np.random.randn(1000, 5) * 10 + 50
    
    # Discretization
    discretizer = QuantileDiscretizer(n_bins=4)
    X_discrete = discretizer.fit_transform(X)
    
    print("Original Data Sample:")
    print(X[:5])
    
    print("\nAfter Discretization:")
    print(X_discrete[:5])
    
    # Statistical distribution per feature
    for i in range(X_discrete.shape[1]):
        unique, counts = np.unique(X_discrete[:, i], return_counts=True)
        print(f"\nFeature {i}: {dict(zip(unique, counts))}")