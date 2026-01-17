"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class EnergyDataPreprocessor:
    """
    Energy Data Preprocessor
    
    Functions:
        1. Data Cleaning (Handling missing values and outliers)
        2. Time Feature Extraction
        3. Sliding Window Sequence Construction
        4. Feature Standardization
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 feature_cols: List[str] = None,
                 target_col: str = 'GlobalActivePower'):
        """
        Parameters:
            sequence_length: Length of the time window
            feature_cols: List of feature column names
            target_col: Name of the target variable column
        """
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or []
        self.target_col = target_col
        
        self.scaler = StandardScaler()
        self.fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data Cleaning
        
        Steps:
            1. Handle missing values (Forward + Backward fill)
            2. Handle outliers (IQR clipping)
        """
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If missing values remain, fill with column mean
        if df.isnull().any().any():
            df = df.fillna(df.mean())
        
        # Handle outliers (IQR method)
        for col in self.feature_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Data cleaning complete, sample count: {len(df)}")
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Time Features
        
        Features:
            - Date: Day of the month (1-31)
            - Day: Day of the week (0-6)
            - Month: Month (1-12)
            - Season: Season (0-3)
            - Weekend: Is weekend (0/1)
        """
        logger.info("Extracting time features...")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            df['Day'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Season'] = ((df['Month'] % 12 + 3) // 3) % 4
            df['Weekend'] = (df['Day'] >= 5).astype(int)
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Sliding Window Sequences
        
        Input:
            data: Feature data [n_samples, n_features]
            target: Target data [n_samples,]
        
        Output:
            X: [n_samples, sequence_length, n_features]
            y: [n_samples,]
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing workflow for training data
        
        Input:
            df: Raw DataFrame
        
        Output:
            X: Sequence features
            y: Target values
        """
        logger.info("Starting data preprocessing...")
        
        # 1. Clean
        df = self.clean_data(df)
        
        # 2. Time features
        df = self.extract_time_features(df)
        
        # 3. Extract features and target
        features = df[self.feature_cols].values
        target = df[self.target_col].values
        
        # 4. Standardize
        features = self.scaler.fit_transform(features)
        self.fitted = True
        
        # 5. Create sequences
        X, y = self.create_sequences(features, target)
        
        logger.info(f"Preprocessing complete, sequence count: {len(X)}, feature dimensions: {X.shape}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing workflow for test data (using fitted scaler)
        """
        if not self.fitted:
            raise ValueError("Please call fit_transform() first to fit the preprocessor")
        
        # Clean and extract features
        df = self.clean_data(df)
        df = self.extract_time_features(df)
        
        # Extract features
        features = df[self.feature_cols].values
        target = df[self.target_col].values
        
        # Standardize
        features = self.scaler.transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features, target)
        
        return X, y