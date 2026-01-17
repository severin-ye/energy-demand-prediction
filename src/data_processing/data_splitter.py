"""
Dataset Splitter

Functions:
1. Split training and test sets by ratio
2. Support sequential splitting for time-series data
3. Support random splitting
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSplitter:
    """Dataset Splitter"""
    
    def __init__(self, output_dir='data/uci/splits'):
        """
        Initialize the splitter
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_sequential(self, df, test_ratio=0.05):
        """
        Sequential Splitting (for time-series data)
        
        Splits data chronologically, using the earlier portion as the training set 
        and the later portion as the test set.
        
        Args:
            df: Dataset
            test_ratio: Ratio of the test set
            
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info(f"Splitting dataset sequentially (Test ratio: {test_ratio*100:.1f}%)")
        
        n_samples = len(df)
        n_test = int(n_samples * test_ratio)
        n_train = n_samples - n_test
        
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()
        
        logger.info(f"  Training set: {len(train_df):,} samples ({len(train_df)/n_samples*100:.1f}%)")
        logger.info(f"  Test set: {len(test_df):,} samples ({len(test_df)/n_samples*100:.1f}%)")
        
        return train_df, test_df
    
    def split_random(self, df, test_ratio=0.05, random_state=42):
        """
        Random Splitting
        
        Args:
            df: Dataset
            test_ratio: Ratio of the test set
            random_state: Random seed
            
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info(f"Splitting dataset randomly (Test ratio: {test_ratio*100:.1f}%)")
        
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_state
        )
        
        logger.info(f"  Training set: {len(train_df):,} samples")
        logger.info(f"  Test set: {len(test_df):,} samples")
        
        return train_df, test_df
    
    def save_splits(self, train_df, test_df, 
                    train_filename='train.csv', 
                    test_filename='test.csv'):
        """
        Save the split datasets
        
        Args:
            train_df: Training set
            test_df: Test set
            train_filename: Filename for training set
            test_filename: Filename for test set
            
        Returns:
            dict: Paths to the saved files
        """
        train_path = self.output_dir / train_filename
        test_path = self.output_dir / test_filename
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✅ Training set saved: {train_path}")
        logger.info(f"✅ Test set saved: {test_path}")
        
        return {
            'train': train_path,
            'test': test_path
        }
    
    def get_split_info(self, train_df, test_df):
        """
        Get statistics about the split
        
        Args:
            train_df: Training set
            test_df: Test set
            
        Returns:
            dict: Statistical information
        """
        total = len(train_df) + len(test_df)
        
        info = {
            'total_samples': total,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_ratio': len(train_df) / total,
            'test_ratio': len(test_df) / total,
        }
        
        # If datetime column exists, add time-range information
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