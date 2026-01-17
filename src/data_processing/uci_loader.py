"""
UCI Dataset Loader

Functions:
1. Download UCI Household Electric Power Consumption dataset
2. Parse and clean data
3. Feature engineering
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
    """UCI Dataset Loading and Preprocessing"""
    
    DATASET_URL = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    DATASET_ID = 235
    
    def __init__(self, data_dir='data/uci'):
        """
        Initialize the loader
        
        Args:
            data_dir: Root directory for data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _download_with_progress(url, output_path):
        """Download with a progress bar"""
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                downloaded = count * block_size / (1024 * 1024)
                total = total_size / (1024 * 1024)
                print(f"\rDownload Progress: {percent:.1f}% ({downloaded:.1f}/{total:.1f} MB)", 
                      end='', flush=True)
            else:
                downloaded = count * block_size / (1024 * 1024)
                print(f"\rDownloaded: {downloaded:.1f} MB", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(url, output_path, progress_hook)
            print()  # New line
        except Exception as e:
            print()  # New line
            raise
    
    def download(self, method='direct'):
        """
        Download UCI dataset
        
        Args:
            method: 'direct' downloads ZIP directly, 'api' uses ucimlrepo package
            
        Returns:
            Path: Path to the raw data file
        """
        txt_path = self.raw_dir / 'household_power_consumption.txt'
        
        # Skip download if file already exists
        if txt_path.exists():
            logger.info(f"✅ Data file already exists: {txt_path}")
            return txt_path
        
        if method == 'direct':
            return self._download_direct()
        elif method == 'api':
            return self._download_api()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _download_direct(self):
        """Directly download the ZIP file"""
        logger.info("Using direct download method...")
        
        zip_path = self.raw_dir / 'uci_household.zip'
        txt_path = self.raw_dir / 'household_power_consumption.txt'
        
        # Download
        logger.info(f"Starting download: {self.DATASET_URL}")
        logger.info("File size: ~126.8 MB, please wait...")
        
        try:
            self._download_with_progress(self.DATASET_URL, str(zip_path))
            logger.info(f"✅ Download complete: {zip_path}")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            logger.info("Tip: You can manually download and place the file in the designated location")
            raise
        
        # Extract
        logger.info("Extracting file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        logger.info(f"✅ Extraction complete")
        
        # Remove zip
        zip_path.unlink()
        logger.info("Cleaning up temporary files")
        
        return txt_path
    
    def _download_api(self):
        """Download using the ucimlrepo API"""
        logger.info("Downloading via ucimlrepo Python API...")
        
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            logger.error("❌ ucimlrepo not installed")
            logger.info("Installation command: pip install ucimlrepo")
            sys.exit(1)
        
        # Fetch dataset
        logger.info(f"Fetching dataset from UCI repository (ID={self.DATASET_ID})...")
        dataset = fetch_ucirepo(id=self.DATASET_ID)
        
        logger.info("✅ Dataset downloaded successfully")
        
        # Save as CSV
        csv_path = self.raw_dir / 'uci_household_api.csv'
        X = dataset.data.features
        X.to_csv(csv_path, index=False)
        logger.info(f"✅ Data saved: {csv_path}")
        
        return csv_path
    
    def load_raw(self, filepath=None):
        """
        Load raw data
        
        Args:
            filepath: Data file path, defaults to the standard path
            
        Returns:
            DataFrame: Loaded data
        """
        if filepath is None:
            filepath = self.raw_dir / 'household_power_consumption.txt'
        
        logger.info(f"Loading data: {filepath}")
        
        # Read data (semicolon separated, '?' for missing values)
        df = pd.read_csv(
            filepath,
            sep=';',
            na_values=['?'],
            low_memory=False
        )
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H:%M:%S'
        )
        
        # Drop original Date and Time columns
        df = df.drop(['Date', 'Time'], axis=1)
        
        # Reorder columns (datetime first)
        cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
        df = df[cols]
        
        logger.info(f"✅ Data loading complete: {df.shape}")
        
        return df
    
    def preprocess(self, df, resample_freq='15T'):
        """
        Preprocess the data
        
        Args:
            df: Raw data
            resample_freq: Resampling frequency (default 15 minutes)
            
        Returns:
            DataFrame: Preprocessed data
        """
        logger.info("Starting data preprocessing...")
        
        df_clean = df.copy()
        
        # 1. Handle missing values
        logger.info("1️⃣ Handling missing values...")
        initial_missing = df_clean.isnull().sum().sum()
        
        # Forward and backward fill
        df_clean = df_clean.ffill().bfill()
        
        remaining_missing = df_clean.isnull().sum().sum()
        logger.info(f"  Before: {initial_missing:,} missing values")
        logger.info(f"  After: {remaining_missing:,} missing values")
        
        # 2. Resampling
        if 'datetime' in df_clean.columns:
            logger.info(f"2️⃣ Resampling to {resample_freq}...")
            df_clean = df_clean.set_index('datetime')
            df_clean = df_clean.resample(resample_freq).mean()
            logger.info(f"  Samples after resampling: {len(df_clean):,}")
            df_clean = df_clean.reset_index()
        
        # 3. Feature Engineering
        logger.info("3️⃣ Feature Engineering...")
        if 'datetime' in df_clean.columns:
            df_clean['hour'] = df_clean['datetime'].dt.hour
            df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
            df_clean['month'] = df_clean['datetime'].dt.month
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            logger.info(f"  Added temporal features: hour, day_of_week, month, is_weekend")
        
        logger.info(f"✅ Preprocessing complete: {df_clean.shape}")
        
        return df_clean
    
    def save_processed(self, df, filename='uci_household_clean.csv'):
        """
        Save processed data
        
        Args:
            df: Processed data
            filename: Filename
            
        Returns:
            Path: Path to the saved file
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Cleaned data saved: {output_path}")
        logger.info(f"  Final shape: {df.shape}")
        
        return output_path
    
    def get_statistics(self, df):
        """Get statistical information about the data"""
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