"""
Download UCI Household Dataset using the ucimlrepo Python API

Dataset Information:
- Name: Individual Household Electric Power Consumption
- ID: 235
- Number of Instances: 2,075,259
- Time Span: 2006/12/16 - 2010/11/26
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_with_progress(url, output_path):
    """Download with a progress bar"""
    import urllib.request
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 / total_size)
            downloaded = count * block_size / (1024 * 1024)
            total = total_size / (1024 * 1024)
            print(f"\rDownload Progress: {percent:.1f}% ({downloaded:.1f}/{total:.1f} MB)", end='', flush=True)
        else:
            downloaded = count * block_size / (1024 * 1024)
            print(f"\rDownloaded: {downloaded:.1f} MB", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # New line
    except Exception as e:
        print()  # New line
        raise


def download_uci_dataset(output_dir='data/raw', method='direct'):
    """
    Download the UCI Dataset
    
    Args:
        output_dir: Output directory
        method: 'api' uses ucimlrepo, 'direct' downloads ZIP directly
    
    Returns:
        DataFrame: Loaded dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if method == 'api':
        logger.info("Downloading via ucimlrepo Python API...")
        
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            logger.error("❌ ucimlrepo is not installed")
            logger.info("Installation command: pip install ucimlrepo")
            sys.exit(1)
        
        # Fetch dataset
        logger.info("Fetching dataset from UCI repository (ID=235)...")
        dataset = fetch_ucirepo(id=235)
        
        # Extract data
        logger.info("✅ Dataset downloaded successfully")
        
        # Display metadata
        logger.info("\n" + "="*70)
        logger.info("Dataset Metadata")
        logger.info("="*70)
        if hasattr(dataset, 'metadata'):
            for key, value in dataset.metadata.items():
                if key in ['name', 'num_instances', 'num_features', 'area', 'task']:
                    logger.info(f"{key}: {value}")
        
        # Combine features and targets (if applicable)
        X = dataset.data.features
        
        logger.info(f"\nFeature data shape: {X.shape}")
        logger.info(f"Column names: {X.columns.tolist()}")
        
        # Save raw data
        output_path = os.path.join(output_dir, 'uci_household_raw.csv')
        X.to_csv(output_path, index=False)
        logger.info(f"✅ Raw data saved: {output_path}")
        
        return X
        
    elif method == 'direct':
        logger.info("Downloading directly...")
        import zipfile
        
        # Download URL
        url = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
        zip_path = os.path.join(output_dir, 'uci_household.zip')
        txt_path = os.path.join(output_dir, 'household_power_consumption.txt')
        
        # Check if already exists
        if os.path.exists(txt_path):
            logger.info(f"✅ Data file already exists: {txt_path}")
            return load_txt_dataset(txt_path)
        
        # Download
        logger.info(f"Starting download: {url}")
        logger.info("File size: ~126.8 MB, please wait...")
        
        try:
            download_with_progress(url, zip_path)
            logger.info(f"✅ Download complete: {zip_path}")
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            logger.info("Tip: If download times out, you can download manually from:")
            logger.info(f"  {url}")
            logger.info(f"  Then place it in: {zip_path}")
            raise
        
        # Unzip
        logger.info("Unzipping file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"✅ Unzip complete")
        
        # Remove zip
        os.remove(zip_path)
        logger.info("Cleaning up temporary files")
        
        # Load data
        return load_txt_dataset(txt_path)


def load_txt_dataset(filepath):
    """Load the UCI dataset from TXT format"""
    logger.info(f"Loading data: {filepath}")
    
    # Read data (Semicolon separated, '?' represents missing values)
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
    
    # Reorder columns (put datetime first)
    cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
    df = df[cols]
    
    logger.info(f"✅ Data load complete: {df.shape}")
    
    return df


def analyze_dataset(df):
    """Analyze basic dataset information"""
    logger.info("\n" + "="*70)
    logger.info("Dataset Analysis")
    logger.info("="*70)
    
    logger.info(f"\n📊 Basic Information:")
    logger.info(f"  Sample count: {len(df):,}")
    logger.info(f"  Feature count: {len(df.columns)}")
    
    # Check time range
    if 'datetime' in df.columns:
        logger.info(f"  Time Range: {df['datetime'].min()} ~ {df['datetime'].max()}")
        time_span = df['datetime'].max() - df['datetime'].min()
        logger.info(f"  Duration: {time_span.days} days ({time_span.days/30.5:.1f} months)")
    
    # Missing value statistics
    missing = df.isnull().sum()
    total_missing = missing.sum()
    missing_pct = total_missing / df.size * 100
    
    logger.info(f"\n⚠️  Missing Values:")
    logger.info(f"  Total: {total_missing:,} ({missing_pct:.2f}%)")
    if total_missing > 0:
        logger.info(f"  Missing by Column:")
        for col, count in missing[missing > 0].items():
            pct = count / len(df) * 100
            logger.info(f"    {col}: {count:,} ({pct:.2f}%)")
    
    # Numerical feature statistics
    logger.info(f"\n📈 Numerical Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()
    logger.info(f"\n{stats.to_string()}")
    
    # Show first few rows
    logger.info(f"\n📋 First 5 Rows:")
    logger.info(f"\n{df.head().to_string()}")


def preprocess_for_training(df, output_path='data/processed/uci_household_clean.csv'):
    """
    Preprocess data for training
    
    Processing Steps:
    1. Handle missing values
    2. Resample to 15 minutes (as used in the paper)
    3. Feature engineering
    4. Save cleaned data
    """
    logger.info("\n" + "="*70)
    logger.info("Data Preprocessing")
    logger.info("="*70)
    
    df_clean = df.copy()
    
    # 1. Handle missing values
    logger.info("\n1️⃣ Handling missing values...")
    initial_missing = df_clean.isnull().sum().sum()
    
    # Forward fill
    df_clean = df_clean.fillna(method='ffill')
    # Backward fill (for initial missing values)
    df_clean = df_clean.fillna(method='bfill')
    
    remaining_missing = df_clean.isnull().sum().sum()
    logger.info(f"  Before: {initial_missing:,} missing values")
    logger.info(f"  After: {remaining_missing:,} missing values")
    
    # 2. Set time index and resample
    if 'datetime' in df_clean.columns:
        logger.info("\n2️⃣ Resampling to 15-minute intervals...")
        df_clean = df_clean.set_index('datetime')
        
        # Resample (15 minutes as per paper)
        df_clean = df_clean.resample('15T').mean()
        logger.info(f"  Samples after resampling: {len(df_clean):,}")
        
        # Reset index
        df_clean = df_clean.reset_index()
    
    # 3. Feature Engineering
    logger.info("\n3️⃣ Feature Engineering...")
    if 'datetime' in df_clean.columns:
        df_clean['hour'] = df_clean['datetime'].dt.hour
        df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
        df_clean['month'] = df_clean['datetime'].dt.month
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
        logger.info(f"  Added temporal features: hour, day_of_week, month, is_weekend")
    
    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"\n✅ Cleaned data saved: {output_path}")
    logger.info(f"  Final Shape: {df_clean.shape}")
    
    return df_clean


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download UCI Household Dataset')
    parser.add_argument(
        '--method',
        choices=['api', 'direct'],
        default='api',
        help='Download method: api=ucimlrepo package, direct=direct ZIP download'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='Raw data output directory'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Whether to perform preprocessing'
    )
    parser.add_argument(
        '--processed-output',
        default='data/processed/uci_household_clean.csv',
        help='Output path for preprocessed data'
    )
    
    args = parser.parse_args()
    
    try:
        # Download data
        df = download_uci_dataset(
            output_dir=args.output_dir,
            method=args.method
        )
        
        # Analyze data
        analyze_dataset(df)
        
        # Preprocess
        if args.preprocess:
            df_clean = preprocess_for_training(
                df,
                output_path=args.processed_output
            )
            
            logger.info("\n" + "="*70)
            logger.info("✅ All tasks complete!")
            logger.info("="*70)
            logger.info(f"Raw data: {args.output_dir}/uci_household_raw.csv")
            logger.info(f"Cleaned data: {args.processed_output}")
            logger.info("\nYou can now use the cleaned data for training:")
            logger.info(f"  python scripts/run_training.py --data {args.processed_output}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()