"""
UCI Household Dataset Download and Preprocessing Script
Data Source: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_uci_household_data(output_path='data/raw/household_power_consumption.txt'):
    """Download the UCI Household dataset"""
    if os.path.exists(output_path):
        logger.info(f"Data file already exists: {output_path}")
        return output_path
    
    logger.info("Starting download of UCI Household dataset...")
    
    import urllib.request
    import zipfile
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
    zip_path = 'data/raw/household_power_consumption.zip'
    
    try:
        logger.info(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        logger.info(f"Extracting to {os.path.dirname(output_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/raw')
        
        os.remove(zip_path)
        logger.info("Download complete!")
        return output_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Falling back to synthetic dataset")
        return None


def create_synthetic_dataset(
    n_samples=2000,
    output_path='data/processed/synthetic_energy_data.csv'
):
    """Create a synthetic energy dataset"""
    logger.info(f"Creating synthetic energy dataset (n={n_samples})...")
    
    np.random.seed(42)
    
    hours = np.arange(n_samples)
    
    # Temperature: Diurnal and seasonal cycles
    temperature = (
        20 +
        5 * np.sin(2 * np.pi * hours / 24) +
        3 * np.sin(2 * np.pi * hours / (24 * 30)) +
        np.random.randn(n_samples) * 2
    )
    
    # Humidity: Negatively correlated with temperature
    humidity = (
        70 -
        0.8 * temperature +
        np.random.randn(n_samples) * 5
    )
    
    # Wind Speed
    windspeed = np.abs(
        10 +
        3 * np.sin(2 * np.pi * hours / (24 * 7)) +
        np.random.randn(n_samples) * 3
    )
    
    # EDP (Energy Demand Prediction): Influenced by multiple factors
    hour_of_day = hours % 24
    peak_hours = ((hour_of_day >= 18) & (hour_of_day <= 22)).astype(float)
    
    edp = (
        100 +
        1.5 * temperature +
        0.3 * humidity +
        -0.5 * windspeed +
        30 * peak_hours +
        np.random.randn(n_samples) * 10
    )
    
    data = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'WindSpeed': windspeed,
        'EDP': edp,
        'Hour': hour_of_day,
        'DayOfWeek': (hours // 24) % 7,
        'Month': ((hours // 24 // 30) % 12) + 1
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    
    logger.info(f"Synthetic data saved to: {output_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data statistics:\n{data.describe()}")
    
    return data


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Data Preparation Script")
    logger.info("="*60)
    
    # Create synthetic dataset
    synthetic_data = create_synthetic_dataset(
        n_samples=2000,
        output_path='data/processed/synthetic_energy_data.csv'
    )
    
    logger.info("\n✅ Data preparation complete!")
    logger.info("Data path: data/processed/synthetic_energy_data.csv")
    logger.info("\n" + "="*60)
    logger.info("Ready to begin model training!")
    logger.info("="*60)