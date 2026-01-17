"""
Compare UCI Real Dataset with Synthetic Dataset

Demonstrates statistical differences between the two datasets.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_analyze(filepath, name):
    """Load and analyze the dataset"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing {name}")
    logger.info(f"{'='*70}")
    
    df = pd.read_csv(filepath)
    
    logger.info(f"📊 Basic Information:")
    logger.info(f"   File Path: {filepath}")
    logger.info(f"   Samples: {len(df):,}")
    logger.info(f"   Features: {len(df.columns)}")
    logger.info(f"   Column Names: {df.columns.tolist()}")
    
    # Statistics for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"\n📈 Numerical Feature Statistics:")
    
    for col in numeric_cols:
        if col in df.columns:
            stats = df[col].describe()
            logger.info(f"\n  {col}:")
            logger.info(f"    Mean: {stats['mean']:.2f}")
            logger.info(f"    Std Dev: {stats['std']:.2f}")
            logger.info(f"    Min: {stats['min']:.2f}")
            logger.info(f"    Max: {stats['max']:.2f}")
    
    return df


def main():
    """Main function"""
    
    # 1. Synthetic Data
    synthetic_df = load_and_analyze(
        'data/synthetic/training_data.csv',
        'Synthetic Dataset'
    )
    
    # 2. UCI Real Data
    uci_df = load_and_analyze(
        'data/processed/uci_household_clean.csv',
        'UCI Real Dataset'
    )
    
    # 3. Comparison
    logger.info(f"\n{'='*70}")
    logger.info(f"Dataset Comparison")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n📊 Scale Comparison:")
    logger.info(f"   Synthetic Data: {len(synthetic_df):,} samples")
    logger.info(f"   UCI Data: {len(uci_df):,} samples")
    logger.info(f"   UCI / Synthetic Ratio: {len(uci_df) / len(synthetic_df):.1f}x")
    
    # Compare power features
    if 'EDP' in synthetic_df.columns and 'Global_active_power' in uci_df.columns:
        logger.info(f"\n⚡ Power Comparison:")
        logger.info(f"   Synthetic Data EDP:")
        logger.info(f"    Mean: {synthetic_df['EDP'].mean():.2f} kWh")
        logger.info(f"    Std Dev: {synthetic_df['EDP'].std():.2f}")
        logger.info(f"    Range: [{synthetic_df['EDP'].min():.2f}, {synthetic_df['EDP'].max():.2f}]")
        
        logger.info(f"\n   UCI Data Global_active_power:")
        logger.info(f"    Mean: {uci_df['Global_active_power'].mean():.2f} kW")
        logger.info(f"    Std Dev: {uci_df['Global_active_power'].std():.2f}")
        logger.info(f"    Range: [{uci_df['Global_active_power'].min():.2f}, {uci_df['Global_active_power'].max():.2f}]")
    
    # Compare temperature (if available)
    if 'temperature' in synthetic_df.columns:
        logger.info(f"\n🌡️  Temperature Comparison:")
        logger.info(f"   Synthetic Data:")
        logger.info(f"    Mean: {synthetic_df['temperature'].mean():.2f}°C")
        logger.info(f"    Range: [{synthetic_df['temperature'].min():.2f}, {synthetic_df['temperature'].max():.2f}]")
        logger.info(f"\n   Note: UCI dataset does not contain temperature information")
    
    logger.info(f"\n💡 Summary:")
    logger.info(f"   ✅ Synthetic Data: Fast generation, controllable parameters, ideal for development/testing")
    logger.info(f"   ✅ UCI Data: Real-world data, ideal for research publication and actual deployment")
    logger.info(f"\nRecommendations:")
    logger.info(f"   - Development Phase: Use Synthetic Data (Fast iteration)")
    logger.info(f"   - Final Evaluation: Use UCI Data (Credible results)")


if __name__ == "__main__":
    main()