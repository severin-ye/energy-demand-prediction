"""
Split UCI Dataset into Training and Test Sets

Usage:
    python scripts/split_uci_dataset.py --test-ratio 0.05
"""
import sys
import os
import argparse
import logging

# Add the project root directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.uci_loader import UCIDataLoader
from src.data_processing.data_splitter import DataSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Split UCI dataset into training and test sets'
    )
    parser.add_argument(
        '--input',
        default='data/uci/processed/uci_household_clean.csv',
        help='Input preprocessed data file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/uci/splits',
        help='Output directory'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.05,
        help='Ratio of the test set (default 0.05, i.e., 5%%)'
    )
    parser.add_argument(
        '--split-method',
        choices=['sequential', 'random'],
        default='sequential',
        help='Split method: sequential or random'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (only used for random method)'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("="*70)
        logger.info("UCI Dataset Splitting")
        logger.info("="*70)
        
        # 1. Load data
        logger.info(f"\n📂 Loading data: {args.input}")
        import pandas as pd
        df = pd.read_csv(args.input)
        logger.info(f"  Samples: {len(df):,}")
        logger.info(f"  Features: {len(df.columns)}")
        
        # Show time range
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"  Time range: {df['datetime'].min()} ~ {df['datetime'].max()}")
        
        # 2. Split dataset
        logger.info(f"\n✂️  Splitting dataset")
        logger.info(f"  Method: {args.split_method}")
        logger.info(f"  Test ratio: {args.test_ratio*100:.1f}%")
        
        splitter = DataSplitter(output_dir=args.output_dir)
        
        if args.split_method == 'sequential':
            train_df, test_df = splitter.split_sequential(df, test_ratio=args.test_ratio)
        else:
            train_df, test_df = splitter.split_random(
                df, 
                test_ratio=args.test_ratio,
                random_state=args.random_state
            )
        
        # 3. Save splits
        logger.info(f"\n💾 Saving split datasets")
        paths = splitter.save_splits(train_df, test_df)
        
        # 4. Display statistics
        logger.info(f"\n📊 Split Statistics")
        info = splitter.get_split_info(train_df, test_df)
        
        logger.info(f"  Total samples: {info['total_samples']:,}")
        logger.info(f"  Training set: {info['train_samples']:,} ({info['train_ratio']*100:.2f}%)")
        logger.info(f"  Test set: {info['test_samples']:,} ({info['test_ratio']*100:.2f}%)")
        
        if 'train_time_range' in info:
            logger.info(f"\n  Training set time range:")
            logger.info(f"    {info['train_time_range']['start']} ~ {info['train_time_range']['end']}")
            logger.info(f"  Test set time range:")
            logger.info(f"    {info['test_time_range']['start']} ~ {info['test_time_range']['end']}")
        
        # 5. Data quality check
        logger.info(f"\n🔍 Data Quality Check")
        
        train_missing = train_df.isnull().sum().sum()
        test_missing = test_df.isnull().sum().sum()
        
        logger.info(f"  Training set missing values: {train_missing}")
        logger.info(f"  Test set missing values: {test_missing}")
        
        # Check target variable
        if 'Global_active_power' in train_df.columns:
            logger.info(f"\n  Target variable (Global_active_power):")
            logger.info(f"    Training set - Mean: {train_df['Global_active_power'].mean():.3f}, "
                       f"Std: {train_df['Global_active_power'].std():.3f}")
            logger.info(f"    Test set - Mean: {test_df['Global_active_power'].mean():.3f}, "
                       f"Std: {test_df['Global_active_power'].std():.3f}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ Split complete!")
        logger.info("="*70)
        logger.info(f"Training set: {paths['train']}")
        logger.info(f"Test set: {paths['test']}")
        logger.info(f"\nYou can now use the training set for model training:")
        logger.info(f"  python scripts/run_training.py --data {paths['train']}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()