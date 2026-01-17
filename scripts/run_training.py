"""
Full Training Script - Supports Synthetic and UCI Real Datasets
"""

import sys
import os
import argparse
sys.path.append('/home/severin/Codelib/YS')

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from src.pipeline.train_pipeline import TrainPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def detect_data_type(data_path):
    """
    Auto-detect data type (Synthetic vs UCI)
    
    Returns:
        str: 'synthetic' or 'uci'
    """
    if 'uci' in data_path.lower():
        return 'uci'
    elif 'synthetic' in data_path.lower():
        return 'synthetic'
    else:
        # Read a small portion of data to check column names
        df = pd.read_csv(data_path, nrows=5)
        if 'Global_active_power' in df.columns:
            return 'uci'
        elif 'EDP' in df.columns:
            return 'synthetic'
        else:
            raise ValueError(f"Unable to recognize data type, columns: {df.columns.tolist()}")


def prepare_uci_data(data_path):
    """
    Prepare UCI dataset for training
    
    UCI Dataset Features:
    - Global_active_power, Global_reactive_power, Voltage, Global_intensity
    - Sub_metering_1, Sub_metering_2, Sub_metering_3
    - hour, day_of_week, month, is_weekend
    
    Returns format required for model input
    """
    logger.info("Loading UCI dataset...")
    df = pd.read_csv(data_path)
    
    # Select features for training
    feature_cols = [
        'Global_reactive_power',  # Reactive Power
        'Voltage',                # Voltage
        'Global_intensity',       # Current Intensity
    ]
    
    target_col = 'Global_active_power'  # Active Power as target
    
    # Check if necessary columns exist
    required_cols = feature_cols + [target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    # Prepare data
    prepared_df = df[[target_col] + feature_cols].copy()
    
    # Rename target column to EDP (compatible with existing code)
    prepared_df = prepared_df.rename(columns={target_col: 'EDP'})
    
    logger.info(f"✅ UCI data preparation complete: {prepared_df.shape}")
    logger.info(f"   Feature columns: {feature_cols}")
    logger.info(f"   Target column: EDP (original {target_col})")
    logger.info(f"   EDP Range: [{prepared_df['EDP'].min():.2f}, {prepared_df['EDP'].max():.2f}]")
    
    return prepared_df, feature_cols, 'EDP'


def prepare_synthetic_data(data_path):
    """Prepare synthetic dataset"""
    logger.info("Loading synthetic dataset...")
    df = pd.read_csv(data_path)
    
    feature_cols = ['Temperature', 'Humidity', 'WindSpeed']
    target_col = 'EDP'
    
    logger.info(f"✅ Synthetic data load complete: {df.shape}")
    
    return df, feature_cols, target_col


def main():
    parser = argparse.ArgumentParser(description='Train Energy Prediction Model')
    parser.add_argument(
        '--data',
        default='data/uci/splits/train.csv',
        help='Training data path (default uses UCI training set)'
    )
    parser.add_argument(
        '--data-type',
        choices=['auto', 'uci', 'synthetic'],
        default='auto',
        help='Data type (auto for auto-detection)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: outputs/training/YY-MM-DD)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=20,
        help='Time series sequence length (number of time steps)'
    )
    
    args = parser.parse_args()
    
    # If output directory is not specified, use outputs/training/date/ format
    if args.output_dir is None:
        date_suffix = datetime.now().strftime('%y-%m-%d')
        args.output_dir = f'./outputs/training/{date_suffix}'
    
    logger.info("="*80)
    logger.info(" "*20 + "Full Training Pipeline" + " "*20)
    logger.info("="*80)
    
    start_time = time.time()
    
    # 1. Detect data type
    if args.data_type == 'auto':
        data_type = detect_data_type(args.data)
        logger.info(f"\n[Auto-detect] Data type: {data_type}")
    else:
        data_type = args.data_type
    
    # 2. Load and prepare data
    logger.info(f"\n[Step 1] Loading training data: {args.data}...")
    
    try:
        if data_type == 'uci':
            train_data, feature_cols, target_col = prepare_uci_data(args.data)
        else:
            train_data, feature_cols, target_col = prepare_synthetic_data(args.data)
            
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}")
        if data_type == 'uci':
            logger.info("Please run: python scripts/split_uci_dataset.py first")
        else:
            logger.info("Please run: python scripts/prepare_data.py first")
        return
    except Exception as e:
        logger.error(f"Data load failed: {e}")
        return
    
    # 2. Configure training parameters
    logger.info("\n[Step 2] Configuring training parameters...")
    
    config = {
        # Data preprocessing
        'sequence_length': args.sequence_length,
        'feature_cols': feature_cols,
        'target_col': target_col,
        
        # Prediction model
        'cnn_filters': [64, 32] if data_type == 'uci' else [32, 16],
        'lstm_units': 64 if data_type == 'uci' else 32,
        'attention_units': 25,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_split': 0.2,
        
        # State classification
        'n_states': 3,
        'state_names': ['Lower', 'Normal', 'Peak'],
        
        # Discretization
        'n_bins': 4,
        'bin_labels': ['Low', 'Medium', 'High', 'VeryHigh'],
        
        # DLP clustering
        'n_cam_clusters': 3,
        'n_attention_clusters': 3,
        
        # Association rules
        'min_support': 0.05,
        'min_confidence': 0.6,
        'min_lift': 1.2,
        
        # Bayesian network
        'bn_score_fn': 'bic',
        'bn_max_iter': 50,
        'bn_estimator': 'mle'
    }
    
    logger.info(f"Data type: {data_type}")
    logger.info(f"Training config: epochs={config['epochs']}, batch_size={config['batch_size']}")
    logger.info(f"Model scale: CNN={config['cnn_filters']}, LSTM={config['lstm_units']}")
    
    # 3. Create training pipeline
    logger.info("\n[Step 3] Creating training pipeline...")
    pipeline = TrainPipeline(
        config=config,
        output_dir=args.output_dir
    )
    
    # 4. Run training
    logger.info("\n[Step 4] Starting training...\n")
    
    try:
        results = pipeline.run(train_data)
        
        # 5. Output results
        logger.info("\n" + "="*80)
        logger.info(" "*30 + "Training Complete!" + " "*30)
        logger.info("="*80)
        
        logger.info("\nTraining Summary:")
        logger.info(f"   Data shapes: {results['data_shapes']}")
        logger.info(f"   DLP shapes: {results['dlp_shapes']}")
        logger.info(f"   State distribution: {results['state_distribution']}")
        logger.info(f"   Cluster distribution:")
        logger.info(f"     CAM: {results['cluster_distributions']['cam']}")
        logger.info(f"     Attention: {results['cluster_distributions']['attention']}")
        logger.info(f"   Candidate edge count: {len(results['candidate_edges'])}")
        logger.info(f"   Bayesian network edges: {len(results['bn_edges'])}")
        
        # Calculate training time
        elapsed_time = time.time() - start_time
        logger.info(f"\nTotal training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        logger.info("\nOutput files:")
        logger.info(f"   Models: {args.output_dir}/models/")
        logger.info(f"   Results: {args.output_dir}/results/")
        logger.info(f"   Config: {args.output_dir}/config.json")
        return results
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n" + "="*80)
        print("Next step: Run inference test")
        print("   python scripts/run_inference.py")
        print("="*80)