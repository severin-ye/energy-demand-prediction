"""
Inference Test Script
Loads the trained model, performs predictions on new data, and generates causal explanations and recommendations.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.inference_pipeline import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(n_samples=10):
    """Create test data (generate complete sequences)"""
    np.random.seed(42)
    
    # Create enough historical data points (at least 30 to generate sequences)
    hours = list(range(30))
    
    # Scenario 1: High Temperature & High Humidity sequence
    temp1 = [30 + 2*np.sin(h/24*2*np.pi) + np.random.randn()*0.5 for h in hours]
    hum1 = [70 + 5*np.cos(h/24*2*np.pi) + np.random.randn()*1 for h in hours]
    wind1 = [3 + np.random.randn()*0.5 for h in hours]
    
    scenario1 = pd.DataFrame({
        'Temperature': temp1,
        'Humidity': hum1,
        'WindSpeed': wind1,
        'EDP': [0.0] * 30,  # Placeholder
        'Hour': [(14 + h) % 24 for h in hours],
        'DayOfWeek': [2] * 30,
        'Month': [7] * 30
    })
    
    # Scenario 2: Low Temperature & Low Humidity sequence
    temp2 = [12 + 2*np.sin(h/24*2*np.pi) + np.random.randn()*0.5 for h in hours]
    hum2 = [40 + 5*np.cos(h/24*2*np.pi) + np.random.randn()*1 for h in hours]
    wind2 = [8 + np.random.randn()*0.5 for h in hours]
    
    scenario2 = pd.DataFrame({
        'Temperature': temp2,
        'Humidity': hum2,
        'WindSpeed': wind2,
        'EDP': [0.0] * 30,  # Placeholder
        'Hour': [(3 + h) % 24 for h in hours],
        'DayOfWeek': [1] * 30,
        'Month': [3] * 30
    })
    
    # Scenario 3: Moderate Temperature sequence
    temp3 = [20 + 2*np.sin(h/24*2*np.pi) + np.random.randn()*0.5 for h in hours]
    hum3 = [55 + 5*np.cos(h/24*2*np.pi) + np.random.randn()*1 for h in hours]
    wind3 = [5 + np.random.randn()*0.5 for h in hours]
    
    scenario3 = pd.DataFrame({
        'Temperature': temp3,
        'Humidity': hum3,
        'WindSpeed': wind3,
        'EDP': [0.0] * 30,  # Placeholder
        'Hour': [(10 + h) % 24 for h in hours],
        'DayOfWeek': [3] * 30,
        'Month': [5] * 30
    })
    
    scenarios = [
        ('High Temp & Humid Scenario', scenario1),
        ('Low Temp & Humid Scenario', scenario2),
        ('Moderate Temp Scenario', scenario3)
    ]
    
    return scenarios


def main():
    logger.info("=" * 80)
    logger.info(" " * 30 + "Inference Test Pipeline")
    logger.info("=" * 80)
    logger.info("")
    
    # 1. Load Model
    logger.info("[Step 1] Loading trained model...")
    model_dir = './outputs/training_run_1/models'
    
    if not os.path.exists(model_dir):
        logger.error(f"❌ Model directory does not exist: {model_dir}")
        logger.error("Please run the training script first: python scripts/run_training.py")
        return
    
    try:
        pipeline = InferencePipeline(model_dir)
        logger.info(f"✅ Model loaded successfully from: {model_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("")
    
    # 2. Create Test Data
    logger.info("[Step 2] Preparing test data...")
    scenarios = create_test_data()
    logger.info(f"✅ Test data preparation complete: {len(scenarios)} scenarios")
    logger.info(f"Scenarios: {', '.join([name for name, _ in scenarios])}")
    logger.info("")
    
    # 3. Run Inference
    logger.info("[Step 3] Executing inference...")
    logger.info("=" * 60)
    
    results_list = []
    
    for idx, (scenario_name, test_data) in enumerate(scenarios, 1):
        logger.info("")
        logger.info(f"Scenario {idx}: {scenario_name}")
        logger.info("-" * 60)
        logger.info(f"Input Data: {len(test_data)} time steps")
        logger.info(f"  Temp Range: {test_data['Temperature'].min():.1f} ~ {test_data['Temperature'].max():.1f}°C")
        logger.info(f"  Humidity Range: {test_data['Humidity'].min():.1f} ~ {test_data['Humidity'].max():.1f}%")
        logger.info(f"  Wind Speed Range: {test_data['WindSpeed'].min():.1f} ~ {test_data['WindSpeed'].max():.1f}m/s")
        logger.info("")
        
        try:
            # Run inference (recommendations disabled to avoid potential Bayesian network issues)
            result = pipeline.predict(test_data, generate_recommendations=False)
            
            # Display results (taking the last prediction)
            idx_last = -1
            logger.info(f"📊 Prediction Results:")
            logger.info(f"  EDP Predicted Value: {result['predictions'][idx_last]:.2f} kWh")
            logger.info(f"  EDP State: {result['edp_states'][idx_last]}")
            logger.info(f"  CAM Cluster: Cluster {result['cam_clusters'][idx_last]}")
            logger.info(f"  Attention Type: {result['attention_types'][idx_last]}")
            logger.info(f"  Sequences Generated: {len(result['predictions'])}")
            
            results_list.append({
                'scenario': scenario_name,
                'predictions': {
                    'edp': float(result['predictions'][idx_last]),
                    'state': result['edp_states'][idx_last],
                    'cam_cluster': int(result['cam_clusters'][idx_last]),
                    'attention_type': result['attention_types'][idx_last]
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("")
    logger.info("=" * 60)
    
    # 4. Save Results
    logger.info("")
    logger.info("[Step 4] Saving inference results...")
    
    output_dir = './outputs/inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'inference_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Results saved to: {output_file}")
    logger.info("")
    
    # 5. Summary Statistics
    logger.info("[Step 5] Results Summary")
    logger.info("=" * 60)
    
    if results_list:
        all_preds = []
        for r in results_list:
            if 'predictions' in r and 'edp' in r['predictions']:
                all_preds.append(r['predictions']['edp'])
        
        if all_preds:
            logger.info(f"EDP Prediction Statistics:")
            logger.info(f"  Minimum: {min(all_preds):.2f} kWh")
            logger.info(f"  Maximum: {max(all_preds):.2f} kWh")
            logger.info(f"  Average: {np.mean(all_preds):.2f} kWh")
    
    logger.info("")
    logger.info("✅ Inference test complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()