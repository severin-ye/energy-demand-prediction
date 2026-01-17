"""
UCI Data Inference Test
Uses a trained model to make predictions on the test set and generates readable result reports.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path

from src.pipeline.inference_pipeline import InferencePipeline
from src.visualization.inference_visualizer import InferenceVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_file, n_samples=100, min_samples=50):
    """
    Load test set data.
    
    Args:
        test_file: Path to the test set file.
        n_samples: Number of samples to use for testing.
        min_samples: Minimum required samples (default 50, corresponds to sequence_length=20).
    """
    logger.info(f"Loading test data: {test_file}")
    df = pd.read_csv(test_file)
    
    # Check if sample count is sufficient
    actual_samples = min(len(df), n_samples)
    if actual_samples < min_samples:
        logger.error(f"\n❌ Insufficient samples!")
        logger.error(f"   Current samples: {actual_samples}")
        logger.error(f"   Minimum required: {min_samples}")
        logger.error(f"\nReason: Model uses sequence_length=20, requiring at least 50 samples to generate enough sequences.")
        logger.error(f"Formula: Num Sequences = Num Samples - 20")
        logger.error(f"   • Samples=20 → Sequences=0  ❌")
        logger.error(f"   • Samples=30 → Sequences=10 ⚠️")
        logger.error(f"   • Samples=50 → Sequences=30 ✅")
        logger.error(f"\nSolution:")
        logger.error(f"   python scripts/run_inference_uci.py \\")
        logger.error(f"     --model-dir {Path(test_file).parent.parent}/training/26-01-16/models \\")
        logger.error(f"     --test-data {test_file} \\")
        logger.error(f"     --n-samples {min_samples}  # or more")
        raise ValueError(f"Sample count ({actual_samples}) is less than minimum requirement ({min_samples}). Cannot generate sequences.")
    
    # Take only the first n_samples to speed up inference
    if len(df) > n_samples:
        df = df.iloc[:n_samples].copy()
        logger.info(f"Using first {n_samples} samples for inference")
    
    # Prepare features
    feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity']
    target_col = 'Global_active_power'
    
    # Rename target column to EDP (compatible with naming during training)
    df_test = df[feature_cols + [target_col]].copy()
    df_test = df_test.rename(columns={target_col: 'EDP'})
    
    logger.info(f"Test data shape: {df_test.shape}")
    logger.info(f"EDP Range: [{df_test['EDP'].min():.2f}, {df_test['EDP'].max():.2f}]")
    
    return df_test


def format_prediction_results(results):
    """
    Format prediction results into a readable report.
    
    Args:
        results: Inference results dictionary.
    """
    report = []
    report.append("=" * 80)
    report.append(" " * 25 + "📊 Inference Result Report")
    report.append("=" * 80)
    
    # 1. Basic Information
    report.append("\n[1. Basic Information]")
    report.append(f"  Test samples: {len(results['predictions'])}")
    report.append(f"  Prediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. Prediction Statistics
    predictions = results['predictions']
    true_values = results.get('true_values', [])
    
    report.append("\n[2. Prediction Statistics]")
    report.append(f"  Predicted EDP Range: [{predictions.min():.3f}, {predictions.max():.3f}] kW")
    report.append(f"  Predicted EDP Mean: {predictions.mean():.3f} kW")
    report.append(f"  Predicted EDP Std Dev: {predictions.std():.3f} kW")
    
    if len(true_values) > 0:
        # Ensure true_values matches predictions length
        true_values = np.array(true_values[:len(predictions)])
        report.append(f"\n  Actual EDP Range: [{true_values.min():.3f}, {true_values.max():.3f}] kW")
        report.append(f"  Actual EDP Mean: {true_values.mean():.3f} kW")
        
        # Calculate error metrics
        errors = predictions - true_values
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = np.mean(np.abs(errors / (true_values + 1e-8))) * 100
        
        report.append(f"\n  [Performance Metrics]")
        report.append(f"    MAE (Mean Absolute Error): {mae:.4f} kW")
        report.append(f"    RMSE (Root Mean Square Error): {rmse:.4f} kW")
        report.append(f"    MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # 3. State Distribution
    if 'edp_states' in results:
        states = results['edp_states']
        state_counts = pd.Series(states).value_counts()
        
        report.append("\n[3. EDP State Distribution]")
        total = len(states)
        for state, count in state_counts.items():
            percentage = count / total * 100
            bar = "█" * int(percentage / 2)
            report.append(f"  {state:8s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # 4. DLP Feature Distribution
    report.append("\n[4. Deep Learning Parameter (DLP) Features]")
    
    if 'cam_clusters' in results:
        cam_clusters = results['cam_clusters']
        cam_counts = pd.Series(cam_clusters).value_counts().sort_index()
        
        report.append("  CAM Cluster Distribution:")
        for cluster, count in cam_counts.items():
            percentage = count / len(cam_clusters) * 100
            report.append(f"    Cluster {cluster}: {count:4d} ({percentage:5.1f}%)")
    
    if 'attention_types' in results:
        attention_types = results['attention_types']
        attention_counts = pd.Series(attention_types).value_counts()
        
        report.append("\n  Attention Type Distribution:")
        for att_type, count in attention_counts.items():
            percentage = count / len(attention_types) * 100
            report.append(f"    {att_type:10s}: {count:4d} ({percentage:5.1f}%)")
    
    # 5. Sample Case Presentation
    report.append("\n[5. Typical Sample Cases]")
    
    # Select 3 representative samples
    n_samples_pred = len(predictions)
    indices = [0, n_samples_pred // 2, n_samples_pred - 1]
    
    for i, idx in enumerate(indices):
        if idx >= n_samples_pred:
            continue
            
        report.append(f"\n  Sample {i+1} (Index {idx}):")
        report.append(f"    Predicted EDP: {predictions[idx]:.3f} kW")
        
        if len(true_values) > 0 and idx < len(true_values):
            error = predictions[idx] - true_values[idx]
            report.append(f"    Actual EDP: {true_values[idx]:.3f} kW")
            report.append(f"    Error: {error:+.3f} kW ({error/true_values[idx]*100:+.1f}%)")
        
        if 'edp_states' in results and idx < len(results['edp_states']):
            report.append(f"    State: {results['edp_states'][idx]}")
        
        if 'cam_clusters' in results and idx < len(results['cam_clusters']):
            report.append(f"    CAM Cluster: {results['cam_clusters'][idx]}")
        
        if 'attention_types' in results and idx < len(results['attention_types']):
            report.append(f"    Attention: {results['attention_types'][idx]}")
    
    # 6. Prediction Interval Statistics
    report.append("\n[6. Prediction Interval Statistics]")
    
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '>3.0']
    
    pred_binned = pd.cut(predictions, bins=bins, labels=labels, include_lowest=True)
    bin_counts = pred_binned.value_counts().sort_index()
    
    report.append("  Predicted EDP Distribution (kW):")
    for label, count in bin_counts.items():
        percentage = count / len(predictions) * 100
        bar = "▓" * int(percentage / 2)
        report.append(f"    {label:10s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def save_detailed_results(results, output_file):
    """
    Save detailed results to a CSV file.
    
    Args:
        results: Inference results dictionary.
        output_file: Output file path.
    """
    data = {
        'index': range(len(results['predictions'])),
        'predicted_edp': results['predictions'],
    }
    
    if 'true_values' in results and len(results['true_values']) > 0:
        true_vals = np.array(results['true_values'][:len(results['predictions'])])
        data['true_edp'] = true_vals
        data['error'] = results['predictions'] - true_vals
        data['abs_error'] = np.abs(data['error'])
        data['relative_error'] = data['error'] / (true_vals + 1e-8)
    
    if 'edp_states' in results:
        data['state'] = results['edp_states']
    
    if 'cam_clusters' in results:
        data['cam_cluster'] = results['cam_clusters']
    
    if 'attention_types' in results:
        data['attention_type'] = results['attention_types']
    
    df_results = pd.DataFrame(data)
    df_results.to_csv(output_file, index=False)
    logger.info(f"Detailed results saved to: {output_file}")
    
    return df_results


def generate_html_reports(results: dict, test_data: pd.DataFrame, output_dir: Path):
    """
    Generate an HTML visualization report for each sample.
    
    Args:
        results: Inference results dictionary.
        test_data: Test data.
        output_dir: Output directory.
    """
    logger.info("\nGenerating HTML visualization reports...")
    
    visualizer = InferenceVisualizer()
    html_dir = output_dir / 'html_reports'
    html_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = results['predictions']
    true_values = results.get('true_values', [])
    edp_states = results.get('edp_states', [])
    cam_clusters = results.get('cam_clusters', [])
    attention_types = results.get('attention_types', [])
    
    median_value = np.median(predictions)
    
    num_samples_to_report = min(len(predictions), 10)
    logger.info(f"Will generate HTML reports for the first {num_samples_to_report} samples")
    
    for idx in range(num_samples_to_report):
        sample_data = {
            'sample_id': idx,
            'window_size': 'N/A',
            'target_name': 'Global Active Power (EDP)',
            'input_features': {
                'Global Reactive Power': test_data.iloc[idx]['Global_reactive_power'],
                'Voltage': test_data.iloc[idx]['Voltage'],
                'Global Intensity': test_data.iloc[idx]['Global_intensity'],
            },
            'cam_cluster': int(cam_clusters[idx]) if idx < len(cam_clusters) else 0,
            'attention_type': attention_types[idx] if idx < len(attention_types) else 'Unknown',
            'prediction': float(predictions[idx]),
            'actual_value': float(true_values[idx]) if idx < len(true_values) else 0,
            'error': float(predictions[idx] - true_values[idx]) if idx < len(true_values) else 0,
            'error_percent': float((predictions[idx] - true_values[idx]) / (true_values[idx] + 1e-8) * 100) if idx < len(true_values) else 0,
            'state': edp_states[idx] if idx < len(edp_states) else 'Unknown',
            'median_value': float(median_value),
            'discrete_features': {
                'Global Reactive Power': _discretize_value(test_data.iloc[idx]['Global_reactive_power'], 'reactive'),
                'Voltage': _discretize_value(test_data.iloc[idx]['Voltage'], 'voltage'),
                'Global Intensity': _discretize_value(test_data.iloc[idx]['Global_intensity'], 'intensity'),
            },
            'causal_explanation': _generate_causal_explanation(
                state=edp_states[idx] if idx < len(edp_states) else 'Unknown',
                prediction=float(predictions[idx]),
                actual=float(true_values[idx]) if idx < len(true_values) else 0,
                features=test_data.iloc[idx]
            ),
            'recommendations': _generate_recommendations(
                state=edp_states[idx] if idx < len(edp_states) else 'Unknown',
                error_percent=float((predictions[idx] - true_values[idx]) / (true_values[idx] + 1e-8) * 100) if idx < len(true_values) else 0,
                features=test_data.iloc[idx]
            )
        }
        
        html_file = html_dir / f'sample_{idx:03d}.html'
        visualizer.generate_html(sample_data, idx, html_file)
        
        if idx == 0:
            logger.info(f"✅ Sample report: {html_file}")
    
    logger.info(f"✅ Generated {num_samples_to_report} HTML reports to: {html_dir}")
    _generate_index_page(html_dir, num_samples_to_report)
    logger.info(f"✅ Index page: {html_dir}/index.html")


def _discretize_value(value: float, feature_type: str) -> str:
    """Discretize numerical values."""
    if feature_type == 'reactive':
        if value < 0.05: return 'Very Low'
        elif value < 0.15: return 'Moderate'
        else: return 'High'
    elif feature_type == 'voltage':
        if value < 230: return 'Low'
        elif value < 245: return 'Normal'
        else: return 'High'
    elif feature_type == 'intensity':
        if value < 5: return 'Low'
        elif value < 15: return 'Moderate'
        else: return 'High'
    return 'Unknown'


def _generate_causal_explanation(state: str, prediction: float, actual: float, features) -> str:
    """Generate causal analysis description."""
    voltage = features['Voltage']
    reactive = features['Global_reactive_power']
    intensity = features['Global_intensity']
    
    explanations = []
    
    if state == 'Peak':
        explanations.append(f"<strong>Load Peak State</strong>: Predicted power is {prediction:.3f} kW, higher than normal levels")
        if voltage > 240: explanations.append(f"• Voltage is high ({voltage:.1f}V), potential grid fluctuations")
        if intensity > 10: explanations.append(f"• Intensity is high ({intensity:.1f}A), heavy device load")
    elif state == 'Lower':
        explanations.append(f"<strong>Low Load State</strong>: Predicted power is {prediction:.3f} kW, at a lower level")
        if voltage < 235: explanations.append(f"• Voltage is low ({voltage:.1f}V), small electricity load")
        if intensity < 3: explanations.append(f"• Intensity is small ({intensity:.1f}A), fewer devices in use")
    else:
        explanations.append(f"<strong>Normal Load State</strong>: Predicted power is {prediction:.3f} kW")
    
    if reactive > 0.2: explanations.append(f"• Reactive power is high ({reactive:.3f} kW), inductive load present")
    elif reactive < 0.05: explanations.append(f"• Reactive power is very low ({reactive:.3f} kW), load is mainly resistive")
    
    if actual > 0:
        error_pct = abs(prediction - actual) / actual * 100
        if error_pct < 10: explanations.append(f"• Prediction error {error_pct:.1f}%, high accuracy")
        elif error_pct < 30: explanations.append(f"• Prediction error {error_pct:.1f}%, moderate accuracy")
        else: explanations.append(f"• Prediction error {error_pct:.1f}%, some deviation")
    
    return '<br>'.join(explanations) if explanations else 'Current data is normal, no abnormal factors detected.'


def _generate_recommendations(state: str, error_percent: float, features) -> list:
    """Generate optimization recommendations."""
    recommendations = []
    voltage = features['Voltage']
    reactive = features['Global_reactive_power']
    intensity = features['Global_intensity']
    
    if state == 'Peak':
        recommendations.append({
            'action': 'Peak Shaving',
            'explanation': 'Currently in load peak; suggest adjusting usage times to avoid peak periods',
            'expected_impact': 'Reduce electricity costs by 10-20%'
        })
        if intensity > 15:
            recommendations.append({
                'action': 'Check High-power Devices',
                'explanation': f'Intensity reached {intensity:.1f}A; check if high-power devices are running concurrently',
                'expected_impact': 'Avoid overload risk'
            })
    
    if abs(error_percent) > 50:
        recommendations.append({
            'action': 'Model Optimization',
            'explanation': f'Large prediction error ({abs(error_percent):.1f}%); suggest: 1) Increase training samples for similar scenarios 2) Check data quality',
            'expected_impact': 'Improve prediction accuracy by 20-30%'
        })
    
    if voltage < 220:
        recommendations.append({
            'action': 'Voltage Monitoring - Under-voltage',
            'explanation': f'Voltage too low ({voltage:.1f}V < 220V), may affect device operation',
            'expected_impact': 'Ensure electrical safety'
        })
    elif voltage > 250:
        recommendations.append({
            'action': 'Voltage Monitoring - Over-voltage',
            'explanation': f'Voltage too high ({voltage:.1f}V > 250V); contact power department',
            'expected_impact': 'Ensure equipment safety'
        })
    
    if reactive > 0.3:
        recommendations.append({
            'action': 'Power Factor Compensation',
            'explanation': f'Reactive power is high ({reactive:.3f} kW); suggest installing compensation capacitors',
            'expected_impact': 'Reduce electricity bill by 5-10%'
        })
    
    if not recommendations:
        recommendations.append({'action': 'Maintain Status Quo', 'explanation': 'Current pattern is reasonable', 'expected_impact': 'Stable operation'})
    
    return recommendations


def _generate_index_page(html_dir: Path, num_samples: int):
    """Generate a simple index page."""
    index_html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Inference Results Index</title>
    <style>
        body {{ font-family: sans-serif; background: #f5f5f5; padding: 30px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: #2c3e50; color: white; padding: 30px; border-bottom: 3px solid #3498db; }}
        .content {{ padding: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .card {{ background: white; border: 2px solid #ecf0f1; border-radius: 6px; padding: 20px; text-align: center; text-decoration: none; color: #2c3e50; }}
        .card:hover {{ border-color: #3498db; transform: translateY(-2px); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Power Load Forecasting - Inference Results</h1>
            <p>UCI Household Electricity Consumption Dataset</p>
        </div>
        <div class="content">
            <div class="grid">
'''
    for i in range(num_samples):
        index_html += f'''
                <a href="sample_{i:03d}.html" class="card">
                    <div style="font-size:1.3em; font-weight:600; color:#3498db;">#{i}</div>
                    <div style="font-size:0.85em; color:#7f8c8d;">Sample Analysis</div>
                </a>
'''
    index_html += f'''
            </div>
            <div style="margin-top:30px; text-align:center; color:#7f8c8d;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </div>
</body>
</html>
'''
    with open(html_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='UCI Dataset Inference Test')
    parser.add_argument('--model-dir', default='outputs/training_uci/models', help='Model directory')
    parser.add_argument('--test-data', default='data/uci/splits/test.csv', help='Test data file')
    parser.add_argument('--n-samples', type=int, default=100, help='Test samples count (default 100)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        model_dir_path = Path(args.model_dir)
        model_name = model_dir_path.parent.parent.name if model_dir_path.parent.name == 'models' else model_dir_path.parent.name
        args.output_dir = f'outputs/inference/{model_name}/{datetime.now().strftime("%y-%m-%d_%H-%M")}'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("=" * 80)
        logger.info(" " * 30 + "Inference Test Started")
        logger.info("=" * 80)
        
        test_data = load_test_data(args.test_data, args.n_samples)
        true_values = test_data['EDP'].values
        
        pipeline = InferencePipeline(models_dir=args.model_dir)
        results = pipeline.predict(test_data)
        results['true_values'] = true_values
        
        report = format_prediction_results(results)
        print("\n" + report)
        
        with open(output_dir / 'inference_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        save_detailed_results(results, output_dir / 'inference_details.csv')
        
        true_vals_matched = true_values[:len(results['predictions'])]
        json_results = {
            'predictions': results['predictions'].tolist(),
            'true_values': true_vals_matched.tolist(),
            'statistics': {
                'mae': float(np.abs(results['predictions'] - true_vals_matched).mean()),
                'rmse': float(np.sqrt(((results['predictions'] - true_vals_matched) ** 2).mean())),
                'n_samples': len(results['predictions'])
            }
        }
        with open(output_dir / 'inference_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        generate_html_reports(results, test_data, output_dir)
        
        print("\n" + "=" * 80)
        print("📊 Performance Summary")
        print("=" * 80)
        print(f"MAE:  {json_results['statistics']['mae']:.4f} kW")
        print(f"RMSE: {json_results['statistics']['rmse']:.4f} kW")
        print(f"💡 View HTML: {output_dir}/html_reports/index.html")
        
    except Exception as e:
        logger.error(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())