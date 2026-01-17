#!/usr/bin/env python3
"""
Test Output Directory Structure
"""

from pathlib import Path
from datetime import datetime

def test_training_dir():
    """Test training directory generation"""
    date_suffix = datetime.now().strftime('%y-%m-%d')
    output_dir = f'./outputs/training/{date_suffix}'
    print(f"Training Directory: {output_dir}")
    return output_dir

def test_inference_dir(model_dir):
    """Test inference directory generation"""
    model_dir_path = Path(model_dir)
    
    # Extract name from the model directory
    if model_dir_path.parent.name == 'models':
        # If it is xxx/models, take the parent directory's name
        model_name = model_dir_path.parent.parent.name
    else:
        model_name = model_dir_path.parent.name
    
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M')
    output_dir = f'outputs/inference/{model_name}/{timestamp}'
    
    print(f"\nModel Directory: {model_dir}")
    print(f"Extracted Model Name: {model_name}")
    print(f"Inference Directory: {output_dir}")
    return output_dir

if __name__ == '__main__':
    print("=" * 60)
    print("Output Directory Structure Test")
    print("=" * 60)
    
    # Test training directory
    train_dir = test_training_dir()
    
    # Test inference directory (multiple scenarios)
    test_cases = [
        'outputs/training/26-01-16/models',
        './outputs/training/26-01-17/models',
        'custom_models/my_model/models',
        'another/path/to/models',
    ]
    
    for model_dir in test_cases:
        test_inference_dir(model_dir)
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)