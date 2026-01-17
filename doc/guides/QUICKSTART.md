# ⚡ Quick Start Guide

## Environment Preparation

### 1. Install Dependencies

```bash
cd /home/severin/Codelib/YS

# Activate virtual environment (if configured)
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run core module tests (approx. 30 seconds)
python tests/test_core_modules.py
```

Expected Output:
```
✅ All 5 core modules tested successfully!
```

---

## Usage Methods

### Option 1: Training a New Model

```python
import sys
sys.path.append('/home/severin/Codelib/YS')

from src.pipeline.train_pipeline import TrainPipeline
import pandas as pd
import numpy as np

# 1. Prepare data (Example: Using random data)
np.random.seed(42)
n_samples = 500

train_data = pd.DataFrame({
    'Temperature': np.random.randn(n_samples) * 10 + 20,
    'Humidity': np.random.randn(n_samples) * 15 + 60,
    'WindSpeed': np.random.randn(n_samples) * 5 + 10,
    'EDP': np.random.randn(n_samples) * 30 + 100
})

# 2. Create training pipeline
pipeline = TrainPipeline(output_dir='./outputs')

# 3. Run full training process (9 steps)
results = pipeline.run(train_data)

# 4. View results
print("\nTraining completed!")
print(f"Data shapes: {results['data_shapes']}")
print(f"State distribution: {results['state_distribution']}")
print(f"Number of candidate edges: {len(results['candidate_edges'])}")
print(f"Bayesian Network edges: {results['bn_edges']}")
```

**Output Directory Structure**:
```
outputs/
├── models/
│   ├── preprocessor.pkl
│   ├── predictor.h5
│   ├── state_classifier.pkl
│   ├── discretizer.pkl
│   ├── cam_clusterer.pkl
│   ├── attention_clusterer.pkl
│   └── bayesian_network.bif
├── results/
│   ├── association_rules.csv
│   └── bayesian_network.png
└── config.json
```

---

### Option 2: Inference with a Trained Model

```python
from src.pipeline.inference_pipeline import InferencePipeline
import pandas as pd
import numpy as np

# 1. Load model
pipeline = InferencePipeline(models_dir='./outputs/models')

# 2. Prepare new data (minimum 20 rows, as sequence_length=20)
new_data = pd.DataFrame({
    'Temperature': [25.3] * 25,
    'Humidity': [62.5] * 25,
    'WindSpeed': [8.2] * 25
})

# 3. Single sample prediction + recommendation generation
report = pipeline.predict_single(new_data, verbose=True)
print(report)
```

**Output Example**:
```
============================================================
           Prediction Result
============================================================
Predicted Load: 125.50 kWh
Load State: Peak
CAM Cluster: 1
Attention Type: Early


============================================================
   Energy Consumption Prediction & Optimization Report
============================================================

【Current State】
  Temperature: High
  Humidity: Medium
  Wind Speed: Low

【Predicted Load】
  125.50 kWh

【Optimization Recommendations】 (3 total)

1. It is recommended to lower the indoor temperature setting, e.g., adjusting air conditioning to Low, which is expected to significantly reduce peak load probability (approx. 23.5%).
   Current: Temperature=High → Recommended: Low
   Expected Effect: Peak probability drops from 68.2% to 44.7%

2. It is recommended to increase dehumidification power, but notice energy balance; this is expected to noticeably reduce peak load probability (approx. 18.3%).
   Current: Humidity=Medium → Recommended: High
   Expected Effect: Peak probability drops from 68.2% to 49.9%

3. ...

============================================================
Note: These recommendations are based on Causal Bayesian Network inference and are for reference only.
```

---

### Option 3: Batch Prediction (No Explanation)

```python
# Batch predict multiple samples
results_df = pipeline.batch_predict(
    new_data,
    output_path='./outputs/predictions.csv'
)

print(results_df)
```

Output:
```
   Prediction  EDP_State  CAM_Cluster Attention_Type
0      125.50       Peak            1          Early
1      118.30     Normal            0          Other
2      132.70       Peak            2           Late
```

---

## Common Features

### 1. Sensitivity Analysis

```python
from src.models.bayesian_net import CausalBayesianNetwork
from src.inference.causal_inference import CausalInference

# Load Bayesian Network
bn = CausalBayesianNetwork()
bn.load_model('./outputs/models/bayesian_network.bif')

# Create Causal Inference tool
ci = CausalInference(bn, target_var='EDP_State')

# Sensitivity analysis
features = ['Temperature', 'Humidity', 'WindSpeed']
sensitivity = ci.sensitivity_analysis(features, target_state='Peak')

print(sensitivity[['Feature', 'Prob_Range', 'Max_Value', 'Max_Prob']])
```

### 2. Tornado Chart Visualization

```python
# Generate Tornado Chart
ci.tornado_chart(
    top_k=5,
    output_path='./outputs/tornado_chart.png'
)
```

### 3. Counterfactual Inference

```python
# What if temperature changes from High to Low?
counterfactual = ci.counterfactual_analysis(
    actual_evidence={'Temperature': 'High', 'Humidity': 'Medium'},
    intervention={'Temperature': 'Low'},
    target_state='Peak'
)

print(counterfactual['interpretation'])
```

### 4. Average Causal Effect (ACE)

```python
# Calculate ACE for Temperature High vs Low
ace = ci.average_causal_effect(
    treatment_var='Temperature',
    treatment_value='High',
    control_value='Low',
    target_state='Peak'
)

print(f"ACE: {ace:.3f}")
```

---

## Using Real Data

### UCI Household Dataset Example

```python
# Download and prepare UCI data
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

import pandas as pd

# Load data
df = pd.read_csv(
    'household_power_consumption.txt',
    sep=';',
    parse_dates={'Datetime': ['Date', 'Time']},
    na_values=['?']
)

# Preprocessing
df = df.dropna()
df = df.set_index('Datetime')

# Resample to hourly frequency
df_hourly = df.resample('H').mean()

# Select features
train_data = df_hourly[['Global_active_power', 'Voltage', 'Global_intensity']].copy()
train_data.columns = ['EDP', 'Temperature', 'Humidity']  # Rename to columns required by model

# Add WindSpeed feature (Example)
train_data['WindSpeed'] = np.random.randn(len(train_data)) * 5 + 10

# Train
pipeline = TrainPipeline(output_dir='./outputs_uci')
results = pipeline.run(train_data[:5000])  # Use first 5000 rows
```

---

## Configuring Custom Parameters

```python
# Custom configuration
custom_config = {
    'sequence_length': 30,  # Change to 30 time steps
    'epochs': 100,          # Train for 100 epochs
    'batch_size': 64,       # Batch size 64
    'n_states': 5,          # 5 state classifications
    'state_names': ['VeryLow', 'Low', 'Normal', 'High', 'VeryHigh'],
    'min_support': 0.03,    # Lower support threshold
}

pipeline = TrainPipeline(
    config=custom_config,
    output_dir='./outputs_custom'
)
```

---

## Troubleshooting

### Problem 1: CUDA Warning
```
UserWarning: CUDA not available
```
**Solution**: This is normal; the code will automatically fall back to CPU. If you have a GPU, install `tensorflow-gpu`.

### Problem 2: Module Import Error
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Ensure you have added the following before running the code:
```python
import sys
sys.path.append('/home/severin/Codelib/YS')
```

### Problem 3: Data Shape Error
```
ValueError: Input data must have at least 20 rows
```
**Solution**: Data must have at least `sequence_length` rows (default is 20).

### Problem 4: Model File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: './outputs/models/predictor.h5'
```
**Solution**: Run the training pipeline first to generate model files.

---

## Performance Optimization Suggestions

1. **GPU Acceleration**: Installing `tensorflow-gpu` can speed up training by 10-100x.
2. **Batch Size**: Increasing `batch_size` can improve training speed (requires more memory).
3. **Sequence Length**: Reducing `sequence_length` can decrease computational load.
4. **Parallel Processing**: Use `joblib` to process multiple samples in parallel.

---

## Next Steps

1. Read [IMPLEMENTATION.md](../IMPLEMENTATION.md) for algorithm details.
2. Check [DESIGN.md](../DESIGN.md) for system architecture.
3. Read the paper PDF for theoretical background.
4. Modify code to adapt to your own dataset.

---

**More Help**: Check docstrings of each module (`help(module)`) or source code comments.

**Last Updated**: 2026-01-16
