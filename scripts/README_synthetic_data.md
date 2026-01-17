# Synthetic Data Generation Tool

A flexible synthetic energy data generator designed to support the generation of training datasets and various testing scenarios.

## Key Features

* ✅ **Training Data Generation**: Generates diverse normal energy data for model training.
* ✅ **Scenario Data Generation**: Generates test data for specific scenarios (high temp, low temp, extreme weather, etc.).
* ✅ **Batch Generation**: Generate multiple predefined scenarios in one go.
* ✅ **Custom Parameters**: Supports fully customizable parameters for bespoke scenarios.
* ✅ **Realistic Simulation**: Energy consumption models that consider seasonality, day/night cycles, time of day, and other environmental factors.

## Installation

No additional installation is required. Simply use the project environment:

```bash
source .venv/bin/activate  # Activate the virtual environment

```

## Usage

### 1. Generating Training Data

Generate 2,000 samples for model training:

```bash
python scripts/generate_synthetic_data.py \
    --mode training \
    --n-samples 2000 \
    --output data/synthetic

```

**Parameter Descriptions**:

* `--n-samples`: Number of samples (default: 2000).
* `--start-date`: Start date (default: 2024-01-01).
* `--output`: Output directory.

**Output File**: `data/synthetic/training_data.csv`

---

### 2. Generating a Single Scenario

Generate test data for a specific scenario:

```bash
# High Temperature & High Humidity Scenario
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type high_temp_humid \
    --duration 30 \
    --output data/synthetic

# Heatwave Scenario (Extreme Heat)
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type heatwave \
    --duration 48 \
    --output data/synthetic

```

**Predefined Scenario Types**:

* `high_temp_humid`: Summer afternoon (32°C, 75% humidity).
* `low_temp_humid`: Winter early morning (12°C, 40% humidity).
* `moderate`: Spring/Autumn (20°C, 55% humidity).
* `peak_hour`: Evening peak hour (28°C).
* `valley_hour`: Late night (18°C).
* `heatwave`: Extreme heat (38°C, 80% humidity).
* `coldwave`: Extreme cold (5°C, 35% humidity).

**Parameter Descriptions**:

* `--scenario-type`: Type of scenario.
* `--duration`: Duration in hours (default: 30).
* `--start-hour`: Starting hour (0-23, default: 0).

**Output File**: `data/synthetic/scenario_{type}.csv`

---

### 3. Batch Generate All Predefined Scenarios

Generate all 7 predefined scenarios at once:

```bash
python scripts/generate_synthetic_data.py \
    --mode batch \
    --output data/synthetic

```

**Generated Scenarios**:

* `high_temp_humid.csv` (30h)
* `low_temp_humid.csv` (30h)
* `moderate.csv` (30h)
* `peak_hour.csv` (24h)
* `valley_hour.csv` (24h)
* `heatwave.csv` (48h)
* `coldwave.csv` (48h)

---

### 4. Custom Scenarios

Fully customize scenario parameters:

```bash
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type custom \
    --temp-base 25 \
    --humid-base 60 \
    --wind-base 4 \
    --duration 24 \
    --output data/synthetic

```

**Parameter Descriptions**:

* `--temp-base`: Baseline Temperature (°C).
* `--humid-base`: Baseline Humidity (%).
* `--wind-base`: Baseline Wind Speed (m/s).

**Output File**: `data/synthetic/scenario_custom.csv`

---

## Data Format

The generated CSV files contain the following columns:

| Column Name | Description | Unit | Range |
| --- | --- | --- | --- |
| Temperature | Ambient Temperature | °C | 0 - 50 |
| Humidity | Relative Humidity | % | 20 - 95 |
| WindSpeed | Wind Speed | m/s | 0 - 25 |
| EDP | Energy Demand (Placeholder) | kWh | - |
| Hour | Hour of Day | - | 0 - 23 |
| DayOfWeek | Day of the Week | - | 0 - 6 |
| Month | Month | - | 1 - 12 |

**Note**: The `EDP` column in scenario data is a placeholder (0.0) and requires model prediction to obtain realistic values.

---

## Examples

### Example 1: Prepare Training Data

```bash
# Generate 3000 samples for training starting from 2023
python scripts/generate_synthetic_data.py \
    --mode training \
    --n-samples 3000 \
    --start-date 2023-01-01 \
    --output data/synthetic \
    --seed 123

# Output:
# INFO: Training data statistics:
# INFO:   Temperature: 26.15 ± 5.07
# INFO:   Humidity: 65.02 ± 5.94
# INFO:   WindSpeed: 9.96 ± 4.69
# INFO:   EDP: 120.12 ± 20.88
# INFO: ✅ Training data saved: data/synthetic/training_data.csv

```

### Example 2: Prepare Inference Test Scenarios

```bash
# Option A: Use batch mode to generate all scenarios
python scripts/generate_synthetic_data.py --mode batch --output data/test_scenarios

# Option B: Generate a specific scenario of interest
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type heatwave \
    --duration 72 \
    --output data/test_scenarios

```

### Example 3: Use in Python Code

```python
from scripts.generate_synthetic_data import EnergyDataGenerator

# Initialize the generator
generator = EnergyDataGenerator(seed=42)

# Generate training data
train_data = generator.generate_training_data(n_samples=2000)

# Generate a test scenario
test_scenario = generator.generate_scenario(
    scenario_type='heatwave',
    duration=48
)

# Batch generate multiple scenarios
scenarios = generator.generate_multiple_scenarios([
    {'name': 'scene1', 'type': 'high_temp_humid', 'duration': 30},
    {'name': 'scene2', 'type': 'low_temp_humid', 'duration': 30}
])

```

---

## Updating Existing Scripts

### Update Training Script

Modify the data loading logic in `scripts/run_training.py`:

```python
# Old way
data_path = 'data/processed/synthetic_energy_data.csv'

# New way
data_path = 'data/synthetic/training_data.csv'

```

### Update Inference Script

Modify `scripts/run_inference.py` to load scenarios from files:

```python
import pandas as pd

# Load pre-generated scenarios
scenarios = [
    ('High Temp Humid', pd.read_csv('data/synthetic/high_temp_humid.csv')),
    ('Low Temp Humid', pd.read_csv('data/synthetic/low_temp_humid.csv')),
    ('Heatwave', pd.read_csv('data/synthetic/heatwave.csv'))
]

```

---

## Data Quality & Characteristics

The generated data exhibits the following properties:

1. **Temporal Correlation**: Accounts for diurnal, weekly, and seasonal variations.
2. **Feature Correlation**: Inverse relationship between temperature and humidity; energy consumption influenced by wind speed.
3. **Realistic Distribution**: Energy consumption models based on physical laws.
4. **Reproducibility**: Results are reproducible via the use of random seeds.

