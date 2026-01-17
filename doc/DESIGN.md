# Causal Explainable AI System for Energy Demand Prediction - Project Design Document

## 1. Project Overview

### 1.1 Project Goals
Reproduce the paper "Causally explainable artificial intelligence on deep learning model for energy demand prediction" (Erlangga & Cho, 2025). The goal is to build an AI system that not only accurately predicts energy demand but also provides causal explanations and actionable recommendations.

**Source**: Engineering Applications of Artificial Intelligence, Vol. 162, 2025

### 1.2 Core Features
- **Prediction Capability**: High-precision time-series prediction based on a parallel CNN-LSTM-Attention architecture.
  - Improvements over serial CNN-LSTM: **34.84%** average improvement on the UCI dataset, **13.63%** average improvement on the REFIT dataset.
- **Causal Explanation**: Provides causal-level explanations via Bayesian Networks (rather than simple feature importance).
  - Explanation consistency (Cosine Similarity): Peak=0.99940, Normal=0.99983, Lower=0.99974.
  - Significantly outperforms SHAP (0.95-0.96), LIME (0.70-0.75), and PD Variance (0.81-0.96).
- **Actionable Recommendations**: Outputs action suggestions that users can directly adopt.
  - Quantifies intervention effects based on sensitivity analysis.

### 1.3 Application Value
- Household power demand prediction.
- Peak warning and load management.
- Energy-saving suggestions and usage optimization.
- Transferable to other fields (medical diagnosis, risk assessment, etc.).

---

## 2. System Architecture Design

### 2.1 Overall Architecture (Dual Pipeline Design)

```
Input Data (Time Series)
    |
    ├─────────────────┬─────────────────┐
    |                 |                 |
    v                 v                 v
[Prediction        [Feature          [Temporal
 Pipeline]          Extraction]       Interpretation]
 CNN Branch         CAM               LSTM+Attention
    |                 |                 |
    └─────────────────┴─────────────────┘
                      |
                      v
            [Numerical Prediction ŷ + Explanation Features]
                      |
                      v
            ┌─────────┴─────────┐
            v                   v
    [State Determination    [Discretization]
         (Sn)]                  |
            |                   v
            v              [Symbol Set]
    [Peak/Normal/Lower]         |
            |                   |
            └─────────┬─────────┘
                      v
            ┌─────────┴─────────┐
            v                   v
    [Clustering Analysis]  [Association Rules]
    (CAM/ATT Typing)           (Apriori)
            |                   |
            └─────────┬─────────┘
                      v
        [Bayesian Network Construction]
         (Structure Learning + Domain Constraints)
                      |
                      v
            ┌─────────┴─────────┐
            v                   v
    [Causal Inference]    [Counterfactual Analysis]
            |                   |
            └─────────┬─────────┘
                      v
        [Recommendation Generation]
                      |
                      v
            [Final Output]
```

---

## 3. Detailed Module Design

### 3.1 Data Preprocessing Module

#### Function
Convert raw time-series data into a format usable by the model.

#### Input
```python
Raw Data:
- Timestamp
- Global_active_power
- Global_reactive_power
- Voltage
- Global_intensity
- Sub-metering (Kitchen, Laundry, ClimateControl, Other)
```

#### Processing Steps
1. **Data Cleaning**: Handle missing values and outliers.
2. **Temporal Feature Extraction**:
   - Date
   - Day of the week
   - Month
   - Season
   - Weekend (binary)
3. **Sliding Window Construction**:
   - Window size ω = 80 time points
   - Prediction step l = 1
4. **Data Normalization**: Normalize continuous variables.

#### Output
```python
Training Samples:
- X: [Number of samples, Window length=80, Number of features]
- y: [Number of samples, 1]  # True electricity value for the next time step
```

---

### 3.2 Prediction Model Module (Core Deep Learning Part)

#### 3.2.1 Parallel CNN Branch

**Goal**: Extract short-term local temporal patterns and cross-feature combinations.

**Structure**:
```python
Input: [batch, window_length, num_features]
  ↓
1D Conv Layer 1: Conv1D(filters=64, kernel_size=3, activation='relu')
  ↓
Max Pooling 1: MaxPooling1D(pool_size=2)
  ↓
1D Conv Layer 2: Conv1D(filters=128, kernel_size=3, activation='relu')
  ↓
Max Pooling 2: MaxPooling1D(pool_size=2)
  ↓
Flatten Layer: Flatten()
  ↓
Output: CNN Feature Vector + CAM (Class Activation Map)
```

**Key Points**:
- CAM is obtained through a Global Average Pooling layer for subsequent explanation.
- Focuses on "which appliances changed in combination during which time period."

#### 3.2.2 LSTM + Attention Branch

**Goal**: Model long-term dependencies and identify key time points.

**Structure**:
```python
Input: [batch, window_length, num_features]
  ↓
LSTM Layer: LSTM(units=128, return_sequences=True)
  # Output hidden states for all time steps: h₁, h₂, ..., hₙ
  # As well as cell state s₁, ..., sₙ and output o₁, ..., oₙ
  ↓
Attention Layer (Based on Paper Eq. 1):
  1. Calculate Attention Scores: score_n = fc(o_n, h_N)
  2. Softmax Normalization: a_n = exp(fc(o_n, h_N)) / Σ_{k=1}^N exp(fc(o_k, h_N))
  3. Weighted Sum (Eq. 2): c_N = Σ_{k=1}^N a_k · h_k
  ↓
Output: Context Vector c_N + Attention Weight Vector a
```

**Mathematical Formulas (from Paper, Page 4)**:
$$a_n = \frac{\exp(f_c(o_n, h_N))}{\sum_{k=1}^{N} f_c(o_k, h_N)} \quad \text{for } t = 1, \ldots, T$$

$$c_N = \sum_{k=1}^{N} a_k \cdot h_k$$

**Key Points**:
- Attention weight *a* is used for subsequent explanation (Early/Late/Other).
- Identifies "which time period is most critical."
- N = Window length (default 80).

#### 3.2.3 Fusion and Regression

**Structure** (Based on Paper Eq. 3):
```python
# Fusion Formula
X_combined = concat(flattened(φ^l_Conv), c_N)

CNN Feature concat Context Vector c_N
  ↓
Fully Connected Layer 1: Dense(256, activation='relu')
  ↓
Dropout(0.3)
  ↓
Fully Connected Layer 2: Dense(128, activation='relu')
  ↓
Output Layer: Dense(1, activation='linear')
  ↓
Predicted Value ŷ (Continuous numeric)
```

**Mathematical Representation**:
$$X_{combined} = \text{concat}(\text{flattened}(\phi^l_{Conv}), c_N)$$

Where:
- $\phi^l_{Conv}$: Output of the last CNN layer (after flattening).
- $c_N$: Context vector from LSTM+Attention.

**Loss Function**: Mean Squared Error (MSE).

**Optimizer**: Adam (Recommended learning rate range: 0.0001-0.001).

**Training Details (from Paper)**:
- Window size ω = 80 time points.
- Prediction step l = 1 (t+1 single-step prediction).
- Batch size: Typically 32-64.
- Early Stopping: Stop if validation loss does not improve for 15 epochs.

---

### 3.3 State Determination Module (Sn Robust Estimation)

#### Function
Map continuous predicted values to three discrete categories.

#### Algorithm: Sn Scale Estimator

```python
def sn_scale_estimator(data):
    """
    Sn = c × median_i{ median_j(|x_i - x_j|) }
    where c is a correction factor.
    
    More robust than standard deviation, insensitive to outliers.
    """
    n = len(data)
    pairwise_diffs = []
    for i in range(n):
        diffs = [abs(data[i] - data[j]) for j in range(n)]
        pairwise_diffs.append(np.median(diffs))
    
    sn = 1.1926 * np.median(pairwise_diffs)  # 1.1926 is the correction factor for normal distribution
    return sn

def classify_state(y_pred, window_data):
    """
    Input: 
      - y_pred: Predicted value
      - window_data: Historical window data
    
    Output:
      - state ∈ {Peak, Normal, Lower}
    """
    median_val = np.median(window_data)
    sn = sn_scale_estimator(window_data)
    
    # Calculate robust Z-score
    z_score = (y_pred - median_val) / sn
    
    # Threshold determination (can be adjusted based on data)
    if z_score > 2.0:
        return "Peak"
    elif z_score < -2.0:
        return "Lower"
    else:
        return "Normal"
```

#### Output
```python
Energy Demand Pattern (EDP) ∈ {Peak, Normal, Lower}
```

---

### 3.4 Discretization and Symbolization Module

#### 3.4.1 Quantile Discretization

**Function**: Map continuous variables to finite levels.

**Method**:
```python
def quantile_discretize(data, n_bins=4):
    """
    Input: Continuous variable sequence
    Output: Discrete levels {Low, Medium, High, VeryHigh}
    
    Split based on quantiles:
    - [0%, 25%): Low
    - [25%, 50%): Medium  
    - [50%, 75%): High
    - [75%, 100%]: VeryHigh
    """
    quantiles = [0, 0.25, 0.50, 0.75, 1.0]
    bins = np.quantile(data, quantiles)
    labels = ['Low', 'Medium', 'High', 'VeryHigh']
    
    return pd.cut(data, bins=bins, labels=labels, include_lowest=True)
```

**Applied Variables**:
- Global Active Power
- Global Reactive Power
- Voltage
- Intensity
- Sub-metering variables

#### 3.4.2 Explanation Parameter Clustering

**Goal**: Typeify CNN's CAM and Attention vectors.

**CAM Clustering**:
```python
# Perform K-means clustering on CAM for all samples
# K value determined by elbow method (K=2 used in paper)

Clustering results example (from Paper Table 6):
- CAM_type1: Laundry, ClimateControl, and other equipment show an upward trend during observation.
- CAM_type2: Laundry, Kitchen, and ClimateControl equipment show a downward trend during observation.
```

**Attention Clustering (Cumulative Clustering Method)**:
```python
# Preprocessing steps (Paper Eq. 15)
# 1. De-minning: Normalize (subtract minimum)
# 2. Normalize by dividing by the sum
# 3. Calculate cumulative sum: C_A = Σ_{i=1}^N a_i

Clustering results example (from Paper Table 6 and Figure 7):
- Late Attention: Overall energy consumption shows an upward trend (higher weight towards the end).
- Early Attention: Overall energy consumption shows a downward trend (higher weight towards the beginning).
- Other Attention: Overall energy consumption is stable or fluctuating.

Decision criteria (based on cumulative Attention):
- Early: Cumulative attention reaches 70% or more within the first 50% of time -> weight concentrated early.
- Late: Cumulative attention reaches 30% or less within the first 50% of time -> weight concentrated late.
- Other: All other cases.
```

**Cumulative Clustering Formula (Paper Eq. 15)**:
$$C_A = \sum_{i=1}^{N} a_i$$

**Explanation Mapping Table (Paper Table 6)**:

| DLP | Meaning / Explanation |
|-----|----------|
| CAM 1 | Laundry, ClimateControl, and other equipment increased in the observation window |
| CAM 2 | Laundry, Kitchen, and ClimateControl equipment decreased in the observation window |
| Late Attention | Overall energy consumption showed an upward trend in the observation window |
| Early Attention | Overall energy consumption showed a downward trend in the observation window |
| Other Attention | Overall energy consumption was stable or fluctuating in the observation window |

---

### 3.5 Association Rule Mining Module (Apriori)

#### Function
Extract candidate causal relationships from discrete data.

#### Algorithm
```python
from mlxtend.frequent_patterns import apriori, association_rules

def mine_association_rules(discrete_data, min_support=0.1, min_confidence=0.6):
    """
    Input: Discretized dataframe (including EDP status)
    Output: Set of association rules
    
    Example Rule:
    {ClimateControl=VeryHigh, GAP=VeryHigh} → Peak
    Confidence: 0.85
    Lift: 2.3
    """
    # 1. Frequent itemset mining
    frequent_itemsets = apriori(discrete_data, 
                                 min_support=min_support,
                                 use_colnames=True)
    
    # 2. Rule generation
    rules = association_rules(frequent_itemsets, 
                               metric="confidence",
                               min_threshold=min_confidence)
    
    # 3. Filtering: Keep only rules where the consequent is EDP
    rules = rules[rules['consequents'].apply(
        lambda x: 'EDP' in str(x)
    )]
    
    return rules
```

#### Output
```python
List of Rules:
[
  {
    antecedents: ['ClimateControl_VeryHigh', 'Season_Summer'],
    consequent: 'Peak',
    confidence: 0.87,
    lift: 2.5
  },
  ...
]
```

---

### 3.6 Bayesian Network Module

#### 3.6.1 Structure Learning (with Domain Constraints)

**Node Grouping (Thematic Constraints)**:

```python
Node Theme Groups:
1. Physical Environment:
   - Date, Day, Month, Season, Weekend
   
2. Appliance Usage:
   - Kitchen, Laundry, ClimateControl, Other
   
3. Electricity Consumption:
   - GlobalActivePower, GlobalReactivePower, Voltage, Intensity
   
4. Deep Learning Parameters (DLPs):
   - CAM_type, ATT_type
   
5. Energy Demand Pattern:
   - EDP (Peak/Normal/Lower)
```

**Directional Constraints (Domain Knowledge)**:

```python
Allowed Causal Directions:
  Physical Environment → Appliance Usage → Electricity Consumption → Energy Demand Pattern
                    ↘
                     Deep Learning Parameters → Energy Demand Pattern

Forbidden Directions (Anti-commonsense):
  ❌ Energy Demand Pattern → Any upstream node
  ❌ Appliance Usage → Physical Environment
  ❌ Electricity Consumption → Appliance Usage (unless physical dependency exists)
```

**Structure Learning Algorithm**:

```python
from pgmpy.estimators import HillClimbSearch, BicScore

def learn_bn_structure(data, domain_constraints):
    """
    Input: Discrete data + Domain constraints
    Output: Bayesian Network Structure (DAG)
    
    Method: Hill-Climbing + BIC score + White/Black list constraints
    """
    # 1. Build white and black lists based on thematic constraints
    white_list = []  # Allowed edges
    black_list = []  # Forbidden edges
    
    # Example: Physical environment can only point to appliance usage
    for env_node in physical_env_nodes:
        for app_node in appliance_nodes:
            white_list.append((env_node, app_node))
        for edp_node in ['EDP']:
            black_list.append((edp_node, env_node))
    
    # ... other constraints
    
    # 2. Structure search
    hc = HillClimbSearch(data)
    best_model = hc.estimate(
        scoring_method=BicScore(data),
        white_list=white_list,
        black_list=black_list
    )
    
    return best_model
```

#### 3.6.2 Parameter Learning

**Method**: Maximum Likelihood Estimation (MLE)

```python
from pgmpy.estimators import MaximumLikelihoodEstimator

def learn_bn_parameters(structure, data):
    """
    Input: BN structure + Data
    Output: Conditional Probability Tables (CPT)
    
    Example CPT:
    P(EDP=Peak | ClimateControl=VeryHigh, Season=Summer) = 0.82
    """
    model = BayesianNetwork(structure.edges())
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    
    return model
```

#### 3.6.3 Three Conditionalized Networks

**Key Design**: Instead of one single network, use three distinct ones.

```python
# Learn BN separately for each EDP state
data_peak = data[data['EDP'] == 'Peak']
data_lower = data[data['EDP'] == 'Lower']
data_normal = data[data['EDP'] == 'Normal']

bn_peak = learn_bn(data_peak, constraints)
bn_lower = learn_bn(data_lower, constraints)
bn_normal = learn_bn(data_normal, constraints)
```

**Reason**: Causal mechanisms may differ completely under different energy states.

---

### 3.7 Causal Inference and Recommendation Generation Module

#### 3.7.1 Causal Inference

```python
def causal_inference(bn_model, evidence):
    """
    Input: BN model + Current observed evidence
    Output: Posterior probability distribution of EDP + Contribution of each factor
    
    Example:
    evidence = {
        'Season': 'Summer',
        'ClimateControl': 'VeryHigh',
        'ATT_type': 'Late'
    }
    
    Output:
    {
        'P(Peak)': 0.91,
        'P(Normal)': 0.07,
        'P(Lower)': 0.02,
        'factors': [
            ('ClimateControl', 0.43),  # Most critical
            ('ATT_type', 0.27),
            ('Season', 0.18)
        ]
    }
    """
    from pgmpy.inference import VariableElimination
    
    infer = VariableElimination(bn_model)
    result = infer.query(variables=['EDP'], evidence=evidence)
    
    # Sensitivity analysis (remove factors one by one to see the impact)
    sensitivity = compute_sensitivity(bn_model, evidence)
    
    return result, sensitivity
```

#### 3.7.2 Counterfactual Inference

```python
def counterfactual_analysis(bn_model, evidence, intervention):
    """
    Input: 
      - bn_model: Bayesian Network
      - evidence: Current observations
      - intervention: Planned intervention
    
    Output: Expected effect after intervention
    
    Example:
    evidence = {'ClimateControl': 'VeryHigh', ...}
    intervention = {'ClimateControl': 'Medium'}
    
    Output:
    {
        'original_P(Peak)': 0.91,
        'intervened_P(Peak)': 0.59,
        'reduction': 0.32  # 32% decrease
    }
    """
    # 1. Original inference
    original_prob = infer.query(
        variables=['EDP'], 
        evidence=evidence
    )
    
    # 2. Interventional inference (based on do-calculus principle)
    # Modify evidence and re-infer
    evidence_intervened = evidence.copy()
    evidence_intervened.update(intervention)
    
    intervened_prob = infer.query(
        variables=['EDP'],
        evidence=evidence_intervened
    )
    
    return {
        'original': original_prob,
        'intervened': intervened_prob,
        'effect': original_prob - intervened_prob
    }
```

#### 3.7.3 Actionable Recommendation Generation

```python
def generate_recommendations(causal_result, counterfactual_results):
    """
    Input: Causal inference results + Multiple counterfactual analysis results
    Output: Executable action recommendations (Natural Language)
    
    Strategy:
    1. Identify the primary cause (factor with the highest contribution).
    2. Perform counterfactual analysis for each controllable factor.
    3. Select the intervention with the best effect.
    4. Translate into user-readable recommendations.
    """
    recommendations = []
    
    # Example: Peak state
    if current_state == 'Peak':
        # Find the most critical and controllable factor
        top_factor = causal_result['factors'][0]  # e.g., ClimateControl
        
        # Calculate intervention effect
        effect = counterfactual_results[top_factor]['reduction']
        
        # Generate recommendation
        recommendation = f"""
        ⚠️ Peak Usage Warning (Probability {causal_result['P(Peak)']:.0%})
        
        Primary Causes:
        - {top_factor} is at a VeryHigh level.
        - Electricity usage has risen rapidly in the recent period.
        
        Recommended Actions:
        - Adjust {top_factor} from VeryHigh to Medium.
          → Peak risk is expected to drop by approximately {effect:.0%}.
        - Avoid turning on other high-power devices simultaneously for a short duration.
        """
        
        recommendations.append(recommendation)
    
    return recommendations
```

---

## 4. Data Flow Design

### 4.1 Training Phase Data Flow

```
Raw Data (CSV)
  ↓
[Data Preprocessing]
  - Cleaning, Temporal feature extraction
  - Sliding window construction (ω=80, l=1)
  - Dataset Description:
    * UCI: 2,075,259 records, 1-minute resolution, 8 attributes
    * REFIT: 5,733,526 records, 8-second resolution, 11 attributes
  ↓
Training Set / Validation Set / Test Set
  ↓
[Deep Learning Model Training]
  - CNN+LSTM+Attention
  - Output: Model weights + CAM + Attention
  - Training Parameters:
    * Window size ω = 80
    * Prediction step l = 1
    * Time resolution: 15min/30min/1h/1d (Four tested in paper)
  ↓
[Prediction + State Determination]
  - Predict on training/test sets
  - Sn state classification (Peak/Normal/Lower)
  ↓
[Discretization + Clustering]
  - Quantile discretization (4 levels)
  - CAM clustering (K=2)
  - Attention cumulative clustering (K=3)
  ↓
[Association Rule Mining]
  - Apriori algorithm
  - Min support/confidence settings
  ↓
[Bayesian Network Learning]
  - Structure learning (3 networks)
    * Tools: Genie Academic v4.1 (Paper) / pgmpy (this project)
    * Structure learning time: Approx. 433.61 seconds
  - Parameter learning (MLE)
  - Domain Knowledge (DK) Restriction
  ↓
Save all models and configurations
```

### 4.2 Inference Phase Data Flow

```
New Input Window (80 time points)
  ↓
[Preprocessing]
  - Normalization
  - Temporal feature extraction
  ↓
[Deep Model Prediction]
  - Output ŷ + CAM + Attention
  ↓
[State Determination]
  - Sn → EDP
  ↓
[Discretization]
  - Continuous values → Symbols
  ↓
[Clustering Mapping]
  - CAM → CAM_type
  - Attention → ATT_type
  ↓
[BN Selection]
  - Select bn_peak/lower/normal based on EDP
  ↓
[Causal Inference]
  - Calculate probabilities + Factor contributions
  ↓
[Counterfactual Analysis]
  - Test multiple intervention plans
  ↓
[Recommendation Generation]
  - Output actionable recommendations
  ↓
Final Output (JSON + Natural Language)
```

---

## 5. Technology Stack Selection

### 5.1 Programming Language and Frameworks

```python
Language: Python 3.8+

Deep Learning Framework:
- TensorFlow 2.x / Keras  # Prediction model
- PyTorch (Optional)        # If more flexible control is needed

Bayesian Networks:
- pgmpy                    # BN modeling and inference

Data Processing:
- pandas                   # Data manipulation
- numpy                    # Numerical computation
- scikit-learn             # Discretization, evaluation metrics

Association Rules:
- mlxtend                  # Apriori implementation

Clustering:
- scikit-learn.cluster     # K-means

Visualization:
- matplotlib
- seaborn
- networkx                 # BN visualization
```

### 5.2 Data Storage

```
Raw Data: CSV
Model Weights: .h5 / .pt
BN Structure: pickle / JSON
Configuration Files: YAML / JSON
```

---

## 6. Evaluation Metrics Design

### 6.1 Prediction Performance Metrics

```python
Regression Metrics (Prediction Accuracy):
- MSE (Mean Squared Error): MSE = (1/n)Σ(y_i - ŷ_i)²
- RMSE (Root Mean Squared Error): RMSE = √[(1/n)Σ(y_i - ŷ_i)²]
- MAE (Mean Absolute Error): MAE = Σ|y_i - ŷ_i| / n

Classification Metrics (State Determination):
- Accuracy
- Precision / Recall / F1 (specifically for Peak)
- Confusion Matrix

Paper Baseline Performance (Table 3, 15-min resolution, UCI dataset):
Method                  MSE      RMSE     MAE
ARIMA                0.1427   0.3778   0.2445
XGBoost              0.0836   0.2891   0.1752
Standalone CNN       0.0794   0.2818   0.1711
Standalone LSTM      0.0776   0.2786   0.1690
Proposed (Parallel)  Best     Best     Best

Performance Improvement (Table 4):
- Compared to serial CNN-LSTM-Attention:
  * UCI Dataset: 34.84% average improvement
  * REFIT Dataset: 13.63% average improvement
```

### 6.2 Explanation Consistency Metric

```python
def explanation_consistency(train_bn, test_bn):
    """
    Measures the similarity between the BN structure learned from the training set and the test set.
    
    Method: Structural Similarity (Cosine Similarity)
    
    Target: > 0.95
    
    Paper Results (Table 5):
    EDP State           PD Variance  LIME    SHAP    Proposed Method
    Lower than usual    0.80629      0.69221 0.95310 0.99974
    No significant      0.92643      0.75056 0.96091 0.99983
    Peak warning        0.96062      0.70791 0.95840 0.99940
    """
    train_edges = set(train_bn.edges())
    test_edges = set(test_bn.edges())
    
    intersection = len(train_edges & test_edges)
    union = len(train_edges | test_edges)
    
    similarity = intersection / union
    return similarity
```

### 6.3 Computational Efficiency Metrics

```python
Inference Time (Paper Tables 8 and 9):
- Deep Model Prediction Time: 
  * Serial CNN-LSTM: 186.065ms ± 2.773ms
  * Serial CNN-LSTM-Att: 236.639ms ± 14.878ms
  * Parallel CNN-LSTM-Att (Proposed): 260.857ms ± 9.831ms
  
- BN Inference Time: Average 1.30 seconds / sample
  
- Total Inference Time: < 1.5 seconds / sample (Fully acceptable for 1-hour ahead prediction)

Training / Learning Time:
- Deep Model Training: Can be performed offline
- BN Structure Learning: One-time, approx. 433.61 seconds (only during initialization)
- BN Parameter Learning: Maximum Likelihood Estimation, fast

Computational Overhead Comparison (Table 7, 100 samples):
Method              Peak State    Normal State  Lower State
Proposed (BN)       35.50s        77.97s        79.78s
LIME                4033.92s      3783.67s      3868.97s
SHAP                27310.59s     27456.40s     28035.31s
                    (~7.6 hours)  (~7.6 hours)  (~7.8 hours)

Key Advantages:
- BN structure learning is done once.
- Inference time is constant and does not grow with the number of samples.
- Suitable for real-time / online applications.
```

---

## 7. Scalability Design

### 7.1 Domain Transfer Design

The system is designed to be **highly transferable** to support other domains:

```python
# Configuration-driven domain adaptation
domain_config = {
    "domain": "energy",  # or "medical", "finance", etc.
    
    "variables": {
        "physical_env": ["Season", "Day", ...],
        "appliances": ["Kitchen", "ClimateControl", ...],
        "consumption": ["GlobalActivePower", ...],
        "dlp": ["CAM_type", "ATT_type"],
        "target": "EDP"
    },
    
    "causal_constraints": {
        "allowed_directions": [
            ("physical_env", "appliances"),
            ("appliances", "consumption"),
            ...
        ],
        "forbidden_directions": [
            ("target", "*"),
            ...
        ]
    },
    
    "discretization": {
        "n_bins": 4,
        "labels": ["Low", "Medium", "High", "VeryHigh"]
    },
    
    "recommendation_templates": {
        "Peak": "Suggestion: Reduce {factor} from {from_level} to {to_level}",
        ...
    }
}
```

### 7.2 Model Plug-ins

```python
# Replaceable prediction models
predictor_registry = {
    "cnn_lstm_att": ParallelCNNLSTMAttention,
    "transformer": TransformerPredictor,
    "tcn": TemporalConvNet,
    ...
}

# Replaceable BN learning algorithms
bn_learner_registry = {
    "hill_climb": HillClimbLearner,
    "pc_algorithm": PCLearner,
    "ges": GESLearner,
    ...
}
```

---

## 8. Key Challenges and Solutions

### 8.1 Imbalanced Training Data

**Problem**: The number of 'Peak' state samples may be significantly lower than 'Normal' samples.

**Solution**:
- Over-sampling 'Peak' samples (using Techniques like SMOTE for time series).
- Using class weights (e.g., `class_weight` in the loss function).
- Stratified sampling to ensure consistent distribution in the validation set.

### 8.2 Computational Complexity of BN Structure Learning

**Problem**: The search space for structure learning grows exponentially with the number of nodes.

**Solution**:
- Strong domain constraints significantly reduce the search space.
- Using association rules to pre-filter candidate edges.
- Adopting heuristic search (e.g., Hill-Climbing instead of exhaustive search).

### 8.3 Readability of Explanations

**Problem**: Bayesian Networks output probabilities, which are difficult for end-users to interpret.

**Solution**:
- Template-based recommendation generation.
- Visualization of causal paths.
- Quantifying intervention effects (e.g., "32% decrease" instead of "Probability reduction of 0.32").

---

## 9. Project Milestones

### Phase 1: Foundation (1-2 weeks)
- Data preprocessing pipeline.
- Deep learning model framework development.
- Basic training workflow.

### Phase 2: Prediction Model Optimization (2-3 weeks)
- Full implementation of CNN+LSTM+Attention.
- Hyperparameter tuning.
- Performance reaching the levels reported in the paper.

### Phase 3: Causal Explanation Module (2-3 weeks)
- Discretization and clustering.
- Association rule mining.
- Bayesian Network learning.

### Phase 4: Inference and Recommendation (1-2 weeks)
- Causal inference implementation.
- Counterfactual analysis.
- Recommendation generation.

### Phase 5: Integration Testing (1 week)
- End-to-end testing.
- Performance optimization.
- Documentation finalization.

---

## 10. Expected Outputs

### 10.1 Model Output

```json
{
  "prediction": {
    "timestamp": "2024-01-16 20:01:00",
    "predicted_power": 4.25,
    "state": "Peak",
    "probability": {
      "Peak": 0.91,
      "Normal": 0.07,
      "Lower": 0.02
    }
  },
  
  "explanation": {
    "main_factors": [
      {
        "factor": "ClimateControl",
        "level": "VeryHigh",
        "contribution": 0.43
      },
      {
        "factor": "ATT_type",
        "value": "Late",
        "contribution": 0.27
      }
    ],
    
    "causal_path": [
      "Season:Summer → ClimateControl:VeryHigh",
      "ClimateControl:VeryHigh → GlobalActivePower:VeryHigh",
      "GlobalActivePower:VeryHigh → Peak"
    ]
  },
  
  "counterfactual": {
    "intervention": {
      "ClimateControl": "Medium"
    },
    "expected_effect": {
      "Peak_probability": 0.59,
      "reduction": 0.32
    }
  },
  
  "recommendation": {
    "priority": "high",
    "action": "Adjust Climate Control from VeryHigh to Medium",
    "expected_benefit": "Peak usage risk is expected to decrease by 32%",
    "additional_tips": [
      "Avoid using other high-power devices simultaneously for a short duration"
    ]
  }
}
```

### 10.2 Visualization Output

1. **Prediction Curve Plot**
   - Actual vs. Predicted values.
   - State labels (Peak/Normal/Lower).
   - Peak detection highlighting (Figure 5a in paper shows effective peak detection).

2. **Attention Heatmap** (Figure 7 in paper)
   - Distribution of attention over the time axis.
   - Visualization of three modes: Early/Late/Other.

3. **CAM Activation Plot** (Figure 6 in paper)
   - Which features are activated at which times.
   - Two types: Upward pattern (Type 1) vs. Downward pattern (Type 2).
   - Use cosine similarity to filter consistent patterns.

4. **Causal Network Diagram** (Figures 3 and 8 in paper)
   - DAG visualization of the three BNs (Peak/Normal/Lower).
   - Node coloring (grouped by theme):
     * Green: Physical Environment
     * Yellow: Appliance Usage
     * Blue: Electricity Consumption
     * Red: Energy Demand Pattern (EDP)
     * Gray: Deep Learning Parameters (DLPs)
   - Edge thickness: Based on conditional probability strength.
   - Highlight activation states.

5. **Sensitivity Analysis Plot** (Figure 9 in paper)
   - Tornado chart showing the influence of various factors.
   - Sensitivity of the three main variables.
   - Green bar: Peak probability after adjusting variables.
   - Example result:
     * Peak state: 'Climate Control: Very High' has the most significant impact.
     * Lower state: 'Other appliances: Very High' is a negative factor.

6. **Computational Overhead Comparison Chart** (Figure 10 in paper)
   - Time complexity curves for SHAP, LIME, and the proposed method.
   - SHAP shows exponential growth.
   - BN remains constant.

7. **Ablation Study Plots** (Figures 11-14 in paper)
   - Performance comparison across different model configurations.
   - Effect of DLP integration.
   - Impact of the number of discrete states.
   - Noise robustness testing.

---

## 11. Summary of Paper Experimental Results (for Validation)

### 11.1 Actionable Recommendation Examples (Paper Page 9)

Based on Bayesian Network inference and sensitivity analysis, the paper provides specific recommendations for three states:

#### Peak Warning
**Observations**:
- Probability of Early Attention: 53% (overall decline followed by a sudden jump).
- Global Active Power: Very High (48%).
- Climate Control: Very High (50%).
- Weekend contribution: Only 33%.

**Sensitivity Analysis**:
- Climate Control (Very High → Low): Peak probability decreases from 66% to 34% (drop of 32%).

**Actionable Recommendation**:
> "Given the high sensitivity of the peak warning state to climate control appliance usage, residents are advised to avoid setting these appliances to the 'Very High' level, as it may lead to excessive power consumption."

#### Lower than Usual
**Observations**:
- Probability of CAM Type 2: 85% (reduction in Kitchen, Laundry, and Climate Control).
- Global Intensity: Low (42%).
- Global Active Power: Reduced (39%).

**Negative Factors**:
- Weekday (relative to Weekend).
- Other appliances: Very High.

**Actionable Recommendation**:
> "Residents should avoid using 'other appliances' at a 'Very High' level and maintain them at 'Medium' intensity. Additionally, reducing the use of kitchen, laundry, and climate control appliances can achieve significant energy savings."

#### No Significant Change
**Observations**:
- Global Active Power and Intensity maintain a stable distribution.
- Low/Medium/High states are evenly distributed.
- 'Other' attention type (stable/fluctuating).
- CAM 2 contribution (downward trend in appliance usage).

**Actionable Recommendation**:
> "Avoiding sudden jumps in energy consumption when using household appliances will maintain stability. Similar to the 'Lower' state scenario, avoiding increases in the use of kitchen, laundry, and climate control appliances also helps maintain this stability."

### 11.2 Ablation Study Conclusions (Paper Figs 11-14, Table 10)

**Model Component Contributions**:
- Full model (p-c-l-a): Best performance.
- Removing attention (p-c-l): Slight performance drop (marginal contribution).
- Serial configuration (s-c-l-a): Significant performance drop.
- **Conclusion**: Parallel configuration is the primary source of gain.

**DLP Integration Effect** (Fig 12):
- Including DLP: Significant increase in explanation consistency.
- Without DLP: Consistency drops.
- **Conclusion**: DLPs make explanations more stable.

**Impact of Discretization Levels** (Fig 13):
- Increasing levels: Slight improvement in BN accuracy.
- But interpretability drops (too granular).
- **Conclusion**: 4 levels represent the balance point between accuracy and readability.

**Noise Robustness** (Fig 14):
- Serial models are more resistant to noise.
- Adding attention improves noise robustness.
- **Conclusion**: Attention helps mitigate the impact of input noise.

### 11.3 Key Numerical Benchmarks

Used to verify if the reproduction is successful:

| Metric | Target Value | Source |
|------|--------|------|
| Prediction Improvement (UCI) | 34.84% | Table 4 |
| Prediction Improvement (REFIT) | 13.63% | Table 4 |
| Peak Consistency | > 0.999 | Table 5 |
| Normal Consistency | > 0.999 | Table 5 |
| Lower Consistency | > 0.999 | Table 5 |
| BN Inference Time | ~1.3 seconds | Table 8 |
| BN Structure Learning Time | ~434 seconds | Table 8 |
| Deep Model Inference Time | ~261ms | Table 9 |

---

## 12. Documentation Deliverables Checklist

1. **Project Design Document** (This document)
2. **Implementation Document** (Detailed code description)
3. **API Documentation** (Function interface description)
4. **User Manual** (How to use the system)
5. **Experimental Report** (Performance evaluation results)

---

## 13. Success Criteria

### 13.1 Prediction Performance
- **Baseline Comparison**: At 15-minute resolution, RMSE/MAE should be better than:
  - ARIMA: RMSE=0.3778, MAE=0.2445
  - XGBoost: RMSE=0.2891, MAE=0.1752
  - Standalone CNN: RMSE=0.2818
  - Standalone LSTM: RMSE=0.2786
  
- **Relative Improvement**: Compared to serial CNN-LSTM-Attention:
  - UCI Dataset: Improvement ≥ 30% (Paper value: 34.84%)
  - REFIT Dataset: Improvement ≥ 10% (Paper value: 13.63%)
  
- **Multi-resolution Stability**: Maintain superiority across four resolutions: 15min/30min/1h/1d.

### 13.2 Explanation Quality
- **Consistency (Cosine Similarity)**:
  - Peak State: > 0.995 (Paper value: 0.99940)
  - Normal State: > 0.995 (Paper value: 0.99983)
  - Lower State: > 0.995 (Paper value: 0.99974)
  
- **Outperforming Baseline Methods**:
  - SHAP Consistency: 0.95-0.96
  - LIME Consistency: 0.70-0.75
  - PD Variance Consistency: 0.81-0.96
  - **Proposed method should significantly outperform all the above methods.**

- **Verifiability of Generated Recommendations**:
  - Sensitivity analysis results are consistent with domain knowledge.
  - Results are verifiable and acceptable by energy management experts.

### 13.3 Computational Efficiency
- **Inference Performance**:
  - Deep Model: < 300ms / sample (Paper value: 260.857ms)
  - BN Inference: < 2 seconds / sample (Paper value: 1.30s)
  - Total Time: < 2 seconds / sample (Suitable for hourly predictions)
  
- **Learning Time**:
  - BN Structure Learning: < 500 seconds (Paper value: 433.61s)
  - Acceptable one-time overhead.

- **Scalability**:
  - Compared to SHAP/LIME:
    * 100-sample inference: < 100s vs. SHAP 7+ hours
    * Does not grow exponentially with the number of samples.

### 13.4 Usability
- **Code Quality**:
  - Modular design, easy to understand and modify.
  - Comprehensive documentation and comments.
  - Verified through unit tests.

- **Configuration Flexibility**:
  - YAML configuration file-driven.
  - Supports rapid switching between different datasets and parameters.
  - Customizable domain knowledge constraints.

- **Reproducibility**:
  - Fixed random seeds.
  - Traceable data processing workflows.
  - Repeatable experimental results.

### 13.5 Ablation Study Verification
- **Parallel vs. Serial**: Parallel configuration performs significantly better than serial.
- **Attention Contribution**: Removing attention leads to a drop in performance.
- **DLP Integration**: Adding CAM/Attention significantly improves explanation consistency.
- **Noise Robustness**: Attention helps resist input noise.

---

**Design Document Version**: v2.0 (Updated based on original PDF)
**Last Updated**: 2026-01-16  
**Updates**: 
- Added mathematical formulas from the paper.
- Included baseline numerical values for experimental results.
- Supplemented DLP explanation mapping table.
- Added actionable recommendation examples.
- Updated computational overhead comparison data.
- Added ablation study conclusions.
