Here is the verbatim English translation of your implementation document.

---

# Causal Explainable AI System for Energy Demand Forecasting - Implementation Document

## 1. Project Structure

```
energy-causal-ai/
├── README.md                    # Project Description
├── requirements.txt             # Dependency Packages
├── config/
│   ├── config.yaml              # Main Configuration
│   ├── domain_config.yaml       # Domain Knowledge Configuration
│   └── model_params.yaml        # Model Hyperparameters
├── data/
│   ├── raw/                     # Raw Data
│   ├── processed/               # Processed Data
│   └── interim/                 # Intermediate Data
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Data Loading
│   │   ├── preprocessor.py      # Data Preprocessing
│   │   └── discretizer.py       # Discretization Module
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictor.py         # Prediction Model (CNN+LSTM+Att)
│   │   ├── state_classifier.py  # State Classification (Sn)
│   │   ├── clustering.py        # CAM/Attention Clustering
│   │   ├── association.py       # Association Rule Mining
│   │   └── bayesian_net.py      # Bayesian Network
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── causal_inference.py  # Causal Inference
│   │   └── recommendation.py    # Recommendation Generation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation Metrics
│   │   ├── visualization.py     # Visualization
│   │   └── logger.py            # Logging
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py    # Training Pipeline
│       └── infer_pipeline.py    # Inference Pipeline
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_causal_analysis.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_inference.py
├── models/                      # Saved Models
│   ├── predictor.h5
│   ├── bn_peak.pkl
│   ├── bn_normal.pkl
│   └── bn_lower.pkl
├── outputs/                     # Output Results
│   ├── predictions/
│   ├── explanations/
│   └── visualizations/
└── docs/                        # Documentation
    ├── Project_Design_Doc.md
    ├── Implementation_Doc.md (This Document)
    └── API_Doc.md

```

---

## 2. Detailed Implementation of Core Modules

### 2.1 Data Preprocessing Module

#### File: `src/data/preprocessor.py`

```python
"""
Data Preprocessing Module
Functions:
1. Data Cleaning
2. Time Feature Extraction
3. Sliding Window Construction
4. Normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class EnergyDataPreprocessor:
    """Energy Data Preprocessor"""
    
    def __init__(self, config: dict):
        """
        Parameters:
            config: Configuration dictionary containing:
                - window_size: Window size (default 80)
                - pred_horizon: Prediction horizon (default 1)
                - time_resolution: Time resolution ('15min', '30min', '1h', '1d')
                - feature_cols: List of feature column names
        """
        self.window_size = config.get('window_size', 80)
        self.pred_horizon = config.get('pred_horizon', 1)
        self.time_resolution = config.get('time_resolution', '15min')
        self.feature_cols = config.get('feature_cols', [])
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data Cleaning
        
        Steps:
        1. Handle missing values (Forward Fill + Backward Fill)
        2. Remove outliers (Using IQR method)
        3. Ensure timestamp continuity
        
        Input:
            df: Raw dataframe, must contain a timestamp column
        
        Output:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        # 1. Ensure timestamp is datetime type
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # 2. Resample to specified time resolution (Ensure continuity)
        df = df.resample(self.time_resolution).mean()
        
        # 3. Handle missing values
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If missing values remain, fill with column mean
        df = df.fillna(df.mean())
        
        # 4. Remove outliers (Conservative strategy: only remove extreme outliers)
        for col in self.feature_cols:
            if col in df.columns:
                # Calculate IQR
                Q1 = df[col].quantile(0.01)  # Using 1% and 99% percentiles (more conservative)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                
                # Define outlier boundaries
                lower_bound = Q1 - 3 * IQR  # 3x IQR (more relaxed)
                upper_bound = Q3 + 3 * IQR
                
                # Clip instead of delete
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Data cleaning complete. Final data volume: {len(df)}")
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Time Features
        
        Features:
        - Date: Day of month (1-31)
        - Day: Day of week (0-6, 0 is Monday)
        - Month: Month (1-12)
        - Season: Season (0-3, 0=Spring, 1=Summer, 2=Autumn, 3=Winter)
        - Weekend: Is weekend (0 or 1)
        - Hour: Hour (if resolution < 1 day)
        
        Input:
            df: Dataframe with time index
        
        Output:
            Dataframe with added time features
        """
        logger.info("Extracting time features...")
        
        # Basic time features
        df['Date'] = df.index.day
        df['Day'] = df.index.dayofweek  # 0=Mon, 6=Sun
        df['Month'] = df.index.month
        
        # Season features (Northern Hemisphere)
        # Spring:3-5, Summer:6-8, Autumn:9-11, Winter:12-2
        df['Season'] = ((df['Month'] % 12 + 3) // 3) % 4
        
        # Weekend indicator
        df['Weekend'] = (df['Day'] >= 5).astype(int)  # 5=Sat, 6=Sun
        
        # If resolution is less than 1 day, add hour feature
        if self.time_resolution in ['15min', '30min', '1h']:
            df['Hour'] = df.index.hour
        
        logger.info(f"Time feature extraction complete. New features: Date, Day, Month, Season, Weekend")
        return df
    
    def create_sequences(self, 
                         df: pd.DataFrame,
                         target_col: str = 'Global_active_power'
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Sliding Window Sequences
        
        Input:
            df: Preprocessed dataframe
            target_col: Target variable column name
        
        Output:
            X: Input sequence with shape [samples, window_size, features]
            y: Target values with shape [samples,]
        
        Example:
            window_size=80, pred_horizon=1
            Sample 0: X[0] = df[0:80], y[0] = df[80][target_col]
            Sample 1: X[1] = df[1:81], y[1] = df[81][target_col]
        """
        logger.info("Creating sliding window sequences...")
        
        # Select feature columns
        feature_df = df[self.feature_cols].copy()
        target_series = df[target_col].values
        
        # Normalize features
        if not self.is_fitted:
            feature_values = self.scaler.fit_transform(feature_df)
            self.is_fitted = True
        else:
            feature_values = self.scaler.transform(feature_df)
        
        # Construct sequences
        X_list = []
        y_list = []
        
        for i in range(len(feature_values) - self.window_size - self.pred_horizon + 1):
            # Input window: [i, i+window_size)
            X_window = feature_values[i:i+self.window_size]
            
            # Target value: i+window_size+pred_horizon-1
            y_target = target_series[i + self.window_size + self.pred_horizon - 1]
            
            X_list.append(X_window)
            y_list.append(y_target)
        
        X = np.array(X_list)  # [samples, window, features]
        y = np.array(y_list)  # [samples,]
        
        logger.info(f"Sequence creation complete. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15
                   ) -> Tuple:
        """
        Split Train, Validation, and Test Sets
        
        Note: Time series cannot be shuffled; must be split chronologically.
        
        Input:
            X, y: Full dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio (Test set = 1 - train - val)
        
        Output:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n_samples = len(X)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        logger.info(f"Data split complete:")
        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Validation set: {len(X_val)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def preprocess_pipeline(self, 
                            df: pd.DataFrame,
                            target_col: str = 'Global_active_power'
                            ) -> Tuple:
        """
        Full Preprocessing Pipeline
        
        Steps:
        1. Data Cleaning
        2. Time Feature Extraction
        3. Sequence Creation
        4. Data Splitting
        
        Input:
            df: Raw dataframe
            target_col: Target column name
        
        Output:
            X and y for Train, Val, and Test sets
        """
        # 1. Clean
        df_clean = self.clean_data(df)
        
        # 2. Time Features
        df_features = self.extract_time_features(df_clean)
        
        # 3. Create Sequences
        X, y = self.create_sequences(df_features, target_col)
        
        # 4. Split
        return self.split_data(X, y)


# Usage Example
if __name__ == "__main__":
    # Config
    config = {
        'window_size': 80,
        'pred_horizon': 1,
        'time_resolution': '15min',
        'feature_cols': [
            'Global_active_power',
            'Global_reactive_power',
            'Voltage',
            'Global_intensity',
            'Sub_metering_1',  # Kitchen
            'Sub_metering_2',  # Laundry
            'Sub_metering_3',  # Climate Control
            # Time features added in extract_time_features
        ]
    }
    
    # Load Data
    df = pd.read_csv('data/raw/household_power_consumption.txt',
                     sep=';',
                     parse_dates={'timestamp': ['Date', 'Time']},
                     na_values=['?'])
    
    # Preprocess
    preprocessor = EnergyDataPreprocessor(config)
    X_train, y_train, X_val, y_val, X_test, y_test = \
        preprocessor.preprocess_pipeline(df)
    
    print(f"Preprocessing complete!")
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")

```

---

### 2.2 Prediction Model Implementation

#### File: `src/models/predictor.py`

```python
"""
Parallel CNN-LSTM-Attention Prediction Model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ParallelCNNLSTMAttention(Model):
    """
    Parallel CNN-LSTM-Attention Architecture
    
    Structure:
    - CNN Branch: Extracts local temporal patterns
    - LSTM+Attention Branch: Models long-term dependencies and locates key time steps
    - Fusion: Concatenation followed by MLP regression
    """
    
    def __init__(self, 
                 window_size: int = 80,
                 n_features: int = 10,
                 cnn_filters: list = [64, 128],
                 lstm_units: int = 128,
                 mlp_units: list = [256, 128],
                 dropout_rate: float = 0.3):
        """
        Parameters:
            window_size: Input window size
            n_features: Number of features
            cnn_filters: Filter counts for CNN layers
            lstm_units: Number of LSTM hidden units
            mlp_units: Neuron counts for MLP layers
            dropout_rate: Dropout ratio
        """
        super(ParallelCNNLSTMAttention, self).__init__()
        
        self.window_size = window_size
        self.n_features = n_features
        
        # ===== CNN Branch =====
        self.conv1 = layers.Conv1D(filters=cnn_filters[0],
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv1')
        self.pool1 = layers.MaxPooling1D(pool_size=2, name='pool1')
        
        self.conv2 = layers.Conv1D(filters=cnn_filters[1],
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same',
                                   name='conv2')
        self.pool2 = layers.MaxPooling1D(pool_size=2, name='pool2')
        
        # Global Average Pooling (used for CAM generation)
        self.global_avg_pool = layers.GlobalAveragePooling1D(name='gap')
        
        # ===== LSTM+Attention Branch =====
        self.lstm = layers.LSTM(units=lstm_units,
                                return_sequences=True,  # Return all time steps
                                name='lstm')
        
        # Attention Layer (Custom)
        self.attention = AttentionLayer(name='attention')
        
        # ===== Fusion and Regression =====
        self.flatten = layers.Flatten(name='flatten')
        
        self.fc1 = layers.Dense(mlp_units[0], activation='relu', name='fc1')
        self.dropout1 = layers.Dropout(dropout_rate, name='dropout1')
        
        self.fc2 = layers.Dense(mlp_units[1], activation='relu', name='fc2')
        self.dropout2 = layers.Dropout(dropout_rate, name='dropout2')
        
        self.output_layer = layers.Dense(1, activation='linear', name='output')
        
    def call(self, inputs, training=False, return_attention=False):
        """
        Forward Pass
        
        Input:
            inputs: [batch, window_size, n_features]
            training: Boolean training mode
            return_attention: Whether to return attention weights (for explanation)
        
        Output:
            If return_attention=False: Prediction [batch, 1]
            If return_attention=True: (Prediction, CAM, attention_weights)
        """
        # ===== CNN Branch =====
        x_cnn = self.conv1(inputs)  # [batch, window_size, 64]
        x_cnn = self.pool1(x_cnn)   # [batch, window_size/2, 64]
        
        x_cnn = self.conv2(x_cnn)   # [batch, window_size/4, 128]
        x_cnn_pooled = self.pool2(x_cnn)  # [batch, window_size/4, 128]
        
        # Generate CAM (at the last convolutional layer)
        # CAM = activation intensity across feature dimensions per time step
        cam = tf.reduce_mean(x_cnn, axis=-1)  # [batch, window_size/4]
        
        # Global Pooling
        cnn_features = self.global_avg_pool(x_cnn_pooled)  # [batch, 128]
        
        # ===== LSTM+Attention Branch =====
        lstm_out = self.lstm(inputs, training=training)  # [batch, window, 128]
        
        # Attention
        context, attention_weights = self.attention(lstm_out)  # context: [batch, 128]
        
        # ===== Fusion =====
        # Concatenate CNN features and LSTM context
        merged = tf.concat([cnn_features, context], axis=-1)  # [batch, 256]
        
        # MLP Regression
        x = self.fc1(merged)
        x = self.dropout1(x, training=training)
        
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        
        prediction = self.output_layer(x)  # [batch, 1]
        
        if return_attention:
            return prediction, cam, attention_weights
        else:
            return prediction
    
    def get_config(self):
        """Save model configuration (for serialization)"""
        return {
            'window_size': self.window_size,
            'n_features': self.n_features
        }


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer
    
    Formula:
    score_k = fc(h_k)
    a_k = exp(score_k) / Σexp(score_k)  # softmax
    context = Σ(a_k × h_k)
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        """
        Input shape: [batch, sequence_length, hidden_dim]
        """
        self.hidden_dim = input_shape[-1]
        
        # Fully connected layer for attention weights calculation
        self.W = self.add_weight(name='attention_weight',
                                 shape=(self.hidden_dim, 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Input:
            inputs: LSTM output [batch, sequence, hidden]
        
        Output:
            context: Context vector [batch, hidden]
            attention_weights: Attention weights [batch, sequence]
        """
        # Calculate attention scores
        # [batch, sequence, hidden] × [hidden, 1] = [batch, sequence, 1]
        score = tf.matmul(inputs, self.W) + self.b
        score = tf.squeeze(score, axis=-1)  # [batch, sequence]
        
        # Softmax normalization
        attention_weights = tf.nn.softmax(score, axis=-1)  # [batch, sequence]
        
        # Weighted sum
        # [batch, sequence, 1] × [batch, sequence, hidden] = [batch, sequence, hidden]
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
        context = tf.reduce_sum(inputs * attention_weights_expanded, axis=1)  # [batch, hidden]
        
        return context, attention_weights
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


# ===== Model Construction and Training =====
def build_model(config: dict) -> ParallelCNNLSTMAttention:
    """
    Build model based on config
    
    Config example:
    {
        'window_size': 80,
        'n_features': 10,
        'cnn_filters': [64, 128],
        'lstm_units': 128,
        'mlp_units': [256, 128],
        'dropout_rate': 0.3
    }
    """
    model = ParallelCNNLSTMAttention(**config)
    return model


def train_model(model: ParallelCNNLSTMAttention,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 100,
                batch_size: int = 64,
                learning_rate: float = 0.001) -> Dict:
    """
    Train Model
    
    Input:
        model: Model instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Output:
        Training history dictionary
    """
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='models/predictor_best.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history.history


# Usage Example
if __name__ == "__main__":
    # Model Config
    config = {
        'window_size': 80,
        'n_features': 10,
        'cnn_filters': [64, 128],
        'lstm_units': 128,
        'mlp_units': [256, 128],
        'dropout_rate': 0.3
    }
    
    # Build Model
    model = build_model(config)
    
    # Assuming data exists
    # X_train, y_train, X_val, y_val = ...
    
    # Train
    # history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Prediction (with explanation)
    # predictions, cam, attention = model(X_test, return_attention=True)
    
    print("Model construction complete!")
    model.summary()

```

---

### 2.3 State Classification Module (Sn Estimation)

#### File: `src/models/state_classifier.py`

```python
"""
State Classifier based on Sn Scale Estimation
"""

import numpy as np
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SnStateClassifier:
    """
    Sn Robust Scale Estimator
    Used to map continuous prediction values to 3 state classes
    """
    
    def __init__(self, 
                 peak_threshold: float = 2.0,
                 lower_threshold: float = -2.0):
        """
        Parameters:
            peak_threshold: Z-score threshold for Peak state
            lower_threshold: Z-score threshold for Lower state
        """
        self.peak_threshold = peak_threshold
        self.lower_threshold = lower_threshold
        
        # Correction factor (makes Sn unbiased under normal distribution)
        self.correction_factor = 1.1926
        
    def sn_scale(self, data: np.ndarray) -> float:
        """
        Calculate Sn Scale Estimate
        
        Sn = c × median_i{ median_j(|x_i - x_j|) }
        
        Advantages:
        - More robust than standard deviation (breakdown point = 50%)
        - Insensitive to outliers
        - Suitable for high-noise data
        
        Input:
            data: 1D array
        
        Output:
            Sn scale value
        """
        n = len(data)
        
        # Calculate medians of all pairwise differences
        pairwise_medians = []
        for i in range(n):
            # Absolute differences between i-th point and all points
            diffs = np.abs(data - data[i])
            pairwise_medians.append(np.median(diffs))
        
        # Take the median of all medians
        sn = self.correction_factor * np.median(pairwise_medians)
        
        return sn
    
    def robust_z_score(self, 
                       value: float,
                       reference_data: np.ndarray) -> float:
        """
        Calculate Robust Z-score
        
        Z = (value - median) / Sn
        
        Input:
            value: Value to evaluate (typically prediction)
            reference_data: Reference data (historical window)
        
        Output:
            Robust Z-score
        """
        median = np.median(reference_data)
        sn = self.sn_scale(reference_data)
        
        # Prevent division by zero
        if sn < 1e-8:
            return 0.0
        
        z_score = (value - median) / sn
        return z_score
    
    def classify(self, 
                 prediction: float,
                 historical_window: np.ndarray) -> str:
        """
        Classify single prediction value
        
        Input:
            prediction: Predicted power consumption for next time step
            historical_window: Historical window data (for reference distribution)
        
        Output:
            State label: 'Peak' / 'Normal' / 'Lower'
        """
        z = self.robust_z_score(prediction, historical_window)
        
        if z > self.peak_threshold:
            return 'Peak'
        elif z < self.lower_threshold:
            return 'Lower'
        else:
            return 'Normal'
    
    def classify_batch(self,
                       predictions: np.ndarray,
                       historical_windows: np.ndarray) -> np.ndarray:
        """
        Batch Classification
        
        Input:
            predictions: [n_samples,] Predicted values array
            historical_windows: [n_samples, window_size] Corresponding historical windows
        
        Output:
            [n_samples,] State labels array
        """
        states = []
        for pred, window in zip(predictions, historical_windows):
            state = self.classify(pred, window)
            states.append(state)
        
        return np.array(states)
    
    def get_state_statistics(self, states: np.ndarray) -> dict:
        """
        Statistically analyze state distribution
        
        Input:
            states: State labels array
        
        Output:
            Statistics dictionary
        """
        unique, counts = np.unique(states, return_counts=True)
        total = len(states)
        
        stats = {
            'total': total,
            'counts': dict(zip(unique, counts)),
            'ratios': {s: c/total for s, c in zip(unique, counts)}
        }
        
        logger.info("State Distribution:")
        for state in ['Peak', 'Normal', 'Lower']:
            if state in stats['counts']:
                count = stats['counts'][state]
                ratio = stats['ratios'][state]
                logger.info(f"  {state}: {count} ({ratio:.2%})")
        
        return stats


# Usage Example
if __name__ == "__main__":
    # Create Classifier
    classifier = SnStateClassifier(
        peak_threshold=2.0,
        lower_threshold=-2.0
    )
    
    # Mock Data
    np.random.seed(42)
    
    # Historical window (normal consumption)
    historical = np.random.normal(loc=3.0, scale=0.5, size=80)
    
    # Test cases
    test_cases = [
        ("Normal Prediction", 3.2, historical),
        ("Peak Prediction", 5.0, historical),
        ("Lower Prediction", 1.5, historical)
    ]
    
    for desc, pred, window in test_cases:
        state = classifier.classify(pred, window)
        z_score = classifier.robust_z_score(pred, window)
        print(f"{desc}: Prediction={pred:.2f}, Z-score={z_score:.2f}, State={state}")

```

---

**To be continued...**

Due to the length of the implementation document, I will continue adding other module implementations. Let's move to the next part.

---

### 2.4 Discretization Module

#### File: `src/data/discretizer.py`

```python
"""
Continuous Variable Discretization Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import pickle
import logging

logger = logging.getLogger(__name__)


class QuantileDiscretizer:
    """
    Quantile-based Discretizer
    
    Maps continuous variables to finite levels (Low, Medium, High, VeryHigh)
    """
    
    def __init__(self, n_bins: int = 4, labels: List[str] = None):
        """
        Parameters:
            n_bins: Number of discretization levels
            labels: List of labels
        """
        self.n_bins = n_bins
        self.labels = labels or ['Low', 'Medium', 'High', 'VeryHigh']
        
        assert len(self.labels) == n_bins, \
            f"Label count ({len(self.labels)}) must equal bin count ({n_bins})"
        
        # Save quantiles for each variable (used for transform)
        self.quantile_dict_ = {}
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, columns: List[str]):
        """
        Fit discretizer (calculate quantiles)
        
        Input:
            data: Dataframe
            columns: List of column names to discretize
        """
        logger.info(f"Starting discretizer fit, variables: {len(columns)}")
        
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        
        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' does not exist, skipping")
                continue
            
            # Calculate quantile bins
            bins = data[col].quantile(quantiles).values
            
            # Ensure boundaries are unique (prevents errors from duplicate boundaries)
            bins = np.unique(bins)
            if len(bins) < len(quantiles):
                logger.warning(f"Column '{col}' has fewer unique quantiles than expected; likely duplicate values")
            
            self.quantile_dict_[col] = bins
        
        self.is_fitted = True
        logger.info("Discretizer fit complete")
    
    def transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform data (Discretize)
        
        Input:
            data: Dataframe
            columns: Column names to discretize
        
        Output:
            Discretized dataframe (original columns replaced)
        """
        if not self.is_fitted:
            raise ValueError("Discretizer not fitted. Call fit() first.")
        
        data_discrete = data.copy()
        
        for col in columns:
            if col not in self.quantile_dict_:
                logger.warning(f"Column '{col}' was not seen during fit, skipping")
                continue
            
            bins = self.quantile_dict_[col]
            
            # Discretize using pd.cut
            data_discrete[col] = pd.cut(
                data[col],
                bins=bins,
                labels=self.labels[:len(bins)-1],  # labels count = bins count - 1
                include_lowest=True,
                duplicates='drop'
            )
        
        return data_discrete
    
    def fit_transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(data, columns)
        return self.transform(data, columns)
    
    def save(self, filepath: str):
        """Save discretizer"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.quantile_dict_, f)
        logger.info(f"Discretizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load discretizer"""
        with open(filepath, 'rb') as f:
            self.quantile_dict_ = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Discretizer loaded from {filepath}")


# Usage Example
if __name__ == "__main__":
    # Mock data
    np.random.seed(42)
    data = pd.DataFrame({
        'power': np.random.normal(3, 1, 1000),
        'voltage': np.random.normal(240, 5, 1000),
        'current': np.random.normal(10, 2, 1000)
    })
    
    # Discretization
    discretizer = QuantileDiscretizer(n_bins=4)
    
    columns_to_discretize = ['power', 'voltage', 'current']
    data_discrete = discretizer.fit_transform(data, columns_to_discretize)
    
    print("First 5 rows after discretization:")
    print(data_discrete.head())
    
    print("\nDistribution per variable:")
    for col in columns_to_discretize:
        print(f"\n{col}:")
        print(data_discrete[col].value_counts().sort_index())

```

---

### 2.5 CAM/Attention Clustering Module

#### File: `src/models/clustering.py`

```python
"""
CAM and Attention Vector Clustering Module
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import logging

logger = logging.getLogger(__name__)


class DLPClusterer:
    """
    Deep Learning Parameters (DLP) Clusterer
    
    Used for:
    1. CAM (Class Activation Map) Clustering
    2. Attention weight vector Clustering
    """
    
    def __init__(self, 
                 n_clusters: int = None,
                 max_clusters: int = 10,
                 random_state: int = 42):
        """
        Parameters:
            n_clusters: Number of clusters (None for automatic selection)
            max_clusters: Maximum clusters for automatic selection
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        
        self.kmeans = None
        self.cluster_centers_ = None
        self.labels_ = None
    
    def find_optimal_k(self, data: np.ndarray) -> int:
        """
        Determine optimal K using the Elbow Method/Silhouette Score
        
        Input:
            data: [n_samples, n_features]
        
        Output:
            Optimal K value
        """
        logger.info("Finding optimal K using Elbow Method...")
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(self.max_clusters + 1, len(data) // 10 + 1))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, 
                            random_state=self.random_state,
                            n_init=10)
            kmeans.fit(data)
            
            inertias.append(kmeans.inertia_)
            
            if k < len(data):
                score = silhouette_score(data, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Use K with the highest Silhouette Score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        logger.info(f"Optimal K: {optimal_k}")
        logger.info(f"Corresponding Silhouette Score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def fit(self, data: np.ndarray):
        """
        Fit Clusterer
        
        Input:
            data: [n_samples, n_features]
        """
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_k(data)
        
        logger.info(f"Starting K-means clustering, K={self.n_clusters}")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state,
                             n_init=20,
                             max_iter=300)
        self.labels_ = self.kmeans.fit_predict(data)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        logger.info("Clustering complete")
        self._print_cluster_info(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Map new data to cluster labels
        
        Input:
            data: [n_samples, n_features]
        
        Output:
            Cluster labels [n_samples,]
        """
        if self.kmeans is None:
            raise ValueError("Clusterer not fitted")
        
        return self.kmeans.predict(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(data)
        return self.labels_
    
    def _print_cluster_info(self, data: np.ndarray):
        """Print cluster stats"""
        unique, counts = np.unique(self.labels_, return_counts=True)
        
        logger.info("Cluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            ratio = count / len(data)
            logger.info(f"  Cluster {cluster_id}: {count} samples ({ratio:.1%})")
    
    def save(self, filepath: str):
        """Save Clusterer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'n_clusters': self.n_clusters,
                'cluster_centers': self.cluster_centers_
            }, f)
        logger.info(f"Clusterer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Clusterer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.n_clusters = data['n_clusters']
            self.cluster_centers_ = data['cluster_centers']
        logger.info(f"Clusterer loaded from {filepath}")


class AttentionClusterer(DLPClusterer):
    """
    Attention Vector Clusterer (Specialized)
    
    Extra functions:
    - Cumulative Attention vector processing
    - Identification of Early/Late/Other patterns
    """
    
    def preprocess_attention(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Preprocess Attention weight vectors
        
        Steps:
        1. Normalization (ensure sum=1)
        2. Compute cumulative sum
        
        Input:
            attention_weights: [n_samples, sequence_length]
        
        Output:
            Cumulative attention vectors [n_samples, sequence_length]
        """
        # Normalization
        normalized = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-8)
        
        # Cumulative Sum
        cumulative = np.cumsum(normalized, axis=1)
        
        return cumulative
    
    def fit(self, attention_weights: np.ndarray):
        """
        Fit (Auto-preprocessing)
        
        Input:
            attention_weights: [n_samples, sequence_length]
        """
        processed = self.preprocess_attention(attention_weights)
        super().fit(processed)
        
        # Name clusters based on cumulative patterns
        self._name_clusters(processed)
    
    def _name_clusters(self, cumulative_attention: np.ndarray):
        """
        Name clusters based on Cumulative Attention patterns
        
        Strategy:
        - Early: Cumulative sum > 70% at the 50% time mark
        - Late: Cumulative sum < 30% at the 50% time mark
        - Other: Otherwise
        """
        self.cluster_names_ = {}
        
        mid_point = cumulative_attention.shape[1] // 2
        
        for cluster_id in range(self.n_clusters):
            # Samples in this cluster
            mask = self.labels_ == cluster_id
            cluster_samples = cumulative_attention[mask]
            
            # Average cumulative value at mid-point
            mid_cumulative = cluster_samples[:, mid_point].mean()
            
            if mid_cumulative > 0.7:
                name = 'Early'
            elif mid_cumulative < 0.3:
                name = 'Late'
            else:
                name = 'Other'
            
            self.cluster_names_[cluster_id] = name
        
        logger.info("Attention Cluster Naming:")
        for cid, name in self.cluster_names_.items():
            logger.info(f"  Cluster {cid} -> {name}")
    
    def transform_to_names(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Convert to named labels
        
        Input:
            attention_weights: [n_samples, sequence_length]
        
        Output:
            Named labels [n_samples,] (e.g., ['Early', 'Late', ...])
        """
        processed = self.preprocess_attention(attention_weights)
        cluster_ids = super().transform(processed)
        
        return np.array([self.cluster_names_[cid] for cid in cluster_ids])


# Usage Example
if __name__ == "__main__":
    # === CAM Clustering ===
    np.random.seed(42)
    
    # Mock CAM data
    cam_data = np.random.rand(1000, 20)  # 1000 samples, 20 steps
    
    cam_clusterer = DLPClusterer(n_clusters=None)  # Auto K
    cam_labels = cam_clusterer.fit_transform(cam_data)
    
    print(f"CAM clustering complete, K={cam_clusterer.n_clusters}")
    print(f"Cluster labels for first 10 samples: {cam_labels[:10]}")
    
    # === Attention Clustering ===
    # Mock Attention weights (Early peak vs Late peak)
    attention_early = np.random.beta(2, 5, size=(500, 80))  # High early
    attention_late = np.random.beta(5, 2, size=(500, 80))   # High late
    attention_data = np.vstack([attention_early, attention_late])
    
    att_clusterer = AttentionClusterer(n_clusters=3)
    att_clusterer.fit(attention_data)
    
    # Convert to names
    att_names = att_clusterer.transform_to_names(attention_data[:10])
    print(f"\nAttention types for first 10 samples: {att_names}")

```

---

The above covers the implementation code for core modules. Due to space constraints, subsequent modules (Association Rules, Bayesian Network, Causal Inference) will be provided in the next sections.

Currently you can see:

1. **Project Design Document** created (Full architecture design)
2. **Implementation Document** being created (Detailed code)

All documents are located in the [doc/](https://www.google.com/search?q=doc/) directory. Next, I will finish the implementation document and start setting up the project code structure.

---

### 2.6 Association Rule Mining Module

#### File: `src/models/association.py`

```python
"""
Association Rule Mining Module (Apriori Algorithm)
Used to extract frequent association patterns between discrete variables
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class AssociationRuleMiner:
    """
    Association Rule Miner
    
    Extracts candidate causal relationships based on the Apriori algorithm
    """
    
    def __init__(self, 
                 min_support: float = 0.1,
                 min_confidence: float = 0.6,
                 min_lift: float = 1.0):
        """
        Parameters:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        
        self.frequent_itemsets_ = None
        self.rules_ = None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'EDP') -> pd.DataFrame:
        """
        Prepare One-hot encoded data
        
        Input:
            df: Discretized dataframe
            target_col: Target column name (EDP state)
        
        Output:
            One-hot encoded dataframe
        """
        logger.info("Preparing data for association rule mining...")
        
        # Create "ColumnName_Value" combinations for each row
        transactions = []
        
        for _, row in df.iterrows():
            items = []
            for col in df.columns:
                if pd.notna(row[col]):
                    # Create "Variable=Value" item
                    item = f"{col}_{row[col]}"
                    items.append(item)
            transactions.append(items)
        
        # One-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        logger.info(f"Encoded features count: {len(df_encoded.columns)}")
        return df_encoded
    
    def mine_frequent_itemsets(self, df_encoded: pd.DataFrame):
        """
        Mine frequent itemsets
        
        Input:
            df_encoded: One-hot encoded dataframe
        """
        logger.info(f"Mining frequent itemsets (min_support={self.min_support})...")
        
        self.frequent_itemsets_ = apriori(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True,
            low_memory=True
        )
        
        logger.info(f"Found {len(self.frequent_itemsets_)} frequent itemsets")
        return self.frequent_itemsets_
    
    def generate_rules(self, target_suffix: str = 'EDP'):
        """
        Generate association rules
        
        Input:
            target_suffix: Target variable suffix (used for filtering rules)
        
        Output:
            Rules dataframe
        """
        if self.frequent_itemsets_ is None:
            raise ValueError("Call mine_frequent_itemsets() first")
        
        logger.info("Generating association rules...")
        
        # Generate all rules
        all_rules = association_rules(
            self.frequent_itemsets_,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        
        # Filter: Keep only rules where the consequent contains the target variable
        def has_target_in_consequent(consequents):
            return any(target_suffix in str(item) for item in consequents)
        
        self.rules_ = all_rules[
            all_rules['consequents'].apply(has_target_in_consequent)
        ]
        
        # Further filter by lift
        self.rules_ = self.rules_[self.rules_['lift'] >= self.min_lift]
        
        # Sort by confidence
        self.rules_ = self.rules_.sort_values(
            by='confidence',
            ascending=False
        ).reset_index(drop=True)
        
        logger.info(f"Generated {len(self.rules_)} valid rules")
        return self.rules_
    
    def filter_rules_for_state(self, state: str) -> pd.DataFrame:
        """
        Filter rules for a specific EDP state
        
        Input:
            state: 'Peak', 'Normal', or 'Lower'
        
        Output:
            Subset of rules for the state
        """
        if self.rules_ is None:
            raise ValueError("Call generate_rules() first")
        
        state_key = f"EDP_{state}"
        
        filtered = self.rules_[
            self.rules_['consequents'].apply(
                lambda x: state_key in str(x)
            )
        ]
        
        logger.info(f"Rule count for state '{state}': {len(filtered)}")
        return filtered
    
    def get_top_rules(self, n: int = 10, state: str = None) -> pd.DataFrame:
        """
        Get Top-N rules
        
        Input:
            n: Number of rules to return
            state: Specific state (None for all)
        
        Output:
            Top rules dataframe
        """
        if state is not None:
            rules = self.filter_rules_for_state(state)
        else:
            rules = self.rules_
        
        return rules.head(n)
    
    def rules_to_constraints(self, state: str) -> List[tuple]:
        """
        Convert rules to candidate edges for Bayesian Network
        
        Input:
            state: EDP state
        
        Output:
            List of candidate edges [(from_var, to_var), ...]
        """
        state_rules = self.filter_rules_for_state(state)
        
        edges = []
        target_node = 'EDP'
        
        for _, rule in state_rules.iterrows():
            # Extract variable names from antecedents (remove value part)
            for antecedent in rule['antecedents']:
                # Format: "VariableName_Value"
                var_name = str(antecedent).split('_')[0]
                
                # Add edge: Variable -> EDP
                if var_name != target_node:
                    edges.append((var_name, target_node))
        
        # De-duplicate
        edges = list(set(edges))
        
        logger.info(f"Extracted {len(edges)} candidate edges for state '{state}'")
        return edges
    
    def print_rules_summary(self, n: int = 5):
        """
        Print rules summary
        
        Input:
            n: Number of rules to display per state
        """
        if self.rules_ is None or len(self.rules_) == 0:
            logger.warning("No rules found")
            return
        
        print("\n" + "=" * 80)
        print("Association Rules Summary")
        print("=" * 80)
        
        for state in ['Peak', 'Normal', 'Lower']:
            print(f"\n--- State: {state} ---")
            state_rules = self.filter_rules_for_state(state)
            
            if len(state_rules) == 0:
                print("  (No rules)")
                continue
            
            for idx, rule in state_rules.head(n).iterrows():
                antecedents_str = ', '.join([str(x) for x in rule['antecedents']])
                consequents_str = ', '.join([str(x) for x in rule['consequents']])
                
                print(f"\n  Rule {idx + 1}:")
                print(f"    IF {antecedents_str}")
                print(f"    THEN {consequents_str}")
                print(f"    Support: {rule['support']:.3f}")
                print(f"    Confidence: {rule['confidence']:.3f}")
                print(f"    Lift: {rule['lift']:.3f}")
        
        print("\n" + "=" * 80)


# Usage Example
if __name__ == "__main__":
    # Mock discretized data
    np.random.seed(42)
    
    data = pd.DataFrame({
        'ClimateControl': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'Kitchen': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'Laundry': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'GlobalActivePower': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 1000),
        'Weekend': np.random.choice(['Yes', 'No'], 1000),
        'EDP': np.random.choice(['Peak', 'Normal', 'Lower'], 1000, p=[0.2, 0.6, 0.2])
    })
    
    # Create rule miner
    miner = AssociationRuleMiner(
        min_support=0.05,
        min_confidence=0.5,
        min_lift=1.2
    )
    
    # Data preparation
    df_encoded = miner.prepare_data(data)
    
    # Mine frequent itemsets
    miner.mine_frequent_itemsets(df_encoded)
    
    # Generate rules
    rules = miner.generate_rules()
    
    # Print summary
    miner.print_rules_summary(n=3)
    
    # Extract candidate edges
    peak_edges = miner.rules_to_constraints('Peak')
    print(f"\nCandidate edges for Peak state: {peak_edges}")

```

---

### 2.7 Bayesian Network Module

#### File: `src/models/bayesian_net.py`

```python
"""
Bayesian Network Construction and Inference Module
"""

import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (
    HillClimbSearch, 
    BicScore, 
    MaximumLikelihoodEstimator,
    BayesianEstimator
)
from pgmpy.inference import VariableElimination
import networkx as nx
import pickle
import logging

logger = logging.getLogger(__name__)


class CausalBayesianNetwork:
    """
    Causal Bayesian Network
    
    Structure learning combined with domain knowledge constraints
    """
    
    def __init__(self, domain_knowledge: dict = None):
        """
        Parameters:
            domain_knowledge: Domain knowledge constraint dictionary
                {
                    'themes': {  # Theme grouping
                        'physical_env': [var list],
                        'appliances': [var list],
                        'consumption': [var list],
                        'dlp': [var list],
                        'target': [target variable]
                    },
                    'allow_directions': [  # Allowed directions
                        ('theme_from', 'theme_to'),
                        ...
                    ]
                }
        """
        self.domain_knowledge = domain_knowledge or self._default_domain_knowledge()
        
        self.model = None
        self.inference = None
        
        # White-list and black-list
        self.white_list = []
        self.black_list = []
    
    def _default_domain_knowledge(self) -> dict:
        """
        Default domain knowledge (based on Table 2 in the paper)
        """
        return {
            'themes': {
                'physical_env': ['Date', 'Day', 'Month', 'Season', 'Weekend'],
                'appliances': ['Kitchen', 'Laundry', 'ClimateControl', 'Other'],
                'consumption': ['GlobalActivePower', 'GlobalReactivePower', 
                                'Voltage', 'GlobalIntensity'],
                'dlp': ['CAM_type', 'ATT_type'],
                'target': ['EDP']
            },
            'allow_directions': [
                ('physical_env', 'appliances'),
                ('appliances', 'consumption'),
                ('consumption', 'target'),
                ('dlp', 'target')  # DLP directly influences target
            ]
        }
    
    def _build_constraints(self, variables: List[str]):
        """
        Construct white-list and black-list based on domain knowledge
        
        Input:
            variables: List of variables actually present in the data
        """
        logger.info("Constructing domain knowledge constraints...")
        
        # Map variables to themes
        var_to_theme = {}
        for theme, vars_list in self.domain_knowledge['themes'].items():
            for var in vars_list:
                if var in variables:
                    var_to_theme[var] = theme
        
        # Generate white-list (allowed edges)
        for from_theme, to_theme in self.domain_knowledge['allow_directions']:
            from_vars = [v for v, t in var_to_theme.items() if t == from_theme]
            to_vars = [v for v, t in var_to_theme.items() if t == to_theme]
            
            for from_var in from_vars:
                for to_var in to_vars:
                    if from_var != to_var:
                        self.white_list.append((from_var, to_var))
        
        # Generate black-list (forbidden edges)
        # Rule 1: Target variable cannot point to other variables
        target_vars = self.domain_knowledge['themes']['target']
        for target in target_vars:
            if target in variables:
                for var in variables:
                    if var != target:
                        self.black_list.append((target, var))
        
        # Rule 2: Reverse causality (consumption -> appliances, appliances -> physical_env)
        consumption_vars = [v for v, t in var_to_theme.items() 
                            if t == 'consumption']
        appliance_vars = [v for v, t in var_to_theme.items() 
                          if t == 'appliances']
        physical_vars = [v for v, t in var_to_theme.items() 
                         if t == 'physical_env']
        
        for cons_var in consumption_vars:
            for app_var in appliance_vars:
                self.black_list.append((cons_var, app_var))
        
        for app_var in appliance_vars:
            for phy_var in physical_vars:
                self.black_list.append((app_var, phy_var))
        
        logger.info(f"White-list edges count: {len(self.white_list)}")
        logger.info(f"Black-list edges count: {len(self.black_list)}")
    
    def learn_structure(self, data: pd.DataFrame, scoring_method='bic'):
        """
        Structure Learning (with domain constraints)
        
        Input:
            data: Discretized data
            scoring_method: Scoring method ('bic', 'k2', 'bdeu')
        
        Output:
            Learned DAG structure
        """
        logger.info("Starting Bayesian Network structure learning...")
        
        variables = list(data.columns)
        self._build_constraints(variables)
        
        # Select scoring function
        if scoring_method == 'bic':
            scoring = BicScore(data)
        else:
            raise ValueError(f"Unsupported scoring method: {scoring_method}")
        
        # Hill-Climbing Search
        hc = HillClimbSearch(data)
        
        best_model = hc.estimate(
            scoring_method=scoring,
            white_list=self.white_list if len(self.white_list) > 0 else None,
            black_list=self.black_list if len(self.black_list) > 0 else None,
            max_indegree=5,  # Limit in-degree to avoid complexity
            max_iter=int(1e4)
        )
        
        logger.info(f"Structure learning complete. Edge count: {len(best_model.edges())}")
        return best_model
    
    def learn_parameters(self, data: pd.DataFrame, structure=None, method='mle'):
        """
        Parameter Learning (CPT Estimation)
        
        Input:
            data: Discretized data
            structure: DAG structure (if None, use learned one)
            method: 'mle' (Maximum Likelihood) or 'bayes' (Bayesian Estimation)
        
        Output:
            Full Bayesian Network model
        """
        if structure is None and self.model is None:
            raise ValueError("Run structure learning first or provide a structure")
        
        if structure is not None:
            self.model = BayesianNetwork(structure.edges())
        
        logger.info(f"Parameter learning (Method: {method})...")
        
        if method == 'mle':
            estimator = MaximumLikelihoodEstimator
        elif method == 'bayes':
            estimator = BayesianEstimator
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.model.fit(data, estimator=estimator)
        
        # Initialize inference engine
        self.inference = VariableElimination(self.model)
        
        logger.info("Parameter learning complete")
        return self.model
    
    def query(self, variables: List[str], evidence: dict = None) -> dict:
        """
        Bayesian Inference
        
        Input:
            variables: List of query variables
            evidence: Observed evidence {'var': 'value', ...}
        
        Output:
            Posterior probability distribution
        """
        if self.inference is None:
            raise ValueError("Model not fitted, cannot perform inference")
        
        result = self.inference.query(
            variables=variables,
            evidence=evidence
        )
        
        return result
    
    def map_query(self, variables: List[str], evidence: dict = None) -> dict:
        """
        Maximum A Posteriori (MAP) Estimation
        
        Input:
            variables: Query variables
            evidence: Evidence
        
        Output:
            Most probable state
        """
        if self.inference is None:
            raise ValueError("Model not fitted")
        
        result = self.inference.map_query(
            variables=variables,
            evidence=evidence
        )
        
        return result
    
    def get_cpd(self, variable: str):
        """Get Conditional Probability Table"""
        return self.model.get_cpds(variable)
    
    def save(self, filepath: str):
        """Save Model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'domain_knowledge': self.domain_knowledge
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.domain_knowledge = data['domain_knowledge']
            self.inference = VariableElimination(self.model)
        logger.info(f"Model loaded from {filepath}")
    
    def visualize(self, filename: str = None):
        """Visualize DAG Structure"""
        if self.model is None:
            raise ValueError("Model not constructed")
        
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph(self.model.edges())
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=3000,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray')
        
        plt.title("Bayesian Network Structure", fontsize=16)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Network structure image saved to {filename}")
        else:
            plt.show()


# Usage Example
if __name__ == "__main__":
    # Mock data
    np.random.seed(42)
    data = pd.DataFrame({
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 1000),
        'Weekend': np.random.choice(['Yes', 'No'], 1000),
        'ClimateControl': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'Kitchen': np.random.choice(['Low', 'Medium', 'High'], 1000),
        'GlobalActivePower': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'CAM_type': np.random.choice(['Type1', 'Type2'], 1000),
        'ATT_type': np.random.choice(['Early', 'Late', 'Other'], 1000),
        'EDP': np.random.choice(['Peak', 'Normal', 'Lower'], 1000)
    })
    
    # Create BN
    bn = CausalBayesianNetwork()
    
    # Structure learning
    structure = bn.learn_structure(data)
    
    # Parameter learning
    bn.learn_parameters(data)
    
    # Inference
    evidence = {
        'Season': 'Summer',
        'ClimateControl': 'VeryHigh',
        'ATT_type': 'Late'
    }
    
    result = bn.query(variables=['EDP'], evidence=evidence)
    print("\nInference Result:")
    print(result)
    
    # Visualize
    bn.visualize('bn_structure.png')

```

---

### 2.8 Causal Inference Module

#### File: `src/inference/causal_inference.py`

```python
"""
Causal Inference and Sensitivity Analysis Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class CausalInferenceEngine:
    """
    Causal Inference Engine
    
    Performs causal inference and sensitivity analysis based on Bayesian Network
    """
    
    def __init__(self, bayesian_network):
        """
        Parameters:
            bayesian_network: CausalBayesianNetwork instance
        """
        self.bn = bayesian_network
        
        if self.bn.inference is None:
            raise ValueError("Bayesian Network not fitted")
    
    def causal_effect(self, 
                     intervention: Dict[str, str],
                     target: str,
                     evidence: Dict[str, str] = None) -> pd.DataFrame:
        """
        Calculate Causal Effect under do-calculus
        
        Input:
            intervention: Intervention variable and value {'var': 'value'}
            target: Target variable
            evidence: Observed evidence (optional)
        
        Output:
            Posterior distribution of the target variable
        """
        logger.info(f"Calculating causal effect: do({intervention}) -> {target}")
        
        # Merge intervention and evidence
        combined_evidence = evidence.copy() if evidence else {}
        combined_evidence.update(intervention)
        
        # Query target variable
        result = self.bn.query(
            variables=[target],
            evidence=combined_evidence
        )
        
        return result
    
    def sensitivity_analysis(self,
                            target: str,
                            variables: List[str],
                            baseline: Dict[str, str] = None,
                            n_samples: int = 100) -> pd.DataFrame:
        """
        Sensitivity Analysis (Univariate Perturbation)
        
        Input:
            target: Target variable
            variables: List of variables to analyze
            baseline: Baseline observation (if None, use empty evidence)
            n_samples: Sample count per variable
        
        Output:
            Sensitivity scores DataFrame
        """
        logger.info(f"Sensitivity Analysis: Target={target}, Variable count={len(variables)}")
        
        baseline = baseline or {}
        
        sensitivity_scores = {}
        
        for var in variables:
            # Get all possible values for variable
            var_cpd = self.bn.get_cpd(var)
            var_states = var_cpd.state_names[var]
            
            # Calculate target distribution under each value
            distributions = []
            
            for state in var_states:
                # Intervene on this variable
                evidence = baseline.copy()
                evidence[var] = state
                
                try:
                    result = self.bn.query(
                        variables=[target],
                        evidence=evidence
                    )
                    
                    # Extract probability distribution
                    probs = result.values
                    distributions.append(probs)
                    
                except Exception as e:
                    logger.warning(f"Query failed: {var}={state}, Error: {e}")
                    distributions.append(None)
            
            # Calculate variance of distributions (as sensitivity indicator)
            valid_dists = [d for d in distributions if d is not None]
            
            if len(valid_dists) > 1:
                # Calculate standard deviation of probabilities for Peak/Normal/Lower
                dist_array = np.array(valid_dists)
                sensitivity = np.mean(np.std(dist_array, axis=0))
            else:
                sensitivity = 0.0
            
            sensitivity_scores[var] = sensitivity
        
        # Convert to DataFrame and sort
        df = pd.DataFrame.from_dict(
            sensitivity_scores,
            orient='index',
            columns=['Sensitivity']
        ).sort_values(by='Sensitivity', ascending=False)
        
        logger.info(f"Sensitivity Analysis complete. Top variables: {df.head(3).index.tolist()}")
        
        return df
    
    def tornado_chart(self, 
                     sensitivity_df: pd.DataFrame,
                     top_n: int = 10,
                     filename: str = None):
        """
        Plot Tornado Chart (Sensitivity Visualization)
        
        Input:
            sensitivity_df: Sensitivity analysis result
            top_n: Display Top-N variables
            filename: Save path (None to show)
        """
        top_vars = sensitivity_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        
        y_pos = np.arange(len(top_vars))
        plt.barh(y_pos, top_vars['Sensitivity'], color='steelblue')
        
        plt.yticks(y_pos, top_vars.index)
        plt.xlabel('Sensitivity Score', fontsize=12)
        plt.title('Tornado Chart: Variable Sensitivity', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest at top
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Tornado chart saved to {filename}")
        else:
            plt.show()
    
    def counterfactual_analysis(self,
                                factual_evidence: Dict[str, str],
                                intervention: Dict[str, str],
                                target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Counterfactual Analysis
        
        Compare:
          - Factual Case: P(Target | Evidence)
          - Counterfactual Case: P(Target | do(Intervention), Evidence)
        
        Input:
            factual_evidence: Factual observation
            intervention: Counterfactual intervention
            target: Target variable
        
        Output:
            (Factual Distribution, Counterfactual Distribution)
        """
        logger.info("Counterfactual Analysis...")
        
        # Factual Inference
        factual_result = self.bn.query(
            variables=[target],
            evidence=factual_evidence
        )
        
        # Counterfactual Inference
        counterfactual_evidence = factual_evidence.copy()
        counterfactual_evidence.update(intervention)
        
        counterfactual_result = self.bn.query(
            variables=[target],
            evidence=counterfactual_evidence
        )
        
        logger.info("Counterfactual Analysis complete")
        
        return factual_result, counterfactual_result
    
    def compare_distributions(self,
                            dist1: pd.DataFrame,
                            dist2: pd.DataFrame,
                            labels: Tuple[str, str] = ('Factual', 'Counterfactual'),
                            filename: str = None):
        """
        Visualize and compare two distributions
        
        Input:
            dist1, dist2: Probability distributions
            labels: Distribution labels
            filename: Save path
        """
        states = dist1.state_names[dist1.variables[0]]
        
        x = np.arange(len(states))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.bar(x - width/2, dist1.values, width, label=labels[0], color='skyblue')
        ax.bar(x + width/2, dist2.values, width, label=labels[1], color='salmon')
        
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(states)
        ax.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution comparison image saved to {filename}")
        else:
            plt.show()
    
    def explain_state(self,
                     target: str,
                     target_state: str,
                     evidence: Dict[str, str],
                     top_n: int = 5) -> pd.DataFrame:
        """
        Explain key factors for a specific state
        
        Input:
            target: Target variable
            target_state: Target state (e.g., 'Peak')
            evidence: Current observation
            top_n: Return Top-N key factors
        
        Output:
            Key factors DataFrame
        """
        logger.info(f"Explaining state: {target}={target_state}")
        
        # Get all parent nodes
        parents = list(self.bn.model.get_parents(target))
        
        if len(parents) == 0:
            logger.warning(f"{target} has no parent nodes, cannot explain")
            return pd.DataFrame()
        
        # Calculate contribution of each parent node
        contributions = {}
        
        # Baseline: No evidence
        baseline_prob = self.bn.query(
            variables=[target],
            evidence={}
        ).values
        
        # Get target_state index
        target_cpd = self.bn.get_cpd(target)
        target_states = target_cpd.state_names[target]
        state_idx = list(target_states).index(target_state)
        baseline_target_prob = baseline_prob[state_idx]
        
        # Calculate contribution for each piece of evidence
        for var in parents:
            if var in evidence:
                # Probability with this evidence
                single_evidence = {var: evidence[var]}
                
                prob_with_evidence = self.bn.query(
                    variables=[target],
                    evidence=single_evidence
                ).values[state_idx]
                
                # Contribution = P(Target=state|Var) - P(Target=state)
                contribution = prob_with_evidence - baseline_target_prob
                
                contributions[var] = {
                    'Value': evidence[var],
                    'Contribution': contribution,
                    'P(Target|Var)': prob_with_evidence
                }
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(contributions, orient='index')
        df = df.sort_values(by='Contribution', ascending=False).head(top_n)
        
        logger.info(f"Top {top_n} key factors identified")
        
        return df


# Usage Example
if __name__ == "__main__":
    from bayesian_net import CausalBayesianNetwork
    
    # Build sample BN
    np.random.seed(42)
    data = pd.DataFrame({
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 1000),
        'ClimateControl': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'GlobalActivePower': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'ATT_type': np.random.choice(['Early', 'Late', 'Other'], 1000),
        'EDP': np.random.choice(['Peak', 'Normal', 'Lower'], 1000)
    })
    
    bn = CausalBayesianNetwork()
    structure = bn.learn_structure(data)
    bn.learn_parameters(data)
    
    # Create Causal Inference Engine
    engine = CausalInferenceEngine(bn)
    
    # Sensitivity Analysis
    sensitivity = engine.sensitivity_analysis(
        target='EDP',
        variables=['Season', 'ClimateControl', 'GlobalActivePower', 'ATT_type']
    )
    print(sensitivity)
    
    # Plot Tornado Chart
    engine.tornado_chart(sensitivity, filename='tornado.png')
    
    # Counterfactual Analysis
    factual_ev = {
        'Season': 'Summer',
        'ClimateControl': 'VeryHigh'
    }
    
    intervention = {
        'ClimateControl': 'Low'  # Counterfactual: What if air con usage was low?
    }
    
    factual, counterfactual = engine.counterfactual_analysis(
        factual_evidence=factual_ev,
        intervention=intervention,
        target='EDP'
    )
    
    print("\nFactual Distribution:")
    print(factual)
    print("\nCounterfactual Distribution:")
    print(counterfactual)
    
    # Visualize Comparison
    engine.compare_distributions(factual, counterfactual, filename='counterfactual.png')

```

---

### 2.9 Recommendation Generation Module

#### File: `src/inference/recommendation.py`

```python
"""
Actionable Recommendation Generation Module
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """
    Causal Inference-based Recommendation Generator
    
    Generates energy-saving suggestions tailored for different EDP states
    """
    
    def __init__(self, causal_engine):
        """
        Parameters:
            causal_engine: CausalInferenceEngine instance
        """
        self.engine = causal_engine
        self.bn = causal_engine.bn
        
        # Recommendation Templates (based on Actionable Insights in Table 7)
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """
        Load Recommendation Templates (based on examples from the paper)
        """
        return {
            'Peak': {
                'ClimateControl': {
                    'VeryHigh': "Excessive air conditioning is the main cause of peak usage. Suggested increase in temp by 2-3°C can reduce energy by {reduction}%.",
                    'High': "Air conditioning usage is high. Suggest use during off-peak or switching to eco mode."
                },
                'Kitchen': {
                    'VeryHigh': "Concentrated kitchen appliance usage causes peak load. Stagger use of high-power devices like ovens/microwaves.",
                    'High': "Kitchen energy usage is high. Suggest distributed usage of high-power appliances."
                },
                'Laundry': {
                    'VeryHigh': "Dense laundry/drying equipment usage. Running during off-peak night hours can save {reduction}% on bills.",
                    'High': "Laundry usage is high. Suggest cold wash mode to reduce heating energy."
                },
                'Season': {
                    'Summer': "High summer temps increase AC load. Suggest blackout curtains to reduce indoor heat accumulation.",
                    'Winter': "High heating demand in winter. Suggest checking window/door seals to reduce heat loss."
                },
                'GlobalActivePower': {
                    'VeryHigh': "Total active power is too high; overload risk. Suggest immediate shutdown of non-essential appliances."
                }
            },
            'Normal': {
                'ClimateControl': {
                    'Medium': "Usage is normal. Keep current AC settings.",
                    'High': "AC usage is slightly high. Consider slight temp adjustments for further optimization."
                },
                'Kitchen': {
                    'Medium': "Kitchen usage is reasonable. Recommend using high energy-efficiency appliances."
                },
                'Laundry': {
                    'Medium': "Laundry usage is normal. Can choose quick wash mode to save time and energy."
                }
            },
            'Lower': {
                'ClimateControl': {
                    'Low': "Energy level is low, target reached. Continue good habits."
                },
                'Season': {
                    'Spring': "Pleasant spring weather, no AC needed. Suggest natural ventilation.",
                    'Fall': "Moderate autumn temps. Suggest reducing AC and utilizing natural temp regulation."
                },
                'Weekend': {
                    'No': "Long hours away on workdays keep usage naturally low. Suggest checking all devices are off before leaving."
                }
            }
        }
    
    def generate_recommendations(self,
                                current_state: str,
                                evidence: Dict[str, str],
                                top_n: int = 3) -> List[Dict]:
        """
        Generate Recommendation List
        
        Input:
            current_state: Current EDP state ('Peak', 'Normal', 'Lower')
            evidence: Current observed variable values
            top_n: Return Top-N recommendations
        
        Output:
            Recommendation list [{'variable': ..., 'action': ..., 'impact': ..., 'text': ...}, ...]
        """
        logger.info(f"Generating Recommendations: State={current_state}, Evidence count={len(evidence)}")
        
        recommendations = []
        
        # 1. Identify key factors
        key_factors = self.engine.explain_state(
            target='EDP',
            target_state=current_state,
            evidence=evidence,
            top_n=top_n * 2  # Take extra for filtering
        )
        
        # 2. Generate recommendation for each key factor
        for var, row in key_factors.iterrows():
            var_value = row['Value']
            contribution = row['Contribution']
            
            # Find template
            recommendation_text = self._get_template(
                state=current_state,
                variable=var,
                value=var_value
            )
            
            if recommendation_text is None:
                # No template, generate generic recommendation
                recommendation_text = self._generate_generic_recommendation(
                    state=current_state,
                    variable=var,
                    value=var_value,
                    contribution=contribution
                )
            
            # Estimate intervention impact
            impact = self._estimate_impact(
                variable=var,
                current_value=var_value,
                target_state=current_state,
                evidence=evidence
            )
            
            recommendations.append({
                'variable': var,
                'current_value': var_value,
                'contribution': contribution,
                'action': self._suggest_action(var, var_value, current_state),
                'impact': impact,
                'text': recommendation_text
            })
        
        # 3. Sort by impact and return Top-N
        recommendations = sorted(
            recommendations,
            key=lambda x: abs(x['impact']),
            reverse=True
        )[:top_n]
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def _get_template(self, state: str, variable: str, value: str) -> str:
        """
        Retrieve recommendation text from template library
        """
        try:
            return self.templates[state][variable][value]
        except KeyError:
            return None
    
    def _generate_generic_recommendation(self,
                                        state: str,
                                        variable: str,
                                        value: str,
                                        contribution: float) -> str:
        """
        Generate generic recommendation text
        """
        if state == 'Peak':
            if contribution > 0.1:
                return f"{variable} is currently {value}, which is a significant factor in peak usage. Suggest adjusting or reducing use."
            else:
                return f"{variable}={value} has some impact on current usage; consider optimizing."
        
        elif state == 'Lower':
            return f"{variable}={value} helps maintain low energy usage. Keep current settings."
        
        else:  # Normal
            return f"{variable}={value}, energy usage is at normal levels."
    
    def _suggest_action(self, variable: str, value: str, state: str) -> str:
        """
        Suggest specific actions
        """
        action_map = {
            'Peak': {
                'ClimateControl': {
                    'VeryHigh': 'Raise AC temp by 2-3°C',
                    'High': 'Switch to eco mode'
                },
                'Kitchen': {
                    'VeryHigh': 'Off-peak use of high-power appliances',
                    'High': 'Reduce concurrent use of multiple appliances'
                },
                'Laundry': {
                    'VeryHigh': 'Delay operation to off-peak night hours',
                    'High': 'Use cold wash / quick wash mode'
                }
            }
        }
        
        try:
            return action_map[state][variable][value]
        except KeyError:
            return 'Optimize usage pattern'
    
    def _estimate_impact(self,
                        variable: str,
                        current_value: str,
                        target_state: str,
                        evidence: Dict[str, str]) -> float:
        """
        Estimate intervention impact (magnitude of Peak probability reduction)
        
        Output:
            Impact coefficient (-1 to 1, negative indicates reduction of Peak prob)
        """
        # Current probability
        current_prob = self.bn.query(
            variables=['EDP'],
            evidence=evidence
        ).values
        
        # Get index for Peak state
        edp_cpd = self.bn.get_cpd('EDP')
        peak_idx = list(edp_cpd.state_names['EDP']).index(target_state)
        current_peak_prob = current_prob[peak_idx]
        
        # Try changing this variable to optimal value
        var_cpd = self.bn.get_cpd(variable)
        var_states = var_cpd.state_names[variable]
        
        best_improvement = 0.0
        
        for alt_value in var_states:
            if alt_value == current_value:
                continue
            
            # Intervention evidence
            intervention_evidence = evidence.copy()
            intervention_evidence[variable] = alt_value
            
            try:
                new_prob = self.bn.query(
                    variables=['EDP'],
                    evidence=intervention_evidence
                ).values[peak_idx]
                
                improvement = current_peak_prob - new_prob
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    
            except Exception:
                continue
        
        return best_improvement
    
    def format_recommendations(self, recommendations: List[Dict]) -> str:
        """
        Format recommendations as readable text
        
        Input:
            recommendations: Recommendation list
        
        Output:
            Formatted text string
        """
        if len(recommendations) == 0:
            return "Current usage pattern is good, no adjustments needed."
        
        output = "=== Energy Saving Recommendations ===\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            output += f"{i}. {rec['text']}\n"
            output += f"    Action: {rec['action']}\n"
            output += f"    Expected Impact: {rec['impact']:.1%}\n"
            output += "\n"
        
        return output
    
    def save_recommendations(self, 
                           recommendations: List[Dict],
                           filename: str):
        """
        Save recommendations to file
        """
        text = self.format_recommendations(recommendations)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Recommendations saved to {filename}")


# Usage Example
if __name__ == "__main__":
    from bayesian_net import CausalBayesianNetwork
    from causal_inference import CausalInferenceEngine
    
    # Build BN
    np.random.seed(42)
    data = pd.DataFrame({
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 1000),
        'ClimateControl': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'Kitchen': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'Laundry': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'GlobalActivePower': np.random.choice(['Low', 'Medium', 'High', 'VeryHigh'], 1000),
        'EDP': np.random.choice(['Peak', 'Normal', 'Lower'], 1000, p=[0.2, 0.6, 0.2])
    })
    
    bn = CausalBayesianNetwork()
    structure = bn.learn_structure(data)
    bn.learn_parameters(data)
    
    # Create Recommendation Generator
    engine = CausalInferenceEngine(bn)
    recommender = RecommendationGenerator(engine)
    
    # Generate recommendations
    current_evidence = {
        'Season': 'Summer',
        'ClimateControl': 'VeryHigh',
        'Kitchen': 'High',
        'Laundry': 'Medium',
        'GlobalActivePower': 'VeryHigh'
    }
    
    recommendations = recommender.generate_recommendations(
        current_state='Peak',
        evidence=current_evidence,
        top_n=3
    )
    
    # Print recommendations
    print(recommender.format_recommendations(recommendations))
    
    # Save recommendations
    recommender.save_recommendations(recommendations, 'recommendations.txt')

```

---

### 2.10 Training Pipeline

#### File: `src/pipeline/train_pipeline.py`

```python
"""
Full Training Pipeline
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import json
import pickle

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-End Training Pipeline
    
    Includes: Data Preprocessing -> Model Training -> State Classification -> Discretization -> Clustering -> Association Rules -> Bayesian Network
    """
    
    def __init__(self, config: dict):
        """
        Parameters:
            config: Config dictionary
                {
                    'data_path': Path to data file,
                    'output_dir': Output directory,
                    'model_params': {...},
                    'training_params': {...},
                    ...
                }
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Module components
        self.preprocessor = None
        self.predictor = None
        self.classifier = None
        self.discretizer = None
        self.cam_clusterer = None
        self.att_clusterer = None
        self.rule_miner = None
        self.bn = None
    
    def run(self):
        """
        Execute full training pipeline
        """
        logger.info("=" * 80)
        logger.info("Starting Full Training Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Data loading and preprocessing
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data()
        
        # Step 2: Train prediction model
        self.train_predictor(X_train, y_train, X_val, y_val)
        
        # Step 3: Evaluate prediction performance
        self.evaluate_predictor(X_test, y_test)
        
        # Step 4: State classification (Sn scale)
        edp_states = self.classify_states(y_test)
        
        # Step 5: Discretization
        discrete_data = self.discretize_features(X_test, edp_states)
        
        # Step 6: DLP clustering
        discrete_data = self.cluster_dlp(X_test, discrete_data)
        
        # Step 7: Association rule mining
        self.mine_association_rules(discrete_data)
        
        # Step 8: Bayesian Network construction
        self.build_bayesian_network(discrete_data)
        
        # Step 9: Save all models
        self.save_models()
        
        logger.info("=" * 80)
        logger.info("Training Pipeline Complete!")
        logger.info("=" * 80)
    
    def prepare_data(self):
        """
        Step 1: Data Preparation
        """
        logger.info("\n[Step 1] Data loading and preprocessing...")
        
        from ..preprocessing.data_preprocessor import EnergyDataPreprocessor
        
        # Load data
        df = pd.read_csv(self.config['data_path'])
        logger.info(f"Dataset size: {df.shape}")
        
        # Init preprocessor
        self.preprocessor = EnergyDataPreprocessor(
            sequence_length=self.config.get('sequence_length', 60),
            feature_cols=self.config.get('feature_cols'),
            target_col=self.config.get('target_col', 'GlobalActivePower')
        )
        
        # Preprocess
        X, y = self.preprocessor.fit_transform(df)
        
        # Split dataset
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_predictor(self, X_train, y_train, X_val, y_val):
        """
        Step 2: Train Parallel CNN-LSTM-Attention Model
        """
        logger.info("\n[Step 2] Training prediction model...")
        
        from ..models.predictor import ParallelCNNLSTMAttention
        
        # Model parameters
        input_shape = (X_train.shape[1], X_train.shape[2])
        model_params = self.config.get('model_params', {})
        
        # Build model
        self.predictor = ParallelCNNLSTMAttention(
            input_shape=input_shape,
            **model_params
        )
        
        # Compile
        self.predictor.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get('learning_rate', 0.001)
            ),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train
        history = self.predictor.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 64),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history.history, f, indent=2)
        
        logger.info("Model training complete")
    
    def evaluate_predictor(self, X_test, y_test):
        """
        Step 3: Evaluate Prediction Performance
        """
        logger.info("\n[Step 3] Evaluating prediction model...")
        
        # Predict
        y_pred = self.predictor.predict(X_test)
        
        # Compute metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape)
        }
        
        logger.info(f"Performance: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        
        # Save metrics
        with open(self.output_dir / 'prediction_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def classify_states(self, y_test):
        """
        Step 4: State Classification using Sn scale estimator
        """
        logger.info("\n[Step 4] State Classification...")
        
        from ..models.state_classifier import SnStateClassifier
        
        # Init classifier
        self.classifier = SnStateClassifier(
            n_states=self.config.get('n_states', 3),
            state_names=self.config.get('state_names', ['Lower', 'Normal', 'Peak'])
        )
        
        # Fit and predict
        edp_states = self.classifier.fit_predict(y_test.flatten())
        
        # Statistics
        unique, counts = np.unique(edp_states, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        logger.info(f"State distribution: {distribution}")
        
        return edp_states
    
    def discretize_features(self, X_test, edp_states):
        """
        Step 5: Feature Discretization
        """
        logger.info("\n[Step 5] Feature Discretization...")
        
        from ..models.discretizer import QuantileDiscretizer
        
        # Extract last-step features from sequence data
        X_last = X_test[:, -1, :]
        
        # Init discretizer
        self.discretizer = QuantileDiscretizer(n_bins=4)
        
        # Discretize
        discrete_features = self.discretizer.fit_transform(X_last)
        
        # Combine into DataFrame
        feature_names = self.preprocessor.feature_cols
        
        discrete_data = pd.DataFrame(
            discrete_features,
            columns=feature_names
        )
        
        # Add EDP state
        discrete_data['EDP'] = edp_states
        
        logger.info(f"Discretized data shape: {discrete_data.shape}")
        
        return discrete_data
    
    def cluster_dlp(self, X_test, discrete_data):
        """
        Step 6: DLP Clustering
        """
        logger.info("\n[Step 6] DLP Clustering...")
        
        from ..models.clustering import DLPClusterer, AttentionClusterer
        
        # Extract CAM
        cam_values = self.predictor.extract_cam(X_test)
        
        # CAM clustering
        self.cam_clusterer = DLPClusterer(
            n_clusters=self.config.get('cam_clusters', 3),
            dlp_type='CAM'
        )
        cam_labels = self.cam_clusterer.fit_predict(cam_values)
        
        discrete_data['CAM_type'] = [f'Type{i+1}' for i in cam_labels]
        
        # Extract Attention weights
        att_weights = self.predictor.extract_attention_weights(X_test)
        
        # Attention clustering
        self.att_clusterer = AttentionClusterer(
            n_clusters=self.config.get('att_clusters', 3)
        )
        att_labels = self.att_clusterer.fit_predict(att_weights)
        
        discrete_data['ATT_type'] = self.att_clusterer.cluster_names_[att_labels]
        
        logger.info("DLP Clustering Complete")
        
        return discrete_data
    
    def mine_association_rules(self, discrete_data):
        """
        Step 7: Association Rule Mining
        """
        logger.info("\n[Step 7] Association Rule Mining...")
        
        from ..models.association import AssociationRuleMiner
        
        self.rule_miner = AssociationRuleMiner(
            min_support=self.config.get('min_support', 0.05),
            min_confidence=self.config.get('min_confidence', 0.5),
            min_lift=self.config.get('min_lift', 1.2)
        )
        
        # Prepare data
        df_encoded = self.rule_miner.prepare_data(discrete_data)
        
        # Mine
        self.rule_miner.mine_frequent_itemsets(df_encoded)
        rules = self.rule_miner.generate_rules()
        
        # Save rules
        rules.to_csv(self.output_dir / 'association_rules.csv', index=False)
        
        # Print summary
        self.rule_miner.print_rules_summary(n=3)
        
        logger.info(f"Mined {len(rules)} rules")
    
    def build_bayesian_network(self, discrete_data):
        """
        Step 8: Bayesian Network Construction
        """
        logger.info("\n[Step 8] Bayesian Network Construction...")
        
        from ..models.bayesian_net import CausalBayesianNetwork
        
        # Init BN
        domain_knowledge = self.config.get('domain_knowledge', None)
        self.bn = CausalBayesianNetwork(domain_knowledge=domain_knowledge)
        
        # Structure learning
        structure = self.bn.learn_structure(discrete_data)
        logger.info(f"Learned {len(structure.edges())} edges")
        
        # Parameter learning
        self.bn.learn_parameters(discrete_data)
        
        # Visualize
        self.bn.visualize(filename=str(self.output_dir / 'bn_structure.png'))
        
        logger.info("Bayesian Network Construction Complete")
    
    def save_models(self):
        """
        Step 9: Save All Models
        """
        logger.info("\n[Step 9] Saving Models...")
        
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save preprocessor
        with open(models_dir / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save prediction model
        self.predictor.save(str(models_dir / 'predictor.h5'))
        
        # Save classifier
        with open(models_dir / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save discretizer
        with open(models_dir / 'discretizer.pkl', 'wb') as f:
            pickle.dump(self.discretizer, f)
        
        # Save clusterers
        with open(models_dir / 'cam_clusterer.pkl', 'wb') as f:
            pickle.dump(self.cam_clusterer, f)
        
        with open(models_dir / 'att_clusterer.pkl', 'wb') as f:
            pickle.dump(self.att_clusterer, f)
        
        # Save rule miner
        with open(models_dir / 'rule_miner.pkl', 'wb') as f:
            pickle.dump(self.rule_miner, f)
        
        # Save Bayesian Network
        self.bn.save(str(models_dir / 'bayesian_network.pkl'))
        
        logger.info(f"All models saved to {models_dir}")


# Usage Example
if __name__ == "__main__":
    # Config
    config = {
        'data_path': 'data/household_power_consumption.txt',
        'output_dir': 'outputs/training',
        
        # Data params
        'sequence_length': 60,
        'feature_cols': [
            'GlobalActivePower', 'GlobalReactivePower',
            'Voltage', 'GlobalIntensity',
            'Kitchen', 'Laundry', 'ClimateControl'
        ],
        'target_col': 'GlobalActivePower',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        
        # Model params
        'model_params': {
            'cnn_filters': 64,
            'lstm_units': 128,
            'attention_units': 64,
            'dense_units': [64, 32]
        },
        
        # Training params
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        
        # Other params
        'n_states': 3,
        'state_names': ['Lower', 'Normal', 'Peak'],
        'cam_clusters': 3,
        'att_clusters': 3,
        'min_support': 0.05,
        'min_confidence': 0.5,
        'min_lift': 1.2
    }
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()

```

---

### 2.11 Inference Pipeline

#### File: `src/pipeline/inference_pipeline.py`

```python
"""
Inference Pipeline (Single sample prediction + Causal explanation + Recommendation)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Inference Pipeline
    
    Input: New Sample -> Prediction -> Causal Explanation -> Recommendation Generation
    """
    
    def __init__(self, models_dir: str):
        """
        Parameters:
            models_dir: Directory containing all trained models
        """
        self.models_dir = Path(models_dir)
        
        # Load all models
        self.load_models()
    
    def load_models(self):
        """
        Load all trained models
        """
        logger.info("Loading models...")
        
        # Preprocessor
        with open(self.models_dir / 'preprocessor.pkl', 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Prediction Model
        from ..models.predictor import ParallelCNNLSTMAttention
        self.predictor = ParallelCNNLSTMAttention.load(
            str(self.models_dir / 'predictor.h5')
        )
        
        # State Classifier
        with open(self.models_dir / 'classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Discretizer
        with open(self.models_dir / 'discretizer.pkl', 'rb') as f:
            self.discretizer = pickle.load(f)
        
        # Clusterers
        with open(self.models_dir / 'cam_clusterer.pkl', 'rb') as f:
            self.cam_clusterer = pickle.load(f)
        
        with open(self.models_dir / 'att_clusterer.pkl', 'rb') as f:
            self.att_clusterer = pickle.load(f)
        
        # Bayesian Network
        from ..models.bayesian_net import CausalBayesianNetwork
        self.bn = CausalBayesianNetwork()
        self.bn.load(str(self.models_dir / 'bayesian_network.pkl'))
        
        # Causal Inference Engine
        from ..inference.causal_inference import CausalInferenceEngine
        self.causal_engine = CausalInferenceEngine(self.bn)
        
        # Recommendation Generator
        from ..inference.recommendation import RecommendationGenerator
        self.recommender = RecommendationGenerator(self.causal_engine)
        
        logger.info("All models loaded successfully")
    
    def predict(self, raw_input: pd.DataFrame) -> dict:
        """
        Full inference process
        
        Input:
            raw_input: Raw input data (single or multiple)
        
        Output:
            Inference results dictionary
        """
        logger.info("Starting inference...")
        
        # 1. Preprocess
        X = self.preprocessor.transform(raw_input)
        
        # 2. Predict
        y_pred = self.predictor.predict(X)
        
        # 3. State Classification
        edp_state = self.classifier.predict(y_pred.flatten())[-1]  # Take the last one
        
        # 4. Extract DLP
        cam = self.predictor.extract_cam(X)
        att = self.predictor.extract_attention_weights(X)
        
        # 5. DLP Clustering
        cam_label = self.cam_clusterer.predict(cam)[-1]
        att_label = self.att_clusterer.predict(att)[-1]
        
        cam_type = f'Type{cam_label + 1}'
        att_type = self.att_clusterer.cluster_names_[att_label]
        
        # 6. Discretize Features
        X_last = X[-1, -1, :]  # Last step of the last sample
        discrete_features = self.discretizer.transform(X_last.reshape(1, -1))[0]
        
        # 7. Construct Evidence
        evidence = {}
        for i, col in enumerate(self.preprocessor.feature_cols):
            evidence[col] = discrete_features[i]
        
        evidence['CAM_type'] = cam_type
        evidence['ATT_type'] = att_type
        
        # 8. Causal Inference
        query_result = self.bn.query(
            variables=['EDP'],
            evidence=evidence
        )
        
        # 9. Generate Recommendations
        recommendations = self.recommender.generate_recommendations(
            current_state=edp_state,
            evidence=evidence,
            top_n=3
        )
        
        # 10. Integrate Results
        result = {
            'prediction': {
                'value': float(y_pred[-1]),
                'state': edp_state
            },
            'dlp': {
                'cam_type': cam_type,
                'attention_type': att_type
            },
            'causal_probability': {
                state: float(prob)
                for state, prob in zip(
                    query_result.state_names['EDP'],
                    query_result.values
                )
            },
            'evidence': evidence,
            'recommendations': recommendations,
            'recommendation_text': self.recommender.format_recommendations(recommendations)
        }
        
        logger.info(f"Inference complete: Pred={y_pred[-1]:.4f}, State={edp_state}")
        
        return result
    
    def explain(self, raw_input: pd.DataFrame, detailed: bool = False) -> dict:
        """
        Detailed Explanation (includes sensitivity, counterfactuals, etc.)
        
        Input:
            raw_input: Raw input
            detailed: Whether to include detailed analysis
        
        Output:
            Explanation results
        """
        # Base inference
        result = self.predict(raw_input)
        
        if not detailed:
            return result
        
        logger.info("Generating detailed explanation...")
        
        evidence = result['evidence']
        state = result['prediction']['state']
        
        # Sensitivity analysis
        sensitivity = self.causal_engine.sensitivity_analysis(
            target='EDP',
            variables=list(self.preprocessor.feature_cols),
            baseline=evidence
        )
        
        result['sensitivity'] = sensitivity.to_dict()
        
        # Key factor explanation
        key_factors = self.causal_engine.explain_state(
            target='EDP',
            target_state=state,
            evidence=evidence,
            top_n=5
        )
        
        result['key_factors'] = key_factors.to_dict()
        
        logger.info("Detailed explanation generated")
        
        return result
    
    def batch_predict(self, raw_inputs: pd.DataFrame) -> list:
        """
        Batch Inference
        
        Input:
            raw_inputs: Multiple raw inputs
        
        Output:
            List of results
        """
        results = []
        
        for i in range(len(raw_inputs)):
            sample = raw_inputs.iloc[i:i+1]
            result = self.predict(sample)
            results.append(result)
        
        return results


# Usage Example
if __name__ == "__main__":
    # Init pipeline
    pipeline = InferencePipeline(models_dir='outputs/training/models')
    
    # Prepare test data
    test_input = pd.DataFrame({
        'Date': ['2025-06-15 14:30:00'],
        'GlobalActivePower': [4.5],
        'GlobalReactivePower': [0.3],
        'Voltage': [240.0],
        'GlobalIntensity': [18.0],
        'Kitchen': [2.0],
        'Laundry': [1.5],
        'ClimateControl': [3.5]
    })
    
    # Inference
    result = pipeline.predict(test_input)
    
    print("\n=== Inference Result ===")
    print(f"Prediction Value: {result['prediction']['value']:.4f}")
    print(f"State: {result['prediction']['state']}")
    print(f"\nDLP: CAM={result['dlp']['cam_type']}, Attention={result['dlp']['attention_type']}")
    print(f"\nCausal Probabilities: {result['causal_probability']}")
    print(f"\n{result['recommendation_text']}")
    
    # Detailed Explanation
    detailed_result = pipeline.explain(test_input, detailed=True)
    
    print("\n=== Sensitivity Analysis ===")
    for var, score in detailed_result['sensitivity'].items():
        print(f"{var}: {score:.4f}")

```

---

## 3. Summary

### 3.1 Inter-module Dependencies

```
Data Preprocessing (preprocessing/data_preprocessor.py)
    ↓
Prediction Model (models/predictor.py)
    ↓
State Classification (models/state_classifier.py)
    ↓
Discretization (models/discretizer.py) + DLP Clustering (models/clustering.py)
    ↓
Association Rules (models/association.py) → Bayesian Network (models/bayesian_net.py)
    ↓
Causal Inference (inference/causal_inference.py)
    ↓
Recommendation Generation (inference/recommendation.py)
    ↓
Pipeline Integration (pipeline/train_pipeline.py, pipeline/inference_pipeline.py)

```

### 3.2 Key Technical Highlights

1. **Parallel Architecture**: CNN and LSTM-Attention processed in parallel, integrated via a fusion layer.
2. **Robust Classification**: Sn scale estimator handles outliers effectively.
3. **DLP Explanation**: Clustering of CAM and Attention weights provides interpretability.
4. **Causal Inference**: Bayesian Networks combined with domain knowledge constraints.
5. **Consistency**: Explanation results are highly stable (cosine similarity reaches 0.999+ as per the paper).

### 3.3 Next Steps

Refer to Section 11 "Development Plan" in `Project_Design_Doc.md`, following this implementation order:

1. ✅ **Complete Implementation Document writing** (Current Task)
2. ⬜ Set up project directory structure
3. ⬜ Implement Data Preprocessing module
4. ⬜ Implement Parallel CNN-LSTM-Attention model
5. ⬜ Implement State Classification and Discretization
6. ⬜ Implement DLP Clustering
7. ⬜ Implement Association Rules and Bayesian Network
8. ⬜ Implement Causal Inference and Recommendation Generation
9. ⬜ Integrate Training and Inference pipelines
10. ⬜ Unit Testing and Integration Testing
11. ⬜ Dataset preparation and model training
12. ⬜ Performance evaluation and tuning

---

**Document Version**: v1.0

**Author**: Severin YE

Would you like me to start creating the actual directory structure and the corresponding Python files based on this document?