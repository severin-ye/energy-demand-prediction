"""
Parallel CNN-LSTM-Attention Prediction Model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Custom Attention Mechanism Layer
    
    Calculation: score = tanh(W * h + b)
                attention_weights = softmax(score)
                context = sum(attention_weights * h)
    """
    
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_W',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Input: [batch, timesteps, features]
        Output: context_vector [batch, features], attention_weights [batch, timesteps]
        """
        # score = tanh(W*h + b)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # attention_weights = softmax(u*score)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        
        # context = weighted sum
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1),
            axis=1
        )
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class ParallelCNNLSTMAttention:
    """
    Parallel CNN-LSTM-Attention Architecture
    
    Structure:
        - CNN Branch: 1D convolution to extract local patterns
        - LSTM-Attention Branch: Long-term dependencies + Attention
        - Fusion Layer: Concatenation followed by MLP regression
    """
    
    def __init__(self,
                 input_shape: tuple,
                 cnn_filters: int = 64,
                 lstm_units: int = 128,
                 attention_units: int = 64,
                 dense_units: list = [64, 32]):
        """
        Parameters:
            input_shape: (sequence_length, n_features)
            cnn_filters: Number of CNN filters
            lstm_units: Number of LSTM hidden units
            attention_units: Number of units in Attention layer
            dense_units: List of units for fully connected layers
        """
        self.input_shape = input_shape
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        
        self.model = self._build_model()
        self.cam_model = None
        self.attention_model = None
    
    def _build_model(self):
        """Constructs the parallel architecture"""
        # Input
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ===== CNN Branch =====
        cnn_branch = layers.Conv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='cnn_conv1'
        )(inputs)
        
        cnn_branch = layers.MaxPooling1D(pool_size=2, name='cnn_pool1')(cnn_branch)
        
        cnn_branch = layers.Conv1D(
            filters=self.cnn_filters * 2,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='cnn_conv2'
        )(cnn_branch)
        
        # Save for CAM extraction
        cnn_features = layers.GlobalAveragePooling1D(name='cnn_gap')(cnn_branch)
        
        # ===== LSTM-Attention Branch =====
        lstm_branch = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            name='lstm'
        )(inputs)
        
        # Attention Layer
        attention_output, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_branch)
        
        # ===== Fusion =====
        merged = layers.Concatenate(name='merge')([cnn_features, attention_output])
        
        # Fully Connected Layers
        x = merged
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
            x = layers.Dropout(0.3, name=f'dropout{i+1}')(x)
        
        # Output Layer
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        # Build Model
        model = keras.Model(inputs=inputs, outputs=outputs, name='ParallelCNNLSTMAttention')
        
        logger.info("Model construction complete")
        logger.info(f"Total Parameters: {model.count_params():,}")
        
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=None):
        """Compiles the model"""
        if metrics is None:
            metrics = ['mae', 'mape']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, X_train, y_train, validation_data=None, **kwargs):
        """Trains the model"""
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            **kwargs
        )
    
    def predict(self, X):
        """Prediction"""
        return self.model.predict(X)
    
    def extract_cam(self, X):
        """
        Extracts CAM (Class Activation Mapping)
        
        Output:
            Array of CAM values [samples, timesteps]
        """
        if self.cam_model is None:
            # Construct CAM extraction model
            cnn_conv2_output = self.model.get_layer('cnn_conv2').output
            self.cam_model = keras.Model(
                inputs=self.model.input,
                outputs=cnn_conv2_output
            )
        
        cam_output = self.cam_model.predict(X)
        
        # Global Average Pooling to get CAM
        cam = np.mean(cam_output, axis=-1)
        
        return cam
    
    def extract_attention_weights(self, X):
        """
        Extracts Attention weights
        
        Output:
            Array of Attention weights [samples, timesteps]
        """
        if self.attention_model is None:
            # Construct Attention extraction model
            attention_layer = self.model.get_layer('attention')
            lstm_output = self.model.get_layer('lstm').output
            
            # Re-apply attention layer to get weights
            _, attention_weights = attention_layer(lstm_output)
            
            self.attention_model = keras.Model(
                inputs=self.model.input,
                outputs=attention_weights
            )
        
        attention_weights = self.attention_model.predict(X)
        
        return attention_weights
    
    def save(self, filepath):
        """Saves the model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Loads the model"""
        model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        # Create instance and set model
        instance = cls.__new__(cls)
        instance.model = model
        instance.cam_model = None
        instance.attention_model = None
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def summary(self):
        """Prints model summary"""
        return self.model.summary()


# Usage Example
if __name__ == "__main__":
    # Build Model
    model = ParallelCNNLSTMAttention(
        input_shape=(60, 10),
        cnn_filters=64,
        lstm_units=128,
        attention_units=64,
        dense_units=[64, 32]
    )
    
    model.summary()
    
    # Compile
    model.compile(optimizer='adam', loss='mse')
    
    # Mock Data
    X_train = np.random.randn(1000, 60, 10)
    y_train = np.random.randn(1000, 1)
    
    # Train
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)
    
    # Extract DLPs (Deep Learning Parameters)
    cam = model.extract_cam(X_train[:10])
    att = model.extract_attention_weights(X_train[:10])
    
    print(f"\nCAM shape: {cam.shape}")
    print(f"Attention weights shape: {att.shape}")