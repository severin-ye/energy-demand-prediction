"""
Baseline模型：串联CNN-LSTM（用于对比实验）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SerialCNNLSTM:
    """
    串联CNN-LSTM架构（对照组）
    
    结构:
        Input → CNN → LSTM → Dense → Output
    
    与并行架构的区别:
        - CNN输出直接送入LSTM（串行处理）
        - 无并行特征提取
        - 无注意力机制
    """
    
    def __init__(self,
                 input_shape: tuple,
                 cnn_filters: int = 64,
                 lstm_units: int = 128,
                 dense_units: list = [64, 32]):
        """
        参数:
            input_shape: (sequence_length, n_features)
            cnn_filters: CNN卷积核数量
            lstm_units: LSTM隐藏单元数
            dense_units: 全连接层单元数列表
        """
        self.input_shape = input_shape
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        
        self.model = self._build_model()
    
    def _build_model(self):
        """构建串联架构"""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN部分
        x = layers.Conv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='cnn_conv1'
        )(inputs)
        
        x = layers.MaxPooling1D(pool_size=2, name='cnn_pool1')(x)
        
        x = layers.Conv1D(
            filters=self.cnn_filters * 2,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='cnn_conv2'
        )(x)
        
        x = layers.MaxPooling1D(pool_size=2, name='cnn_pool2')(x)
        
        # LSTM部分
        x = layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            name='lstm'
        )(x)
        
        # 全连接层
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
            x = layers.Dropout(0.3, name=f'dropout{i+1}')(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SerialCNNLSTM')
        
        logger.info("串联CNN-LSTM模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=None):
        """编译模型"""
        if metrics is None:
            metrics = ['mae', 'mse']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, X_train, y_train, validation_data=None, **kwargs):
        """训练模型"""
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            **kwargs
        )
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载模型"""
        model = keras.models.load_model(filepath)
        
        instance = cls.__new__(cls)
        instance.model = model
        
        logger.info(f"模型已从 {filepath} 加载")
        return instance
    
    def summary(self):
        """打印模型结构"""
        return self.model.summary()


class SerialCNNLSTMAttention:
    """
    串联CNN-LSTM-Attention架构（对照组）
    
    结构:
        Input → CNN → LSTM-Attention → Dense → Output
    
    与并行架构的区别:
        - CNN输出直接送入LSTM（串行处理）
        - 有注意力机制但无并行提取
    """
    
    def __init__(self,
                 input_shape: tuple,
                 cnn_filters: int = 64,
                 lstm_units: int = 128,
                 attention_units: int = 64,
                 dense_units: list = [64, 32]):
        """
        参数:
            input_shape: (sequence_length, n_features)
            cnn_filters: CNN卷积核数量
            lstm_units: LSTM隐藏单元数
            attention_units: 注意力层单元数
            dense_units: 全连接层单元数列表
        """
        self.input_shape = input_shape
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        
        self.model = self._build_model()
    
    def _build_model(self):
        """构建串联架构"""
        from .predictor import AttentionLayer
        
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN部分
        x = layers.Conv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='cnn_conv1'
        )(inputs)
        
        x = layers.MaxPooling1D(pool_size=2, name='cnn_pool1')(x)
        
        x = layers.Conv1D(
            filters=self.cnn_filters * 2,
            kernel_size=3,
            activation='relu',
            padding='same',
            name='cnn_conv2'
        )(x)
        
        x = layers.MaxPooling1D(pool_size=2, name='cnn_pool2')(x)
        
        # LSTM部分
        lstm_out = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            name='lstm'
        )(x)
        
        # 注意力层
        context, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_out)
        
        # 全连接层
        x = context
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
            x = layers.Dropout(0.3, name=f'dropout{i+1}')(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SerialCNNLSTMAttention')
        
        logger.info("串联CNN-LSTM-Attention模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=None):
        """编译模型"""
        if metrics is None:
            metrics = ['mae', 'mse']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, X_train, y_train, validation_data=None, **kwargs):
        """训练模型"""
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            **kwargs
        )
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载模型"""
        from .predictor import AttentionLayer
        model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        instance = cls.__new__(cls)
        instance.model = model
        
        logger.info(f"模型已从 {filepath} 加载")
        return instance
    
    def summary(self):
        """打印模型结构"""
        return self.model.summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试串联模型
    print("\n=== 串联CNN-LSTM ===")
    model1 = SerialCNNLSTM(
        input_shape=(80, 10),
        cnn_filters=64,
        lstm_units=128,
        dense_units=[64, 32]
    )
    model1.summary()
    
    print("\n=== 串联CNN-LSTM-Attention ===")
    model2 = SerialCNNLSTMAttention(
        input_shape=(80, 10),
        cnn_filters=64,
        lstm_units=128,
        attention_units=64,
        dense_units=[64, 32]
    )
    model2.summary()
    
    # 对比参数量
    from .predictor import ParallelCNNLSTMAttention
    model3 = ParallelCNNLSTMAttention(
        input_shape=(80, 10),
        cnn_filters=64,
        lstm_units=128,
        attention_units=64,
        dense_units=[64, 32]
    )
    
    print("\n=== 参数量对比 ===")
    print(f"串联CNN-LSTM: {model1.model.count_params():,}")
    print(f"串联CNN-LSTM-Attention: {model2.model.count_params():,}")
    print(f"并行CNN-LSTM-Attention: {model3.model.count_params():,}")
