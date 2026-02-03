"""
基础模型类 - 提取所有模型的共同功能
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


class BaseTimeSeriesModel:
    """
    时间序列预测模型基类
    
    提供共同的功能：
    - CNN模块构建
    - 全连接层构建
    - 训练、预测、保存、加载接口
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
            cnn_filters: CNN卷积核数量（第一层）
            lstm_units: LSTM隐藏单元数
            attention_units: 注意力层单元数
            dense_units: 全连接层单元数列表
        """
        self.input_shape = input_shape
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        
        # 子类需要实现_build_model
        self.model = self._build_model()
    
    def _build_cnn_block(self, inputs, name_prefix='cnn'):
        """
        构建标准CNN块
        
        结构：
            Conv1D(64, 3) → MaxPool(2) → Conv1D(128, 3) → MaxPool(2)
        
        参数:
            inputs: 输入张量
            name_prefix: 层名称前缀
        
        返回:
            CNN处理后的张量（保持序列形式）
        """
        x = layers.Conv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            name=f'{name_prefix}_conv1'
        )(inputs)
        
        x = layers.MaxPooling1D(pool_size=2, name=f'{name_prefix}_pool1')(x)
        
        x = layers.Conv1D(
            filters=self.cnn_filters * 2,
            kernel_size=3,
            activation='relu',
            padding='same',
            name=f'{name_prefix}_conv2'
        )(x)
        
        x = layers.MaxPooling1D(pool_size=2, name=f'{name_prefix}_pool2')(x)
        
        return x
    
    def _build_dense_block(self, inputs, name_prefix='dense'):
        """
        构建标准全连接块
        
        结构：
            Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(1)
        
        参数:
            inputs: 输入张量
            name_prefix: 层名称前缀
        
        返回:
            输出张量
        """
        x = inputs
        
        # 隐藏层
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'{name_prefix}{i+1}')(x)
            x = layers.Dropout(0.3, name=f'dropout{i+1}')(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        return outputs
    
    def _build_model(self):
        """
        构建模型架构
        
        子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现_build_model方法")
    
    def compile(self, optimizer='adam', loss='mse', metrics=None):
        """编译模型"""
        if metrics is None:
            metrics = ['mae']
        
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
    def load(cls, filepath, custom_objects=None):
        """加载模型"""
        if custom_objects is None:
            custom_objects = {}
        
        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        
        instance = cls.__new__(cls)
        instance.model = model
        
        logger.info(f"模型已从 {filepath} 加载")
        return instance
    
    def summary(self):
        """打印模型结构"""
        return self.model.summary()
