"""
Baseline模型：串联CNN-LSTM（用于对比实验）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging

from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


class SerialCNNLSTM(BaseTimeSeriesModel):
    """
    串联CNN-LSTM架构（对照组）
    
    结构:
        Input → CNN → LSTM → Dense → Output
    
    特点:
        - CNN输出直接送入LSTM（串行处理）
        - 无并行特征提取
        - 无注意力机制
    """
    
    def _build_model(self):
        """构建串联架构"""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN部分（使用父类方法）
        cnn_out = self._build_cnn_block(inputs)
        
        # LSTM部分（return_sequences=False，只返回最后时间步）
        lstm_out = layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            name='lstm'
        )(cnn_out)
        
        # 全连接层（使用父类方法）
        outputs = self._build_dense_block(lstm_out)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SerialCNNLSTM')
        
        logger.info("串联CNN-LSTM模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model


class SerialCNNLSTMAttention(BaseTimeSeriesModel):
    """
    串联CNN-LSTM-Attention架构（对照组）
    
    结构:
        Input → CNN → LSTM-Attention → Dense → Output
    
    特点:
        - CNN输出直接送入LSTM（串行处理）
        - 有注意力机制但无并行提取
    """
    
    def _build_model(self):
        """构建串联+注意力架构"""
        from .predictor import AttentionLayer
        
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN部分（使用父类方法）
        cnn_out = self._build_cnn_block(inputs)
        
        # LSTM部分（return_sequences=True，返回所有时间步）
        lstm_out = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            name='lstm'
        )(cnn_out)
        
        # 注意力层
        context, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_out)
        
        # 全连接层（使用父类方法）
        outputs = self._build_dense_block(context)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SerialCNNLSTMAttention')
        
        logger.info("串联CNN-LSTM-Attention模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model

    @classmethod
    def load(cls, filepath):
        """加载模型（需要注册自定义层）"""
        from .predictor import AttentionLayer
        return super().load(filepath, custom_objects={'AttentionLayer': AttentionLayer})