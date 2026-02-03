"""
串联和并行CNN-LSTM模型（用于公平对比）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    自定义注意力机制层（基于论文公式1-2）
    
    论文公式:
    a_n = exp(fc(o_n, h_N)) / Σ_k exp(fc(o_k, h_N))
    c_N = Σ_k a_k · h_k
    """
    
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        self.W_o = self.add_weight(
            name='attention_W_o',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_h = self.add_weight(
            name='attention_W_h',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.v = self.add_weight(
            name='attention_v',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        h_final = inputs[:, -1, :]
        h_final_expanded = tf.expand_dims(h_final, 1)
        h_final_tiled = tf.tile(h_final_expanded, [1, tf.shape(inputs)[1], 1])
        
        score_o = tf.tensordot(inputs, self.W_o, axes=[[2], [0]])
        score_h = tf.tensordot(h_final_tiled, self.W_h, axes=[[2], [0]])
        
        score = tf.nn.tanh(score_o + score_h + self.b)
        score = tf.tensordot(score, self.v, axes=[[2], [0]])
        
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1),
            axis=1
        )
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class SerialCNNLSTM(BaseTimeSeriesModel):
    """
    串联CNN-LSTM架构（论文baseline）
    
    结构：Input → CNN → LSTM → Dense → Output
    
    特点：
        - CNN特征提取后送入LSTM
        - 简单串联结构
        - 参数量较少
    """
    
    def _build_model(self):
        """构建串联架构"""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN特征提取
        cnn_output = self._build_cnn_block(inputs)
        # 输出: (batch, 20, 128)
        
        # LSTM时序建模
        lstm_output = layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,  # 只返回最后一个时间步
            name='lstm'
        )(cnn_output)
        # 输出: (batch, 128)
        
        # 全连接层
        outputs = self._build_dense_block(lstm_output)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SerialCNNLSTM')
        
        logger.info("串联CNN-LSTM模型构建完成 (baseline)")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    @classmethod
    def load(cls, filepath):
        return super().load(filepath)


class SerialCNNLSTMAttention(BaseTimeSeriesModel):
    """
    串联CNN-LSTM-Attention架构
    
    结构：Input → CNN → LSTM → Attention → Dense → Output
    
    特点：
        - CNN特征提取后送入LSTM
        - 添加Attention机制捕获重要时间步
        - 参数量适中
    """
    
    def _build_model(self):
        """构建串联+Attention架构"""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN特征提取
        cnn_output = self._build_cnn_block(inputs)
        # 输出: (batch, 20, 128)
        
        # LSTM时序建模
        lstm_output = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,  # 返回所有时间步
            name='lstm'
        )(cnn_output)
        # 输出: (batch, 20, 128)
        
        # Attention机制
        attention_output, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_output)
        # 输出: (batch, 128)
        
        # 全连接层
        outputs = self._build_dense_block(attention_output)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SerialCNNLSTMAttention')
        
        logger.info("串联CNN-LSTM-Attention模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    @classmethod
    def load(cls, filepath):
        return super().load(filepath, custom_objects={'AttentionLayer': AttentionLayer})


class ParallelCNNLSTMAttention(BaseTimeSeriesModel):
    """
    并行CNN-LSTM-Attention架构（论文方法 - 严格按照Fig. 1）
    
    结构：
        Input → CNN → Flatten ──┐
          ↓                      ├→ Concat → Dense → Output
          └→ LSTM → Attention ──┘
    
    特点：
        - CNN和LSTM真正并行：LSTM从原始输入开始
        - 两个分支互补：CNN捕获局部模式，LSTM捕获长期依赖
        - 论文中性能最好的架构
    """
    
    def _build_model(self):
        """构建真正的并行架构（严格按照论文Fig. 1）"""
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ===== 分支1：CNN → Flatten =====
        # CNN提取局部空间特征
        cnn_branch = self._build_cnn_block(inputs)
        # CNN输出: (batch, 20, 128)
        cnn_features = layers.Flatten(name='cnn_flatten')(cnn_branch)
        # Flatten后: (batch, 2560)
        
        # ===== 分支2：LSTM → Attention =====
        # 关键：LSTM从原始输入开始，保留完整的80步时序信息
        lstm_branch = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            name='lstm'
        )(inputs)  # ← 关键：从inputs开始，不是cnn_branch！
        # LSTM输出: (batch, 80, 128) - 保持完整序列长度
        
        # Attention机制
        attention_output, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_branch)
        # Attention输出: (batch, 128)
        
        # ===== 特征融合 =====
        merged = layers.Concatenate(name='merge')([cnn_features, attention_output])
        # 融合后: (batch, 2688) = 2560 + 128
        
        # 全连接层
        outputs = self._build_dense_block(merged)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ParallelCNNLSTMAttention')
        
        logger.info("并行CNN-LSTM-Attention模型构建完成 (严格按照论文Fig. 1)")
        logger.info(f"架构: LSTM从原始80步输入开始，保持真正的并行互补")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    @classmethod
    def load(cls, filepath):
        return super().load(filepath, custom_objects={'AttentionLayer': AttentionLayer})
