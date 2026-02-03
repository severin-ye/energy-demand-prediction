"""
并行CNN-LSTM-Attention预测模型
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
    
    其中:
    - o_n: LSTM在时间步n的输出
    - h_N: LSTM的最终隐藏状态
    - fc(o_n, h_N): 计算o_n与h_N的相关性得分
    """
    
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # input_shape: [batch, timesteps, features]
        feature_dim = input_shape[-1]
        
        # 用于计算fc(o_n, h_N)的权重矩阵
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
        """
        输入: [batch, timesteps, features] - LSTM所有时间步的输出
        输出: context_vector [batch, features], attention_weights [batch, timesteps]
        """
        # 提取最终隐藏状态 h_N
        h_final = inputs[:, -1, :]  # [batch, features]
        
        # 计算fc(o_n, h_N) = v^T * tanh(W_o*o_n + W_h*h_N + b)
        # o_n: inputs [batch, timesteps, features]
        # h_N: h_final [batch, features]
        
        # 扩展h_final维度以便广播
        h_final_expanded = tf.expand_dims(h_final, 1)  # [batch, 1, features]
        h_final_tiled = tf.tile(h_final_expanded, [1, tf.shape(inputs)[1], 1])  # [batch, timesteps, features]
        
        # W_o * o_n
        score_o = tf.tensordot(inputs, self.W_o, axes=[[2], [0]])  # [batch, timesteps, units]
        
        # W_h * h_N
        score_h = tf.tensordot(h_final_tiled, self.W_h, axes=[[2], [0]])  # [batch, timesteps, units]
        
        # fc(o_n, h_N) = v^T * tanh(W_o*o_n + W_h*h_N + b)
        score = tf.nn.tanh(score_o + score_h + self.b)  # [batch, timesteps, units]
        score = tf.tensordot(score, self.v, axes=[[2], [0]])  # [batch, timesteps]
        
        # 计算注意力权重 (论文公式1)
        attention_weights = tf.nn.softmax(score, axis=1)  # [batch, timesteps]
        
        # 计算上下文向量 (论文公式2): c_N = Σ a_k · h_k
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1),
            axis=1
        )  # [batch, features]
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class ParallelCNNLSTMAttention(BaseTimeSeriesModel):
    """
    并行CNN-LSTM-Attention架构（论文方法）
    
    结构:
        - CNN分支: 从原始输入提取局部模式
        - LSTM-Attention分支: 从原始输入提取长期依赖
        - 融合层: 拼接两路特征后MLP回归
    
    特点:
        - CNN和LSTM并行处理原始输入
        - 双路特征互补融合
        - 参数量较大但特征表达能力强
    """
    
    def _build_model(self):
        """构建并行架构"""
        # 输入
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ===== CNN分支（从原始输入提取局部特征）=====
        cnn_branch = self._build_cnn_block(inputs)
        
        # 方案1：保留CNN的完整特征（避免信息抽象损失）
        # Flatten展平CNN输出，直接使用所有特征（论文强调"减少信息抽象损失"）
        cnn_features = layers.Flatten(name='cnn_flatten')(cnn_branch)
        # 移除原来的 Dense(128) 压缩层 - 这是导致信息损失的关键问题
        
        # ===== LSTM-Attention分支（从原始输入提取长期依赖）=====
        lstm_branch = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            name='lstm'
        )(inputs)  # 注意：输入是inputs而不是cnn_branch
        
        # 注意力层
        attention_output, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_branch)
        
        # ===== 特征融合 =====
        merged = layers.Concatenate(name='merge')([cnn_features, attention_output])
        
        # 全连接层（使用父类方法）
        outputs = self._build_dense_block(merged)
        
        # 构建模型
        model = keras.Model(inputs=inputs, outputs=outputs, name='ParallelCNNLSTMAttention')
        
        logger.info("并行CNN-LSTM-Attention模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    @classmethod
    def load(cls, filepath):
        """加载模型（需要注册自定义层）"""
        return super().load(filepath, custom_objects={'AttentionLayer': AttentionLayer})
    
    def extract_cam(self, X):
        """
        提取CAM (Class Activation Mapping)
        
        输出:
            CAM值数组 [样本数, 时间步]
        """
        if not hasattr(self, 'cam_model') or self.cam_model is None:
            # 构建CAM提取模型
            cnn_conv2_output = self.model.get_layer('cnn_conv2').output
            self.cam_model = keras.Model(
                inputs=self.model.input,
                outputs=cnn_conv2_output
            )
        
        cam_output = self.cam_model.predict(X)
        
        # 全局平均池化得到CAM
        cam = np.mean(cam_output, axis=-1)
        
        return cam
    
    def extract_attention_weights(self, X):
        """
        提取Attention权重
        
        输出:
            Attention权重数组 [样本数, 时间步]
        """
        if not hasattr(self, 'attention_model') or self.attention_model is None:
            # 构建Attention提取模型
            attention_layer = self.model.get_layer('attention')
            lstm_output = self.model.get_layer('lstm').output
            
            # 重新应用attention层获取权重
            _, attention_weights = attention_layer(lstm_output)
            
            self.attention_model = keras.Model(
                inputs=self.model.input,
                outputs=attention_weights
            )
        
        attention_weights = self.attention_model.predict(X)
        
        return attention_weights


# 使用示例
if __name__ == "__main__":
    # 构建模型
    model = ParallelCNNLSTMAttention(
        input_shape=(60, 10),
        cnn_filters=64,
        lstm_units=128,
        attention_units=64,
        dense_units=[64, 32]
    )
    
    model.summary()
    
    # 编译
    model.compile(optimizer='adam', loss='mse')
    
    # 模拟数据
    X_train = np.random.randn(1000, 60, 10)
    y_train = np.random.randn(1000, 1)
    
    # 训练
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)
    
    # 提取DLP
    cam = model.extract_cam(X_train[:10])
    att = model.extract_attention_weights(X_train[:10])
    
    print(f"\nCAM shape: {cam.shape}")
    print(f"Attention weights shape: {att.shape}")
