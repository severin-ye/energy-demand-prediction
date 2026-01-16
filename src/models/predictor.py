"""
并行CNN-LSTM-Attention预测模型
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    自定义注意力机制层
    
    计算: score = tanh(W * h + b)
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
        输入: [batch, timesteps, features]
        输出: context_vector [batch, features], attention_weights [batch, timesteps]
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
    并行CNN-LSTM-Attention架构
    
    结构:
        - CNN分支: 1D卷积提取局部模式
        - LSTM-Attention分支: 长期依赖+注意力
        - 融合层: 拼接后MLP回归
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
        self.cam_model = None
        self.attention_model = None
    
    def _build_model(self):
        """构建并行架构"""
        # 输入
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ===== CNN分支 =====
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
        
        # 保存用于CAM提取
        cnn_features = layers.GlobalAveragePooling1D(name='cnn_gap')(cnn_branch)
        
        # ===== LSTM-Attention分支 =====
        lstm_branch = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            name='lstm'
        )(inputs)
        
        # 注意力层
        attention_output, attention_weights = AttentionLayer(
            units=self.attention_units,
            name='attention'
        )(lstm_branch)
        
        # ===== 融合 =====
        merged = layers.Concatenate(name='merge')([cnn_features, attention_output])
        
        # 全连接层
        x = merged
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
            x = layers.Dropout(0.3, name=f'dropout{i+1}')(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        # 构建模型
        model = keras.Model(inputs=inputs, outputs=outputs, name='ParallelCNNLSTMAttention')
        
        logger.info("模型构建完成")
        logger.info(f"参数量: {model.count_params():,}")
        
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=None):
        """编译模型"""
        if metrics is None:
            metrics = ['mae', 'mape']
        
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
    
    def extract_cam(self, X):
        """
        提取CAM (Class Activation Mapping)
        
        输出:
            CAM值数组 [样本数, 时间步]
        """
        if self.cam_model is None:
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
        if self.attention_model is None:
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
    
    def save(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载模型"""
        model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        # 创建实例并设置模型
        instance = cls.__new__(cls)
        instance.model = model
        instance.cam_model = None
        instance.attention_model = None
        
        logger.info(f"模型已从 {filepath} 加载")
        return instance
    
    def summary(self):
        """打印模型结构"""
        return self.model.summary()


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
