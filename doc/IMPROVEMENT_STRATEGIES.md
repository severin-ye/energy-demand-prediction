# 并行模型性能提升策略

## 📊 当前状态

| 模型 | MAE (验证) | MAE (测试) | 参数量 |
|------|-----------|-----------|-------|
| 串联Baseline | 0.0309 | - | ~30万 |
| 串联+Attention | 0.0292 | - | ~30万 |
| 并行(20步) | 0.0330 | 0.0323 | 35万 |
| 并行(80步) | 0.0345 | 0.0340 | 84万 |

**问题**：并行模型一直比baseline差，步长不是根本原因

---

## 🎯 可能的改进方向

### 1️⃣ **架构改进** ⭐⭐⭐⭐⭐

#### A. Bidirectional LSTM
```python
# 当前：单向LSTM
lstm_branch = LSTM(128, return_sequences=True)(cnn_branch)

# 改进：双向LSTM（论文可能用了这个）
lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(cnn_branch)
# 输出128维（64*2），参数量不变但效果更好
```
**优势**：能同时看到过去和未来的上下文
**成本**：训练时间增加约2倍

#### B. 多头注意力机制
```python
# 当前：单头注意力
# 改进：多头注意力（类似Transformer）
attention_output = MultiHeadAttention(
    num_heads=4, 
    key_dim=32
)(lstm_branch, lstm_branch)
```
**优势**：捕获不同类型的时序模式
**成本**：参数量增加

#### C. 残差连接
```python
# 在CNN和LSTM中添加skip connection
x = Conv1D(128, 3, padding='same')(inputs)
x = Add()([x, inputs_projected])  # 残差连接
```
**优势**：缓解梯度消失，帮助训练更深的网络
**成本**：实现复杂度增加

#### D. Batch Normalization
```python
x = Conv1D(64, 3, padding='same')(inputs)
x = BatchNormalization()(x)  # 添加BN
x = Activation('relu')(x)
```
**优势**：加速收敛，提高稳定性
**成本**：推理时略慢

---

### 2️⃣ **超参数调优** ⭐⭐⭐⭐

#### A. LSTM单元数
- 当前：128
- 建议尝试：**64, 96, 160, 192, 256**
- 论文中可能用了不同的值

#### B. CNN Filters
- 当前：[64, 128]
- 建议尝试：
  - **[32, 64]** - 更轻量
  - **[128, 256]** - 更强大
  - **[64, 128, 256]** - 三层CNN

#### C. Dropout率
- 当前：0.3
- 建议尝试：**0.1, 0.2, 0.4, 0.5**
- 过高的Dropout可能损害性能

#### D. Attention Units
- 当前：64
- 建议尝试：**32, 128** (与LSTM units匹配)

#### E. Dense层结构
- 当前：[64, 32]
- 建议尝试：
  - **[128, 64]** - 更强表达力
  - **[32]** - 更简单
  - **[128, 64, 32]** - 更深

---

### 3️⃣ **训练策略优化** ⭐⭐⭐⭐

#### A. 学习率调度
```python
lr_schedule = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```

#### B. 更激进的Early Stopping
- 当前：patience=15
- 建议：**patience=20-30**，给模型更多机会

#### C. 学习率调优
- 当前：0.001 (Adam默认)
- 建议尝试：**0.0005, 0.0003, 0.002**

#### D. Batch Size
- 当前：64
- 建议尝试：**32, 128, 256**
- 小batch可能提高泛化能力

#### E. 梯度裁剪
```python
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
```

---

### 4️⃣ **特征工程** ⭐⭐⭐

#### A. 时间特征
```python
# 添加周期性时间编码
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day / 7)
day_cos = np.cos(2 * np.pi * day / 7)
```

#### B. 滞后特征
```python
# 添加t-1, t-7, t-24的历史值
lag_features = [y[t-1], y[t-7], y[t-24]]
```

---

### 5️⃣ **损失函数改进** ⭐⭐⭐

#### A. Huber Loss
```python
# 对异常值更鲁棒
model.compile(loss='huber', ...)
```

#### B. 自定义加权损失
```python
# 对峰值时段的预测给予更高权重
def weighted_mse(y_true, y_pred):
    weights = 1 + tf.abs(y_true)  # 高值权重更大
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))
```

---

### 6️⃣ **数据预处理改进** ⭐⭐

#### A. 不同的归一化方法
- 当前：MinMaxScaler(0,1)
- 尝试：**StandardScaler** (均值0方差1)
- 尝试：**RobustScaler** (对异常值鲁棒)

#### B. 数据平滑
```python
# 移动平均平滑
df['smoothed'] = df['value'].rolling(3).mean()
```

---

## 🚀 推荐优先级

### 第一优先 - 快速见效 ⚡
1. **Bidirectional LSTM** - 论文很可能用了这个
2. **调整Dropout率** - 0.3可能太高
3. **减少Dense层** - 当前可能过拟合

### 第二优先 - 系统优化 🔧
4. **超参数网格搜索** - LSTM units, CNN filters
5. **学习率调度** - ReduceLROnPlateau
6. **Batch Normalization** - 加速收敛

### 第三优先 - 深度优化 🎨
7. **多头注意力** - 更强的特征捕获
8. **特征工程** - 时间编码
9. **损失函数** - Huber Loss

---

## 💡 我的建议

**最有可能的问题**：
1. ✅ **Bidirectional LSTM** - 论文没明说，但这是标准做法
2. ✅ **Dropout太高** - 0.3可能抑制了学习
3. ✅ **Dense层太深** - [64,32]可能过度正则化

**建议立即尝试**：
```python
# 改进方案组合
1. Bidirectional LSTM (128→64双向)
2. Dropout降低到0.2
3. Dense层简化为[64]
4. 添加Batch Normalization
```

这个组合应该能看到明显改进！要试试吗？
