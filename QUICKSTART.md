# ⚡ 快速开始指南

## 环境准备

### 1. 安装依赖

```bash
cd /home/severin/Codelib/YS

# 激活虚拟环境（如果已配置）
source .venv/bin/activate

# 安装所有依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 运行核心模块测试（约30秒）
python tests/test_core_modules.py
```

预期输出：
```
✅ All 5 core modules tested successfully!
```

---

## 使用方式

### 方式一：训练新模型

```python
import sys
sys.path.append('/home/severin/Codelib/YS')

from src.pipeline.train_pipeline import TrainPipeline
import pandas as pd
import numpy as np

# 1. 准备数据（示例：使用随机数据）
np.random.seed(42)
n_samples = 500

train_data = pd.DataFrame({
    'Temperature': np.random.randn(n_samples) * 10 + 20,  # 温度
    'Humidity': np.random.randn(n_samples) * 15 + 60,     # 湿度
    'WindSpeed': np.random.randn(n_samples) * 5 + 10,     # 风速
    'EDP': np.random.randn(n_samples) * 30 + 100           # 能耗
})

# 2. 创建训练流水线
pipeline = TrainPipeline(output_dir='./outputs')

# 3. 运行完整训练流程（9步）
results = pipeline.run(train_data)

# 4. 查看结果
print("\n训练完成！")
print(f"数据形状: {results['data_shapes']}")
print(f"状态分布: {results['state_distribution']}")
print(f"候选边数量: {len(results['candidate_edges'])}")
print(f"贝叶斯网络边: {results['bn_edges']}")
```

**输出目录结构**:
```
outputs/
├── models/
│   ├── preprocessor.pkl
│   ├── predictor.h5
│   ├── state_classifier.pkl
│   ├── discretizer.pkl
│   ├── cam_clusterer.pkl
│   ├── attention_clusterer.pkl
│   └── bayesian_network.bif
├── results/
│   ├── association_rules.csv
│   └── bayesian_network.png
└── config.json
```

---

### 方式二：使用已训练模型推理

```python
from src.pipeline.inference_pipeline import InferencePipeline
import pandas as pd
import numpy as np

# 1. 加载模型
pipeline = InferencePipeline(models_dir='./outputs/models')

# 2. 准备新数据（至少20行，因为sequence_length=20）
new_data = pd.DataFrame({
    'Temperature': [25.3] * 25,
    'Humidity': [62.5] * 25,
    'WindSpeed': [8.2] * 25
})

# 3. 单样本预测 + 生成建议
report = pipeline.predict_single(new_data, verbose=True)
print(report)
```

**输出示例**:
```
============================================================
           预测结果
============================================================
预测负荷: 125.50 kWh
负荷状态: Peak
CAM聚类: 1
注意力类型: Early


============================================================
           能源消耗预测与优化建议报告
============================================================

【当前状态】
  温度: High
  湿度: Medium
  风速: Low

【预测负荷】
  125.50 kWh

【优化建议】（共3条）

1. 建议降低室内温度设定，例如调低空调温度至Low℃左右，预计可显著降低高峰负荷概率（约23.5%）
   当前: 温度=High → 推荐: Low
   预期效果: 高峰概率从 68.2% 降至 44.7%

2. 建议提高除湿设备功率，但注意能耗平衡，预计可明显降低高峰负荷概率（约18.3%）
   当前: 湿度=Medium → 推荐: High
   预期效果: 高峰概率从 68.2% 降至 49.9%

3. ...

============================================================
注：以上建议基于因果贝叶斯网络推断，供参考。
```

---

### 方式三：批量预测（无解释）

```python
# 批量预测多个样本
results_df = pipeline.batch_predict(
    new_data,
    output_path='./outputs/predictions.csv'
)

print(results_df)
```

输出：
```
   Prediction  EDP_State  CAM_Cluster Attention_Type
0      125.50       Peak            1          Early
1      118.30     Normal            0          Other
2      132.70       Peak            2           Late
```

---

## 常用功能

### 1. 敏感性分析

```python
from src.models.bayesian_net import CausalBayesianNetwork
from src.inference.causal_inference import CausalInference

# 加载贝叶斯网络
bn = CausalBayesianNetwork()
bn.load_model('./outputs/models/bayesian_network.bif')

# 创建因果推断工具
ci = CausalInference(bn, target_var='EDP_State')

# 敏感性分析
features = ['Temperature', 'Humidity', 'WindSpeed']
sensitivity = ci.sensitivity_analysis(features, target_state='Peak')

print(sensitivity[['Feature', 'Prob_Range', 'Max_Value', 'Max_Prob']])
```

### 2. 龙卷风图可视化

```python
# 生成龙卷风图
ci.tornado_chart(
    top_k=5,
    output_path='./outputs/tornado_chart.png'
)
```

### 3. 反事实推断

```python
# 如果温度从High变为Low会怎样？
counterfactual = ci.counterfactual_analysis(
    actual_evidence={'Temperature': 'High', 'Humidity': 'Medium'},
    intervention={'Temperature': 'Low'},
    target_state='Peak'
)

print(counterfactual['interpretation'])
```

### 4. 平均因果效应

```python
# 计算温度High vs Low的平均因果效应
ace = ci.average_causal_effect(
    treatment_var='Temperature',
    treatment_value='High',
    control_value='Low',
    target_state='Peak'
)

print(f"ACE: {ace:.3f}")
```

---

## 使用真实数据

### UCI Household数据集示例

```python
# 下载并准备UCI数据
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

import pandas as pd

# 读取数据
df = pd.read_csv(
    'household_power_consumption.txt',
    sep=';',
    parse_dates={'Datetime': ['Date', 'Time']},
    na_values=['?']
)

# 预处理
df = df.dropna()
df = df.set_index('Datetime')

# 重采样为小时频率
df_hourly = df.resample('H').mean()

# 选择特征
train_data = df_hourly[['Global_active_power', 'Voltage', 'Global_intensity']].copy()
train_data.columns = ['EDP', 'Temperature', 'Humidity']  # 重命名为模型需要的列

# 添加风速特征（示例）
train_data['WindSpeed'] = np.random.randn(len(train_data)) * 5 + 10

# 训练
pipeline = TrainPipeline(output_dir='./outputs_uci')
results = pipeline.run(train_data[:5000])  # 使用前5000条数据
```

---

## 配置自定义参数

```python
# 自定义配置
custom_config = {
    'sequence_length': 30,  # 改为30时间步
    'epochs': 100,          # 训练100轮
    'batch_size': 64,       # 批大小64
    'n_states': 5,          # 5个状态分类
    'state_names': ['VeryLow', 'Low', 'Normal', 'High', 'VeryHigh'],
    'min_support': 0.03,    # 更低的支持度阈值
}

pipeline = TrainPipeline(
    config=custom_config,
    output_dir='./outputs_custom'
)
```

---

## 故障排查

### 问题1：CUDA警告
```
UserWarning: CUDA not available
```
**解决**: 这是正常的，代码会自动使用CPU。如果有GPU，安装`tensorflow-gpu`。

### 问题2：模块导入错误
```
ModuleNotFoundError: No module named 'src'
```
**解决**: 确保运行代码前添加了：
```python
import sys
sys.path.append('/home/severin/Codelib/YS')
```

### 问题3：数据形状错误
```
ValueError: Input data must have at least 20 rows
```
**解决**: 数据至少需要`sequence_length`行（默认20行）。

### 问题4：模型文件未找到
```
FileNotFoundError: [Errno 2] No such file or directory: './outputs/models/predictor.h5'
```
**解决**: 先运行训练流水线生成模型文件。

---

## 性能优化建议

1. **GPU加速**: 安装`tensorflow-gpu`可将训练速度提升10-100倍
2. **批大小**: 增大`batch_size`可提升训练速度（需要更多内存）
3. **序列长度**: 减小`sequence_length`可减少计算量
4. **并行处理**: 使用`joblib`并行处理多个样本

---

## 下一步学习

1. 阅读 [实现文档.md](doc/实现文档.md) 了解算法细节
2. 查看 [项目设计文档.md](doc/项目设计文档.md) 理解系统架构
3. 阅读论文PDF了解理论背景
4. 修改代码适配自己的数据集

---

**更多帮助**: 查看各模块的文档字符串（`help(module)`）或源代码注释

**最后更新**: 2026-01-16
