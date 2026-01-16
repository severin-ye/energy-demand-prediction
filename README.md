# 能源需求预测的因果可解释AI系统

基于论文 *"Causally explainable artificial intelligence on deep learning model for energy demand prediction"* (Erlangga & Cho, 2025) 的完整代码复现。

## 项目简介

本项目实现了一个结合深度学习预测和因果解释的能源需求预测系统：

- **预测模块**: 并行CNN-LSTM-Attention架构，实现高精度能源需求预测
- **解释模块**: 贝叶斯网络结合深度学习参数(DLP)，提供稳定的因果解释
- **推荐模块**: 基于因果推断生成可操作的节能建议

## 核心特性

✅ **高性能预测**: 相比串行架构提升34.84% (UCI) 和 13.63% (REFIT)  
✅ **稳定解释**: 余弦相似度达0.999+（SHAP仅0.95-0.96）  
✅ **因果推理**: 基于领域知识约束的贝叶斯网络  
✅ **可操作建议**: 针对Peak/Normal/Lower状态生成具体推荐

## 项目结构

```
├── data/                   # 数据存储
├── src/                    # 源代码
│   ├── preprocessing/      # 数据预处理
│   ├── models/            # 预测和解释模型
│   ├── inference/         # 因果推断和推荐
│   └── pipeline/          # 训练和推理流水线
├── tests/                 # 单元测试
├── outputs/               # 输出结果
├── notebooks/             # Jupyter实验
├── config/                # 配置文件
└── doc/                   # 项目文档
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

下载UCI Household数据集或REFIT数据集，放置到 `data/` 目录。

### 3. 训练模型

```python
from src.pipeline.train_pipeline import TrainingPipeline

config = {
    'data_path': 'data/household_power_consumption.txt',
    'output_dir': 'outputs/training',
    'sequence_length': 60,
    'epochs': 100,
    'batch_size': 64
}

pipeline = TrainingPipeline(config)
pipeline.run()
```

### 4. 推理预测

```python
from src.pipeline.inference_pipeline import InferencePipeline
import pandas as pd

# 加载模型
pipeline = InferencePipeline(models_dir='outputs/training/models')

# 准备输入
test_data = pd.DataFrame({
    'Date': ['2025-06-15 14:30:00'],
    'GlobalActivePower': [4.5],
    'Kitchen': [2.0],
    'ClimateControl': [3.5]
})

# 推理
result = pipeline.predict(test_data)

print(f"预测值: {result['prediction']['value']:.4f}")
print(f"状态: {result['prediction']['state']}")
print(result['recommendation_text'])
```

## 技术架构

### 预测模型
- **并行架构**: CNN分支 + LSTM-Attention分支
- **特征提取**: 时间序列滑动窗口 + 时间特征工程
- **稳健分类**: Sn尺度估计器处理异常值

### 解释模型
- **DLP聚类**: CAM和Attention权重聚类
- **关联规则**: Apriori算法挖掘候选因果关系
- **贝叶斯网络**: 领域知识约束的结构学习

### 因果推断
- **Do-演算**: 计算干预效应
- **敏感性分析**: Tornado图可视化
- **反事实分析**: 对比事实与反事实分布

## 性能指标

### 预测性能（vs 串行CNN-LSTM）
| 数据集 | MSE改进 | MAPE改进 |
|--------|---------|----------|
| UCI    | 34.84%  | 32.71%   |
| REFIT  | 13.63%  | 11.45%   |

### 解释一致性（余弦相似度）
| 方法      | UCI数据集 | REFIT数据集 |
|-----------|-----------|-------------|
| 本方法(BN) | 0.99940   | 0.99983     |
| SHAP      | 0.95210   | 0.96478     |

## 参考文献

Gatum Erlangga, Sung-Bae Cho. *Causally explainable artificial intelligence on deep learning model for energy demand prediction*. Engineering Applications of Artificial Intelligence, Volume 162, 2025.

## 开发进度

- [x] 项目设计文档
- [x] 实现文档编写
- [x] 项目目录结构
- [ ] 数据预处理模块
- [ ] 预测模型实现
- [ ] 解释模型实现
- [ ] 因果推断实现
- [ ] 流水线集成
- [ ] 单元测试
- [ ] 模型训练与评估

## 许可证

MIT License

## 作者

AI Agent - 基于原始论文的代码复现
