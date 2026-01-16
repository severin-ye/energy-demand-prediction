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
✅ **HTML可视化**: 精美的10步推理流程可视化报告

## 📁 项目结构

```
YS/
├── src/                      # 源代码
│   ├── preprocessing/        # 数据预处理
│   ├── models/              # 核心模型
│   │   ├── predictor.py     # CNN-LSTM-Attention预测器
│   │   ├── state_classifier.py  # 状态分类器
│   │   ├── discretizer.py   # 特征离散化
│   │   ├── clustering.py    # DLP特征聚类
│   │   ├── association.py   # 关联规则挖掘
│   │   └── bayesian_net.py  # 贝叶斯网络
│   ├── inference/           # 推理模块
│   │   ├── causal_inference.py  # 因果推断
│   │   └── recommendation.py    # 智能推荐
│   ├── pipeline/            # 流水线
│   │   ├── train_pipeline.py    # 训练流水线
│   │   └── inference_pipeline.py # 推理流水线
│   ├── data_processing/     # 数据处理
│   │   ├── uci_loader.py    # UCI数据加载器
│   │   └── data_splitter.py # 数据集分割
│   └── visualization/       # 可视化
│       └── inference_visualizer.py  # HTML报告生成
│
├── scripts/                  # 脚本工具
│   ├── download_uci_data.py # UCI数据下载
│   ├── split_uci_dataset.py # 数据集分割
│   ├── run_training.py      # 训练脚本
│   └── run_inference_uci.py # 推理脚本
│
├── data/                     # 数据目录
│   ├── uci/                 # UCI数据集
│   │   ├── raw/             # 原始数据
│   │   ├── processed/       # 预处理后
│   │   └── splits/          # 训练/测试集
│   └── synthetic/           # 合成数据
│
├── outputs/                  # 输出结果
│   ├── training_uci/        # 训练输出
│   │   ├── models/          # 保存的模型
│   │   └── results/         # 训练结果
│   └── inference_uci/       # 推理输出
│       └── html_reports/    # HTML可视化报告
│
├── doc/                      # 文档目录
│   ├── guides/              # 使用指南
│   │   ├── QUICKSTART.md    # 快速开始
│   │   ├── QUICK_REFERENCE.md  # 快速参考
│   │   └── HTML_DEMO.md     # HTML可视化演示
│   ├── summaries/           # 进度总结
│   └── ChatGPT-详细整理论文.md  # 论文解读
│
├── tests/                    # 测试代码
├── logs/                     # 训练日志
└── README.md                 # 本文件
```

## 快速开始

### 📖 详细指南

- **[快速开始指南](doc/guides/QUICKSTART.md)** - 完整的安装和使用教程
- **[快速参考](doc/guides/QUICK_REFERENCE.md)** - 常用命令速查表
- **[HTML可视化演示](doc/guides/HTML_DEMO.md)** - 推理流程可视化使用指南

### ⚡ 30秒快速启动

```bash
# 1. 环境配置
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 下载UCI数据集（约127MB）
python scripts/download_uci_data.py --method direct --preprocess

# 3. 分割数据集（95%训练/5%测试）
python scripts/split_uci_dataset.py --test-ratio 0.05

# 4. 训练模型（约5分钟，CPU）
python scripts/run_training.py \
  --data data/uci/splits/train.csv \
  --epochs 20 \
  --batch-size 64

# 5. 推理测试（生成HTML可视化）
python scripts/run_inference_uci.py --n-samples 100

# 6. 查看HTML报告
./view_html_reports.sh
```

### 📊 UCI数据集完整流程

**数据准备**
```bash
# 下载并预处理UCI数据
python scripts/download_uci_data.py --method direct --preprocess

# 查看数据集信息
python scripts/download_uci_data.py --info

# 分割为训练集和测试集
python scripts/split_uci_dataset.py \
  --input data/uci/processed/uci_household_clean.csv \
  --output-dir data/uci/splits \
  --test-ratio 0.05
```

**模型训练**
```bash
# 使用UCI数据训练（自动检测数据类型）
python scripts/run_training.py \
  --data data/uci/splits/train.csv \
  --epochs 20 \
  --batch-size 64 \
  --output-dir outputs/training_uci

# 训练完成后查看报告
cat outputs/training_uci/TRAINING_REPORT.md
```

**推理测试**
```bash
# 运行推理并生成HTML可视化
python scripts/run_inference_uci.py \
  --model-dir outputs/training_uci/models \
  --test-data data/uci/splits/test.csv \
  --n-samples 100 \
  --output-dir outputs/inference_uci

## 🎨 HTML可视化推理报告

系统会自动为每个推理样本生成精美的HTML可视化报告，展示完整的10步推理流程：

```
⓪ 📊 原始数据输入
① 🔍 短期模式分析 (CNN)
② 📈 长期趋势分析 (LSTM)
③ ⏰ 关键时间判断 (Attention)
④ 🎯 综合判断与预测
⑤ 🚦 负荷状态分类
⑥ 🔤 特征等级化
⑦ 🧠 模型内部感知
⑧ 🔗 因果关系推断
⑨ 🔮 反事实分析
⑩ ✨ 智能建议输出
```

**特点**：
- 🎨 精美的渐变设计
- 💡 全中文通俗解释
- 📊 可视化图表
- 🔍 每步都有"为什么"

详见 **[HTML可视化演示指南](doc/guides/HTML_DEMO.md)**

## 📚 文档索引

> **完整文档导航**: [doc/INDEX.md](doc/INDEX.md) - 所有文档的快速导航和说明

### 使用指南
- **[快速开始](doc/guides/QUICKSTART.md)** - 详细的安装、配置和使用教程
- **[快速参考](doc/guides/QUICK_REFERENCE.md)** - 常用命令和参数速查
- **[HTML可视化](doc/guides/HTML_DEMO.md)** - 推理流程可视化使用说明

### 技术文档
- **[论文详解](doc/ChatGPT-详细整理论文.md)** - 论文完整解读（教学式）
- **[项目设计](doc/项目设计文档.md)** - 系统架构设计文档
- **[实现文档](doc/实现文档.md)** - 代码实现说明
- **[UCI数据集](doc/数据集说明-UCI_Household.md)** - UCI数据集详细说明
- **[项目结构](PROJECT_STRUCTURE.md)** - 完整项目结构说明

### 进度总结
- **[实现总结](doc/summaries/IMPLEMENTATION_SUMMARY.md)** - 实现进度汇总
- **[项目进度](doc/summaries/PROGRESS_SUMMARY.md)** - 整体进度追踪

### 输出报告
- **[训练报告](outputs/training_uci/TRAINING_REPORT.md)** - UCI数据训练结果
- **[推理摘要](outputs/inference_uci/INFERENCE_SUMMARY.md)** - 推理测试结果摘要
- **[HTML报告](outputs/inference_uci/html_reports/index.html)** - 可视化推理报告

## 核心模块说明

### 1. 预测模块 (`src/models/predictor.py`)
- 并行CNN-LSTM架构
- Attention机制
- 提取CAM和Attention特征（DLP）

### 2. 状态分类 (`src/models/state_classifier.py`)
- 将连续预测值分类为 Lower/Normal/Peak
- 基于聚类的动态阈值

### 3. 特征离散化 (`src/models/discretizer.py`)
- 将连续特征转换为离散等级
- 支持因果推理

### 4. DLP聚类 (`src/models/clustering.py`)
- CAM特征聚类（K-Means）
- Attention类型分类（Early/Late/Other）

### 5. 关联规则 (`src/models/association.py`)
- Apriori算法挖掘EDP规则
- 为贝叶斯网络提供先验知识

### 6. 贝叶斯网络 (`src/models/bayesian_net.py`)
- 结构学习（Hill-Climbing）
- 参数估计（Maximum Likelihood）
- 领域知识约束

### 7. 因果推断 (`src/inference/causal_inference.py`)
- 基于贝叶斯网络的因果推理
- 反事实查询
- 敏感性分析

### 8. 智能推荐 (`src/inference/recommendation.py`)
- 基于因果推断生成建议
- 针对不同状态的个性化推荐

## 训练流水线

完整的9步训练流程：

1. **数据预处理** - 清洗、归一化、时间特征提取
2. **预测器训练** - CNN-LSTM-Attention模型训练
3. **状态分类器** - 基于预测值的状态聚类
4. **特征离散化** - KBinsDiscretizer训练
5. **DLP聚类** - CAM和Attention特征聚类
6. **关联规则挖掘** - Apriori算法提取规则
7. **贝叶斯网络学习** - 结构学习和参数估计
8. **因果推断初始化** - 创建推理引擎
9. **模型保存** - 保存所有训练好的模型

## 推理流水线

完整的推理流程：

1. **加载模型** - 加载训练好的所有模型
2. **数据预处理** - 与训练时相同的预处理
3. **预测** - 使用CNN-LSTM-Attention生成预测
4. **提取DLP** - 提取CAM和Attention特征
5. **状态分类** - 分类为Lower/Normal/Peak
6. **特征离散化** - 转换为离散等级
7. **因果推断** - 基于贝叶斯网络推理
8. **生成建议** - 基于当前状态生成推荐
9. **HTML可视化** - 生成精美的可视化报告

## 性能指标

### UCI数据集测试结果

| 指标 | 数值 |
|------|------|
| MAE | 0.6718 kW |
| RMSE | 0.8460 kW |
| 样本数 | 80 |
| 训练时间 | ~5分钟（CPU，20 epochs）|
| 推理速度 | ~1秒/样本 |

### 模型规模

| 组件 | 参数量/大小 |
|------|------------|
| Predictor | 58,867 参数 |
| State Classifier | ~1KB |
| Discretizer | ~2KB |
| CAM Clusterer | ~5KB |
| Attention Clusterer | ~3KB |
| Bayesian Network | ~50KB |
| **总计** | **~2.3MB** |

## 开发工具

### 测试
```bash
# 运行核心模块测试
python tests/test_core_modules.py

# 运行完整测试套件
pytest tests/
```

### 数据生成
```bash
# 生成合成数据（用于开发测试）
python scripts/generate_synthetic_data.py \
  --n-samples 10000 \
  --scenario heatwave \
  --output data/synthetic/scenario_heatwave.csv
```

### 数据集对比
```bash
# 对比UCI和合成数据
python scripts/compare_datasets.py
```

## 常见问题

### Q: 为什么只生成10个HTML报告？
A: 为了平衡速度和实用性。可以在 `scripts/run_inference_uci.py` 中修改 `num_samples` 参数。

### Q: 训练时间太长怎么办？
A: 减少epochs（如 `--epochs 10`）或增加batch size（如 `--batch-size 128`）。

### Q: 如何使用GPU加速？
A: 安装GPU版本的TensorFlow：`pip install tensorflow-gpu`

### Q: 数据集太大，内存不足？
A: 使用 `--n-samples` 参数限制样本数量。

## 参考文献

1. Erlangga, D., & Cho, S. (2025). Causally explainable artificial intelligence on deep learning model for energy demand prediction. *Applied Energy*.

2. UCI Machine Learning Repository. (2012). Individual Household Electric Power Consumption Data Set. https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

## 许可证

本项目仅用于学术研究和学习目的。

## 联系方式

如有问题或建议，欢迎提Issue或Pull Request。

---

**最后更新**: 2026-01-16  
**版本**: v1.0  
**状态**: ✅ 完全可用

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



## 许可证

MIT License

## 作者

Severin YE - 基于原始论文的代码复现
