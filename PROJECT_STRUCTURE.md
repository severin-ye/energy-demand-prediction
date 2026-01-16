# 项目结构说明

## 📁 完整目录树

生成时间: 2026-01-16

```
YS/
├── README.md                 # 项目主文档（入口）
├── requirements.txt          # Python依赖列表
├── tree.md                   # 项目结构树（本文件的原始版本）
├── view_html_reports.sh      # HTML报告快捷查看脚本
│
├── src/                      # 源代码目录
│   ├── __init__.py          
│   │
│   ├── preprocessing/        # 数据预处理模块
│   │   ├── __init__.py
│   │   └── data_preprocessor.py    # 数据清洗、归一化、特征提取
│   │
│   ├── models/              # 核心模型模块
│   │   ├── __init__.py
│   │   ├── predictor.py            # CNN-LSTM-Attention预测器（核心）
│   │   ├── state_classifier.py     # 状态分类器（Lower/Normal/Peak）
│   │   ├── discretizer.py          # 特征离散化器
│   │   ├── clustering.py           # DLP特征聚类（CAM + Attention）
│   │   ├── association.py          # 关联规则挖掘（Apriori）
│   │   └── bayesian_net.py         # 贝叶斯网络（结构学习+参数估计）
│   │
│   ├── inference/           # 推理和推荐模块
│   │   ├── __init__.py
│   │   ├── causal_inference.py     # 因果推断引擎
│   │   └── recommendation.py       # 智能推荐引擎
│   │
│   ├── pipeline/            # 训练和推理流水线
│   │   ├── __init__.py
│   │   ├── train_pipeline.py       # 9步训练流水线
│   │   └── inference_pipeline.py   # 完整推理流水线
│   │
│   ├── data_processing/     # 数据处理工具
│   │   ├── __init__.py
│   │   ├── uci_loader.py           # UCI数据加载和预处理
│   │   └── data_splitter.py        # 数据集分割工具
│   │
│   └── visualization/       # 可视化模块
│       ├── __init__.py
│       └── inference_visualizer.py # HTML报告生成器
│
├── scripts/                 # 可执行脚本
│   ├── README_synthetic_data.md    # 合成数据说明
│   ├── download_uci_data.py        # UCI数据下载脚本
│   ├── split_uci_dataset.py        # 数据集分割脚本
│   ├── run_training.py             # 训练脚本（主）
│   ├── run_inference_uci.py        # 推理脚本（主）
│   ├── generate_synthetic_data.py  # 合成数据生成
│   ├── compare_datasets.py         # 数据集对比工具
│   └── prepare_data.py             # 数据准备脚本（已弃用）
│
├── data/                    # 数据目录
│   ├── README.md            # 数据目录说明
│   │
│   ├── uci/                # UCI数据集
│   │   ├── raw/            # 原始下载数据（127MB，gitignored）
│   │   │   └── household_power_consumption.txt
│   │   ├── processed/      # 预处理后数据（16MB，gitignored）
│   │   │   └── uci_household_clean.csv
│   │   └── splits/         # 训练/测试集（gitignored）
│   │       ├── train.csv   # 训练集（95%，131,435样本）
│   │       └── test.csv    # 测试集（5%，6,917样本）
│   │
│   ├── synthetic/          # 合成数据（用于开发测试）
│   │   ├── raw/
│   │   │   └── training_data.csv
│   │   ├── scenarios/      # 各种场景数据
│   │   │   ├── heatwave.csv
│   │   │   ├── coldwave.csv
│   │   │   ├── peak_hour.csv
│   │   │   ├── valley_hour.csv
│   │   │   └── moderate.csv
│   │   └── scenario_custom.csv
│   │
│   ├── processed/          # 通用处理数据（已弃用）
│   │   └── synthetic_energy_data.csv
│   │
│   └── raw/                # 通用原始数据（空）
│
├── outputs/                 # 输出结果目录
│   │
│   ├── training_uci/       # UCI数据训练输出
│   │   ├── TRAINING_REPORT.md      # 训练结果报告
│   │   ├── config.json             # 训练配置
│   │   ├── models/                 # 保存的模型（7个文件，2.3MB）
│   │   │   ├── predictor.keras     # CNN-LSTM-Attention模型
│   │   │   ├── preprocessor.pkl    # 预处理器
│   │   │   ├── state_classifier.pkl # 状态分类器
│   │   │   ├── discretizer.pkl     # 离散化器
│   │   │   ├── cam_clusterer.pkl   # CAM聚类器
│   │   │   ├── attention_clusterer.pkl # Attention聚类器
│   │   │   └── bayesian_network.bif # 贝叶斯网络
│   │   └── results/                # 训练结果
│   │       ├── association_rules.csv   # 关联规则
│   │       └── bayesian_network.png    # 贝叶斯网络图
│   │
│   ├── inference_uci/      # UCI数据推理输出
│   │   ├── INFERENCE_SUMMARY.md    # 推理结果摘要
│   │   ├── HTML_VISUALIZATION_GUIDE.md # HTML使用指南
│   │   ├── inference_report.txt    # 文本报告
│   │   ├── inference_details.csv   # 详细结果（CSV）
│   │   ├── inference_results.json  # 结构化结果（JSON）
│   │   └── html_reports/           # HTML可视化报告
│   │       ├── index.html          # 索引页面（入口）
│   │       ├── sample_000.html     # 样本0详细报告
│   │       ├── sample_001.html     # 样本1详细报告
│   │       └── ...                 # 更多样本（共10个）
│   │
│   ├── inference/          # 旧版推理输出（空）
│   └── inference_results/  # 旧版推理结果
│       └── inference_results.json
│
├── doc/                     # 文档目录
│   │
│   ├── guides/             # 使用指南
│   │   ├── QUICKSTART.md           # 快速开始指南（详细教程）
│   │   ├── QUICK_REFERENCE.md      # 快速参考（命令速查）
│   │   └── HTML_DEMO.md            # HTML可视化演示说明
│   │
│   ├── summaries/          # 进度总结
│   │   ├── IMPLEMENTATION_SUMMARY.md  # 实现总结
│   │   ├── PROGRESS.md             # 项目进度
│   │   └── PROGRESS_SUMMARY.md     # 进度汇总
│   │
│   ├── ChatGPT-详细整理论文.md     # 论文完整解读（教学式）
│   ├── 项目设计文档.md             # 系统架构设计
│   ├── 实现文档.md                 # 代码实现说明
│   ├── 数据集说明-UCI_Household.md # UCI数据集详解
│   └── 能源预测--基于深度学习模型的因果可解释人工智能在能源需求预测中的应用.pdf
│
├── tests/                   # 测试代码
│   └── test_core_modules.py        # 核心模块测试
│
├── logs/                    # 训练日志
│   ├── training_uci.log            # UCI训练日志（1.1MB）
│   ├── training_complete.log       # 完整训练日志
│   ├── training_full.log           # 全量训练日志
│   └── training_output.log         # 训练输出日志
│
├── notebooks/              # Jupyter笔记本（空，预留）
│
└── config/                 # 配置文件（空，预留）
```

## 📊 统计信息

- **总目录数**: 38
- **总文件数**: 115+
- **代码文件**: ~30个Python文件
- **文档文件**: ~15个Markdown文件
- **模型文件**: 7个训练好的模型
- **数据文件**: UCI数据集 + 合成数据

## 🔗 重要文件引用关系

### 主入口文件
- `README.md` → 项目主文档，引用所有其他文档

### 核心脚本
- `scripts/run_training.py` → 使用 `src/pipeline/train_pipeline.py`
- `scripts/run_inference_uci.py` → 使用 `src/pipeline/inference_pipeline.py`
- `scripts/download_uci_data.py` → 使用 `src/data_processing/uci_loader.py`

### 流水线依赖
- `src/pipeline/train_pipeline.py` → 依赖所有 `src/models/` 中的模块
- `src/pipeline/inference_pipeline.py` → 依赖所有训练好的模型

### 文档引用
- `README.md` → `doc/guides/QUICKSTART.md`
- `README.md` → `doc/guides/QUICK_REFERENCE.md`
- `README.md` → `doc/guides/HTML_DEMO.md`
- `doc/guides/HTML_DEMO.md` → `outputs/inference_uci/HTML_VISUALIZATION_GUIDE.md`

### 模型依赖图
```
predictor.py (CNN-LSTM-Attention)
    ├── 输出预测值 → state_classifier.py
    ├── 输出CAM特征 → clustering.py (CAM聚类)
    └── 输出Attention特征 → clustering.py (Attention聚类)

state_classifier.py
    └── 输出状态 → bayesian_net.py

discretizer.py
    └── 输出离散特征 → association.py, bayesian_net.py

association.py
    └── 输出规则 → bayesian_net.py (先验知识)

bayesian_net.py
    └── 输出贝叶斯网络 → causal_inference.py

causal_inference.py
    └── 输出因果推断 → recommendation.py

recommendation.py
    └── 输出智能建议 → inference_visualizer.py (HTML)
```

## 🎯 核心模块说明

### 1. 预测模块 (`src/models/`)
- **predictor.py**: 并行CNN-LSTM-Attention架构，核心预测模型
- **state_classifier.py**: 基于聚类的状态分类器
- **discretizer.py**: KBinsDiscretizer，特征离散化
- **clustering.py**: K-Means聚类DLP特征

### 2. 因果模块 (`src/models/` + `src/inference/`)
- **association.py**: Apriori算法挖掘关联规则
- **bayesian_net.py**: 贝叶斯网络结构学习和参数估计
- **causal_inference.py**: 基于贝叶斯网络的因果推断引擎
- **recommendation.py**: 基于因果推断的智能推荐

### 3. 流水线 (`src/pipeline/`)
- **train_pipeline.py**: 完整的9步训练流程
- **inference_pipeline.py**: 完整的推理流程

### 4. 工具模块
- **data_preprocessor.py**: 数据预处理（清洗、归一化、特征提取）
- **uci_loader.py**: UCI数据加载、下载、预处理
- **data_splitter.py**: 数据集分割（时间序列/随机）
- **inference_visualizer.py**: HTML报告生成（24KB模板）

## 📝 文档分类

### 入门文档
1. `README.md` - 项目概览和快速开始
2. `doc/guides/QUICKSTART.md` - 详细安装和使用教程
3. `doc/guides/QUICK_REFERENCE.md` - 命令速查表

### 技术文档
1. `doc/ChatGPT-详细整理论文.md` - 论文完整解读
2. `doc/项目设计文档.md` - 系统架构设计
3. `doc/实现文档.md` - 代码实现细节
4. `doc/数据集说明-UCI_Household.md` - 数据集说明

### 结果文档
1. `outputs/training_uci/TRAINING_REPORT.md` - 训练结果报告
2. `outputs/inference_uci/INFERENCE_SUMMARY.md` - 推理结果摘要
3. `outputs/inference_uci/HTML_VISUALIZATION_GUIDE.md` - HTML使用指南

### 进度文档
1. `doc/summaries/IMPLEMENTATION_SUMMARY.md` - 实现总结
2. `doc/summaries/PROGRESS_SUMMARY.md` - 进度汇总

## 🚀 快速导航

| 我想... | 去哪里 |
|---------|--------|
| 了解项目 | `README.md` |
| 快速开始 | `doc/guides/QUICKSTART.md` |
| 查命令 | `doc/guides/QUICK_REFERENCE.md` |
| 看论文解读 | `doc/ChatGPT-详细整理论文.md` |
| 训练模型 | `scripts/run_training.py` |
| 运行推理 | `scripts/run_inference_uci.py` |
| 查看训练结果 | `outputs/training_uci/TRAINING_REPORT.md` |
| 查看推理结果 | `outputs/inference_uci/INFERENCE_SUMMARY.md` |
| 浏览HTML报告 | `outputs/inference_uci/html_reports/index.html` |
| 下载数据 | `scripts/download_uci_data.py` |
| 测试代码 | `tests/test_core_modules.py` |

## 🔄 数据流向

```
1. 原始数据
   data/uci/raw/household_power_consumption.txt (127MB)
   
2. 预处理
   ↓ scripts/download_uci_data.py (--preprocess)
   data/uci/processed/uci_household_clean.csv (16MB)
   
3. 分割
   ↓ scripts/split_uci_dataset.py
   data/uci/splits/train.csv (15MB, 131K样本)
   data/uci/splits/test.csv (776KB, 6.9K样本)
   
4. 训练
   ↓ scripts/run_training.py
   outputs/training_uci/models/* (7个模型文件, 2.3MB)
   outputs/training_uci/TRAINING_REPORT.md
   
5. 推理
   ↓ scripts/run_inference_uci.py
   outputs/inference_uci/inference_report.txt
   outputs/inference_uci/inference_details.csv
   outputs/inference_uci/html_reports/*.html (10个HTML, 250KB)
```

---

**生成时间**: 2026-01-16  
**最后更新**: 2026-01-16  
**版本**: v1.0
