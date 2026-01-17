# Causal Explainable AI System for Energy Demand Prediction

Complete code reproduction based on the paper *"Causally explainable artificial intelligence on deep learning model for energy demand prediction"* (Erlangga & Cho, 2025).

## Project Overview

This project implements an energy demand prediction system combining deep learning prediction and causal explanation:

- **Prediction Module**: Parallel CNN-LSTM-Attention architecture for high-precision energy demand prediction
- **Explanation Module**: Bayesian Network combined with Deep Learning Parameters (DLP) for stable causal explanation
- **Recommendation Module**: Actionable energy-saving recommendations based on causal inference

## Core Features

✅ **High-performance Prediction**: 34.84% improvement over serial architecture (UCI) and 13.63% (REFIT)  
✅ **Stable Explanation**: Cosine similarity of 0.999+ (SHAP only 0.95-0.96)  
✅ **Causal Reasoning**: Bayesian Network with domain knowledge constraints  
✅ **Actionable Recommendations**: Specific recommendations for Peak/Normal/Lower states  
✅ **HTML Visualization**: Beautiful 10-step inference process visualization reports

## 📁 Project Structure

```
YS/
├── src/                      # Source code
│   ├── preprocessing/        # Data preprocessing
│   ├── models/              # Core models
│   │   ├── predictor.py     # CNN-LSTM-Attention predictor
│   │   ├── state_classifier.py  # State classifier
│   │   ├── discretizer.py   # Feature discretization
│   │   ├── clustering.py    # DLP feature clustering
│   │   ├── association.py   # Association rule mining
│   │   └── bayesian_net.py  # Bayesian Network
│   ├── inference/           # Inference module
│   │   ├── causal_inference.py  # Causal inference
│   │   └── recommendation.py    # Intelligent recommendations
│   ├── pipeline/            # Pipelines
│   │   ├── train_pipeline.py    # Training pipeline
│   │   └── inference_pipeline.py # Inference pipeline
│   ├── data_processing/     # Data processing
│   │   ├── uci_loader.py    # UCI data loader
│   │   └── data_splitter.py # Dataset splitting
│   └── visualization/       # Visualization
│       └── inference_visualizer.py  # HTML report generation
│
├── scripts/                  # Script tools
│   ├── download_uci_data.py # UCI data download
│   ├── split_uci_dataset.py # Dataset splitting
│   ├── run_training.py      # Training script
│   ├── run_inference_uci.py # Inference script
│   └── view_html_reports.sh # HTML report viewer
│
├── data/                     # Data directory
│   ├── uci/                 # UCI dataset
│   │   ├── raw/             # Raw data
│   │   ├── processed/       # Preprocessed
│   │   └── splits/          # Train/test splits
│   └── synthetic/           # Synthetic data
│
├── outputs/                  # Output results
│   ├── training_uci/        # Training output
│   │   ├── models/          # Saved models
│   │   └── results/         # Training results
│   └── inference_uci/       # Inference output
│       └── html_reports/    # HTML visualization reports
│
├── doc/                      # Documentation directory
│   ├── guides/              # User guides
│   │   ├── QUICKSTART.md    # Quick start
│   │   ├── QUICK_REFERENCE.md  # Quick reference
│   │   └── HTML_DEMO.md     # HTML visualization demo
│   ├── summaries/           # Progress summaries
│   └── ChatGPT-Detailed Paper Summary.md  # Paper interpretation
│
├── tests/                    # Test code
├── logs/                     # Training logs
└── README.md                 # This file
```

## Quick Start

### 📖 Detailed Guides

- **[Quick Start Guide](doc/guides/QUICKSTART.md)** - Complete installation and usage tutorial
- **[Quick Reference](doc/guides/QUICK_REFERENCE.md)** - Command cheat sheet
- **[HTML Visualization Demo](doc/guides/HTML_DEMO.md)** - Inference process visualization usage guide

### ⚡ 30-Second Quick Launch

```bash
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download UCI dataset (~127MB)
python scripts/download_uci_data.py --method direct --preprocess

# 3. Split dataset (95% train / 5% test)
python scripts/split_uci_dataset.py --test-ratio 0.05

# 4. Train model (~5 minutes, CPU)
python scripts/run_training.py \
  --data data/uci/splits/train.csv \
  --epochs 20 \
  --batch-size 64

# 5. Inference testing (generate HTML visualization)
python scripts/run_inference_uci.py --n-samples 100

# 6. View HTML reports
./scripts/view_html_reports.sh
```

### 📊 UCI Dataset Complete Workflow

**Data Preparation**
```bash
# Download and preprocess UCI data
python scripts/download_uci_data.py --method direct --preprocess

# View dataset information
python scripts/download_uci_data.py --info

# Split into training and test sets
python scripts/split_uci_dataset.py \
  --input data/uci/processed/uci_household_clean.csv \
  --output-dir data/uci/splits \
  --test-ratio 0.05
```

**Model Training**
```bash
# Train using UCI data (auto-detect data type)
python scripts/run_training.py \
  --data data/uci/splits/train.csv \
  --epochs 20 \
  --batch-size 64 \
  --output-dir outputs/training_uci

# View report after training
cat outputs/training_uci/TRAINING_REPORT.md
```

**Inference Testing**
```bash
# Run inference and generate HTML visualization
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
- **[Training Report](outputs/training_uci/TRAINING_REPORT.md)** - UCI data training results
- **[Inference Summary](outputs/inference_uci/INFERENCE_SUMMARY.md)** - Inference test results summary
- **[HTML Reports](outputs/inference_uci/html_reports/index.html)** - Visualization inference reports

## Core Module Descriptions

### 1. Prediction Module (`src/models/predictor.py`)
- Parallel CNN-LSTM architecture
- Attention mechanism
- Extract CAM and Attention features (DLP)

### 2. State Classification (`src/models/state_classifier.py`)
- Classify continuous predictions into Lower/Normal/Peak
- Dynamic thresholds based on clustering

### 3. Feature Discretization (`src/models/discretizer.py`)
- Convert continuous features to discrete levels
- Support causal inference

### 4. DLP Clustering (`src/models/clustering.py`)
- CAM feature clustering (K-Means)
- Attention type classification (Early/Late/Other)

### 5. Association Rules (`src/models/association.py`)
- Apriori algorithm for EDP rule mining
- Provide prior knowledge for Bayesian Network

### 6. Bayesian Network (`src/models/bayesian_net.py`)
- Structure learning (Hill-Climbing)
- Parameter estimation (Maximum Likelihood)
- Domain knowledge constraints

### 7. Causal Inference (`src/inference/causal_inference.py`)
- Causal reasoning based on Bayesian Network
- Counterfactual queries
- Sensitivity analysis

### 8. Intelligent Recommendations (`src/inference/recommendation.py`)
- Generate recommendations based on causal inference
- Personalized recommendations for different states

## Training Pipeline

Complete 9-step training process:

1. **Data Preprocessing** - Cleaning, normalization, time feature extraction
2. **Predictor Training** - CNN-LSTM-Attention model training
3. **State Classifier** - State clustering based on predictions
4. **Feature Discretization** - KBinsDiscretizer training
5. **DLP Clustering** - CAM and Attention feature clustering
6. **Association Rule Mining** - Apriori algorithm to extract rules
7. **Bayesian Network Learning** - Structure learning and parameter estimation
8. **Causal Inference Initialization** - Create inference engine
9. **Model Saving** - Save all trained models

## Inference Pipeline

Complete inference process:

1. **Load Models** - Load all trained models
2. **Data Preprocessing** - Same preprocessing as training
3. **Prediction** - Generate predictions using CNN-LSTM-Attention
4. **Extract DLP** - Extract CAM and Attention features
5. **State Classification** - Classify into Lower/Normal/Peak
6. **Feature Discretization** - Convert to discrete levels
7. **Causal Inference** - Inference based on Bayesian Network
8. **Generate Recommendations** - Generate recommendations based on current state
9. **HTML Visualization** - Generate beautiful visualization reports

## Performance Metrics

### UCI Dataset Test Results

| Metric | Value |
|------|------|
| MAE | 0.6718 kW |
| RMSE | 0.8460 kW |
| Samples | 80 |
| Training Time | ~5 minutes (CPU, 20 epochs)|
| Inference Speed | ~1 second/sample |

### Model Size

| Component | Parameters/Size |
|------|------------|
| Predictor | 58,867 parameters |
| State Classifier | ~1KB |
| Discretizer | ~2KB |
| CAM Clusterer | ~5KB |
| Attention Clusterer | ~3KB |
| Bayesian Network | ~50KB |
| **Total** | **~2.3MB** |

## Development Tools

### Testing
```bash
# Run core module tests
python tests/test_core_modules.py

# Run complete test suite
pytest tests/
```

### Data Generation
```bash
# Generate synthetic data (for development testing)
python scripts/generate_synthetic_data.py \
  --n-samples 10000 \
  --scenario heatwave \
  --output data/synthetic/scenario_heatwave.csv
```

### Dataset Comparison
```bash
# Compare UCI and synthetic data
python scripts/compare_datasets.py
```

## Frequently Asked Questions

### Q: Why only generate 10 HTML reports?
A: To balance speed and practicality. You can modify the `num_samples` parameter in `scripts/run_inference_uci.py`.

### Q: What if training takes too long?
A: Reduce epochs (e.g., `--epochs 10`) or increase batch size (e.g., `--batch-size 128`).

### Q: How to use GPU acceleration?
A: Install GPU version of TensorFlow: `pip install tensorflow-gpu`

### Q: Dataset too large, insufficient memory?
A: Use `--n-samples` parameter to limit the number of samples.

## References

1. Erlangga, D., & Cho, S. (2025). Causally explainable artificial intelligence on deep learning model for energy demand prediction. *Applied Energy*.

2. UCI Machine Learning Repository. (2012). Individual Household Electric Power Consumption Data Set. https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

## License

This project is for academic research and learning purposes only.

## Contact

Welcome to submit Issues or Pull Requests if you have questions or suggestions.

---

**Last Updated**: 2026-01-16  
**Version**: v1.0  
**Status**: ✅ Fully Functional

## Technical Architecture

### Prediction Model
- **Parallel Architecture**: CNN branch + LSTM-Attention branch
- **Feature Extraction**: Time series sliding window + time feature engineering
- **Robust Classification**: Sn scale estimator for outlier handling

### Explanation Model
- **DLP Clustering**: CAM and Attention weight clustering
- **Association Rules**: Apriori algorithm for mining candidate causal relationships
- **Bayesian Network**: Structure learning with domain knowledge constraints

### Causal Inference
- **Do-Calculus**: Calculate intervention effects
- **Sensitivity Analysis**: Tornado diagram visualization
- **Counterfactual Analysis**: Compare factual and counterfactual distributions

## Performance Metrics

### Prediction Performance (vs Serial CNN-LSTM)
| Dataset | MSE Improvement | MAPE Improvement |
|--------|---------|----------|
| UCI    | 34.84%  | 32.71%   |
| REFIT  | 13.63%  | 11.45%   |

### Explanation Consistency (Cosine Similarity)
| Method      | UCI Dataset | REFIT Dataset |
|-----------|-----------|-------------|
| This Method (BN) | 0.99940   | 0.99983     |
| SHAP      | 0.95210   | 0.96478     |

## References

Gatum Erlangga, Sung-Bae Cho. *Causally explainable artificial intelligence on deep learning model for energy demand prediction*. Engineering Applications of Artificial Intelligence, Volume 162, 2025.



## License

MIT License

## Author

Severin YE - Code reproduction based on the original paper
