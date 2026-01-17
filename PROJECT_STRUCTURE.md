# Project Structure Documentation

## 📁 Complete Directory Tree

Generated: 2026-01-16

```
YS/
├── README.md                 # Main project documentation (entry point)
├── requirements.txt          # Python dependencies list
├── tree.md                   # Project structure tree (original version of this file)
│
├── src/                      # Source code directory
│   ├── __init__.py          
│   │
│   ├── preprocessing/        # Data preprocessing module
│   │   ├── __init__.py
│   │   └── data_preprocessor.py    # Data cleaning, normalization, feature extraction
│   │
│   ├── models/              # Core model module
│   │   ├── __init__.py
│   │   ├── predictor.py            # CNN-LSTM-Attention predictor (core)
│   │   ├── state_classifier.py     # State classifier (Lower/Normal/Peak)
│   │   ├── discretizer.py          # Feature discretizer
│   │   ├── clustering.py           # DLP feature clustering (CAM + Attention)
│   │   ├── association.py          # Association rule mining (Apriori)
│   │   └── bayesian_net.py         # Bayesian Network (structure learning + parameter estimation)
│   │
│   ├── inference/           # Inference and recommendation module
│   │   ├── __init__.py
│   │   ├── causal_inference.py     # Causal inference engine
│   │   └── recommendation.py       # Intelligent recommendation engine
│   │
│   ├── pipeline/            # Training and inference pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py       # 9-step training pipeline
│   │   └── inference_pipeline.py   # Complete inference pipeline
│   │
│   ├── data_processing/     # Data processing tools
│   │   ├── __init__.py
│   │   ├── uci_loader.py           # UCI data loading and preprocessing
│   │   └── data_splitter.py        # Dataset splitting tool
│   │
│   └── visualization/       # Visualization module
│       ├── __init__.py
│       └── inference_visualizer.py # HTML report generator
│
├── scripts/                 # Executable scripts
│   ├── README_synthetic_data.md    # Synthetic data description
│   ├── download_uci_data.py        # UCI data download script
│   ├── split_uci_dataset.py        # Dataset splitting script
│   ├── run_training.py             # Training script (main)
│   ├── run_inference_uci.py        # Inference script (main)
│   ├── generate_synthetic_data.py  # Synthetic data generation
│   ├── compare_datasets.py         # Dataset comparison tool
│   ├── prepare_data.py             # Data preparation script (deprecated)
│   └── view_html_reports.sh        # Quick view script for HTML reports
│
├── data/                    # Data directory
│   ├── README.md            # Data directory description
│   │
│   ├── uci/                # UCI dataset
│   │   ├── raw/            # Raw downloaded data (127MB, gitignored)
│   │   │   └── household_power_consumption.txt
│   │   ├── processed/      # Preprocessed data (16MB, gitignored)
│   │   │   └── uci_household_clean.csv
│   │   └── splits/         # Train/test splits (gitignored)
│   │       ├── train.csv   # Training set (95%, 131,435 samples)
│   │       └── test.csv    # Test set (5%, 6,917 samples)
│   │
│   ├── synthetic/          # Synthetic data (for development testing)
│   │   ├── raw/
│   │   │   └── training_data.csv
│   │   ├── scenarios/      # Various scenario data
│   │   │   ├── heatwave.csv
│   │   │   ├── coldwave.csv
│   │   │   ├── peak_hour.csv
│   │   │   ├── valley_hour.csv
│   │   │   └── moderate.csv
│   │   └── scenario_custom.csv
│   │
│   ├── processed/          # Generic processed data (deprecated)
│   │   └── synthetic_energy_data.csv
│   │
│   └── raw/                # Generic raw data (empty)
│
├── outputs/                 # Output results directory
│   │
│   ├── training_uci/       # UCI data training output
│   │   ├── TRAINING_REPORT.md      # Training results report
│   │   ├── config.json             # Training configuration
│   │   ├── models/                 # Saved models (7 files, 2.3MB)
│   │   │   ├── predictor.keras     # CNN-LSTM-Attention model
│   │   │   ├── preprocessor.pkl    # Preprocessor
│   │   │   ├── state_classifier.pkl # State classifier
│   │   │   ├── discretizer.pkl     # Discretizer
│   │   │   ├── cam_clusterer.pkl   # CAM clusterer
│   │   │   ├── attention_clusterer.pkl # Attention clusterer
│   │   │   └── bayesian_network.bif # Bayesian Network
│   │   └── results/                # Training results
│   │       ├── association_rules.csv   # Association rules
│   │       └── bayesian_network.png    # Bayesian Network diagram
│   │
│   ├── inference_uci/      # UCI data inference output
│   │   ├── INFERENCE_SUMMARY.md    # Inference results summary
│   │   ├── HTML_VISUALIZATION_GUIDE.md # HTML usage guide
│   │   ├── inference_report.txt    # Text report
│   │   ├── inference_details.csv   # Detailed results (CSV)
│   │   ├── inference_results.json  # Structured results (JSON)
│   │   └── html_reports/           # HTML visualization reports
│   │       ├── index.html          # Index page (entry point)
│   │       ├── sample_000.html     # Sample 0 detailed report
│   │       ├── sample_001.html     # Sample 1 detailed report
│   │       └── ...                 # More samples (10 total)
│   │
│   ├── inference/          # Legacy inference output (empty)
│   └── inference_results/  # Legacy inference results
│       └── inference_results.json
│
├── doc/                     # Documentation directory
│   │
│   ├── guides/             # User guides
│   │   ├── QUICKSTART.md           # Quick start guide (detailed tutorial)
│   │   ├── QUICK_REFERENCE.md      # Quick reference (command cheat sheet)
│   │   └── HTML_DEMO.md            # HTML visualization demo description
│   │
│   ├── summaries/          # Progress summaries
│   │   ├── IMPLEMENTATION_SUMMARY.md  # Implementation summary
│   │   ├── PROGRESS.md             # Project progress
│   │   └── PROGRESS_SUMMARY.md     # Progress summary
│   │
│   ├── ChatGPT-Detailed Paper Summary.md     # Complete paper interpretation (tutorial-style)
│   ├── Project Design Document.md             # System architecture design
│   ├── Implementation Document.md                 # Code implementation description
│   ├── Dataset Description-UCI_Household.md # UCI dataset detailed explanation
│   └── Energy Prediction--Causally explainable artificial intelligence on deep learning model for energy demand prediction.pdf
│
├── tests/                   # Test code
│   └── test_core_modules.py        # Core module tests
│
├── logs/                    # Training logs
│   ├── training_uci.log            # UCI training log (1.1MB)
│   ├── training_complete.log       # Complete training log
│   ├── training_full.log           # Full training log
│   └── training_output.log         # Training output log
│
├── notebooks/              # Jupyter notebooks (empty, reserved)
│
└── config/                 # Configuration files (empty, reserved)
```

## 📊 Statistics

- **Total directories**: 38
- **Total files**: 115+
- **Code files**: ~30 Python files
- **Documentation files**: ~15 Markdown files
- **Model files**: 7 trained models
- **Data files**: UCI dataset + synthetic data

## 🔗 Key File Reference Relationships

### Main Entry Files
- `README.md` → Main project documentation, referencing all other documents

### Core Scripts
- `scripts/run_training.py` → Uses `src/pipeline/train_pipeline.py`
- `scripts/run_inference_uci.py` → Uses `src/pipeline/inference_pipeline.py`
- `scripts/download_uci_data.py` → Uses `src/data_processing/uci_loader.py`

### Pipeline Dependencies
- `src/pipeline/train_pipeline.py` → Depends on all modules in `src/models/`
- `src/pipeline/inference_pipeline.py` → Depends on all trained models

### Documentation References
- `README.md` → `doc/guides/QUICKSTART.md`
- `README.md` → `doc/guides/QUICK_REFERENCE.md`
- `README.md` → `doc/guides/HTML_DEMO.md`
- `doc/guides/HTML_DEMO.md` → `outputs/inference_uci/HTML_VISUALIZATION_GUIDE.md`

### Model Dependency Diagram
```
predictor.py (CNN-LSTM-Attention)
    ├── Output predictions → state_classifier.py
    ├── Output CAM features → clustering.py (CAM clustering)
    └── Output Attention features → clustering.py (Attention clustering)

state_classifier.py
    └── Output states → bayesian_net.py

discretizer.py
    └── Output discrete features → association.py, bayesian_net.py

association.py
    └── Output rules → bayesian_net.py (prior knowledge)

bayesian_net.py
    └── Output Bayesian Network → causal_inference.py

causal_inference.py
    └── Output causal inference → recommendation.py

recommendation.py
    └── Output intelligent recommendations → inference_visualizer.py (HTML)
```

## 🎯 Core Module Descriptions

### 1. Prediction Module (`src/models/`)
- **predictor.py**: Parallel CNN-LSTM-Attention architecture, core prediction model
- **state_classifier.py**: Clustering-based state classifier
- **discretizer.py**: KBinsDiscretizer, feature discretization
- **clustering.py**: K-Means clustering for DLP features

### 2. Causal Module (`src/models/` + `src/inference/`)
- **association.py**: Apriori algorithm for association rule mining
- **bayesian_net.py**: Bayesian Network structure learning and parameter estimation
- **causal_inference.py**: Causal inference engine based on Bayesian Network
- **recommendation.py**: Intelligent recommendations based on causal inference

### 3. Pipeline (`src/pipeline/`)
- **train_pipeline.py**: Complete 9-step training process
- **inference_pipeline.py**: Complete inference process

### 4. Utility Modules
- **data_preprocessor.py**: Data preprocessing (cleaning, normalization, feature extraction)
- **uci_loader.py**: UCI data loading, downloading, preprocessing
- **data_splitter.py**: Dataset splitting (time series/random)
- **inference_visualizer.py**: HTML report generation (24KB template)

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
1. `doc/summaries/IMPLEMENTATION_SUMMARY.md` - Implementation summary
2. `doc/summaries/PROGRESS_SUMMARY.md` - Progress summary

## 🚀 Quick Navigation

| I want to... | Go to |
|---------|--------|
| Understand the project | `README.md` |
| Quick start | `doc/guides/QUICKSTART.md` |
| Look up commands | `doc/guides/QUICK_REFERENCE.md` |
| Read paper interpretation | `doc/ChatGPT-Detailed Paper Summary.md` |
| Train model | `scripts/run_training.py` |
| Run inference | `scripts/run_inference_uci.py` |
| View training results | `outputs/training_uci/TRAINING_REPORT.md` |
| View inference results | `outputs/inference_uci/INFERENCE_SUMMARY.md` |
| Browse HTML reports | `outputs/inference_uci/html_reports/index.html` |
| Download data | `scripts/download_uci_data.py` |
| Test code | `tests/test_core_modules.py` |

## 🔄 Data Flow

```
1. Raw data
   data/uci/raw/household_power_consumption.txt (127MB)
   
2. Preprocessing
   ↓ scripts/download_uci_data.py (--preprocess)
   data/uci/processed/uci_household_clean.csv (16MB)
   
3. Splitting
   ↓ scripts/split_uci_dataset.py
   data/uci/splits/train.csv (15MB, 131K samples)
   data/uci/splits/test.csv (776KB, 6.9K samples)
   
4. Training
   ↓ scripts/run_training.py
   outputs/training_uci/models/* (7 model files, 2.3MB)
   outputs/training_uci/TRAINING_REPORT.md
   
5. Inference
   ↓ scripts/run_inference_uci.py
   outputs/inference_uci/inference_report.txt
   outputs/inference_uci/inference_details.csv
   outputs/inference_uci/html_reports/*.html (10 HTML files, 250KB)
```

---
