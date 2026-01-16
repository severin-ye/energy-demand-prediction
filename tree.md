.
├── PROJECT_STRUCTURE.md
├── README.md
├── config
├── data
│   ├── README.md
│   ├── processed
│   │   └── synthetic_energy_data.csv
│   ├── raw
│   ├── synthetic
│   │   ├── raw
│   │   ├── scenario_custom.csv
│   │   └── scenarios
│   └── uci
│       ├── processed
│       ├── raw
│       └── splits
├── doc
│   ├── ChatGPT-详细整理论文.md
│   ├── INDEX.md
│   ├── guides
│   │   ├── HTML_DEMO.md
│   │   ├── QUICKSTART.md
│   │   └── QUICK_REFERENCE.md
│   ├── summaries
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── PROGRESS.md
│   │   └── PROGRESS_SUMMARY.md
│   ├── 实现文档.md
│   ├── 数据集说明-UCI_Household.md
│   ├── 能源预测--基于深度学习模型的因果可解释人工智能在能源需求预测中的应用.pdf
│   └── 项目设计文档.md
├── logs
│   ├── training_complete.log
│   ├── training_full.log
│   ├── training_output.log
│   └── training_uci.log
├── notebooks
├── outputs
│   ├── inference
│   ├── inference_results
│   │   └── inference_results.json
│   ├── inference_uci
│   │   ├── HTML_VISUALIZATION_GUIDE.md
│   │   ├── INFERENCE_SUMMARY.md
│   │   ├── html_reports
│   │   ├── inference_details.csv
│   │   ├── inference_report.txt
│   │   └── inference_results.json
│   └── training_uci
│       ├── TRAINING_REPORT.md
│       ├── config.json
│       ├── models
│       └── results
├── requirements.txt
├── scripts
│   ├── README_synthetic_data.md
│   ├── compare_datasets.py
│   ├── download_uci_data.py
│   ├── generate_synthetic_data.py
│   ├── prepare_data.py
│   ├── run_inference.py
│   ├── run_inference_uci.py
│   ├── run_training.py
│   └── split_uci_dataset.py
├── src
│   ├── __init__.py
│   ├── data_processing
│   │   ├── __init__.py
│   │   ├── data_splitter.py
│   │   └── uci_loader.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── causal_inference.py
│   │   └── recommendation.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── association.py
│   │   ├── bayesian_net.py
│   │   ├── clustering.py
│   │   ├── discretizer.py
│   │   ├── predictor.py
│   │   └── state_classifier.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── inference_pipeline.py
│   │   └── train_pipeline.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   └── visualization
│       ├── __init__.py
│       └── inference_visualizer.py
├── tests
│   └── test_core_modules.py
├── tree.md
└── view_html_reports.sh

34 directories, 63 files
