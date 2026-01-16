# 论文使用的数据集详细信息

本文档记录论文《基于深度学习模型的因果可解释人工智能在能源需求预测中的应用》中使用的真实数据集信息。

## 📊 数据集概述

论文使用了 **两个公开数据集**：

### 1. UCI Individual Household Electric Power Consumption

**基本信息**：
- **来源**: UCI Machine Learning Repository
- **数据量**: 2,075,259 条记录
- **时间范围**: 2006年12月 - 2010年11月（47个月）
- **采样频率**: 1分钟
- **地点**: 法国Sceaux（距巴黎7公里）
- **特征数**: 9个

**数据特点**：
- ✅ 公开可下载
- ✅ 长时间跨度（近4年）
- ✅ 高频采样（1分钟级）
- ⚠️ 约1.25%数据缺失
- 📦 文件大小：126.8 MB

**变量说明**：

| 变量名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| Date | 分类 | - | 日期 (dd/mm/yyyy) |
| Time | 分类 | - | 时间 (hh:mm:ss) |
| Global_active_power | 连续 | kW | 全局有功功率（分钟平均） |
| Global_reactive_power | 连续 | kW | 全局无功功率（分钟平均） |
| Voltage | 连续 | V | 电压（分钟平均） |
| Global_intensity | 连续 | A | 全局电流强度（分钟平均） |
| Sub_metering_1 | 连续 | Wh | 子计量1：厨房（洗碗机、烤箱、微波炉） |
| Sub_metering_2 | 连续 | Wh | 子计量2：洗衣房（洗衣机、烘干机、冰箱、灯） |
| Sub_metering_3 | 连续 | Wh | 子计量3：电热水器和空调 |

**重要注释**：
```
未计量的能耗 = (global_active_power * 1000/60) - sub_metering_1 - sub_metering_2 - sub_metering_3
```
这代表其他未被子计量覆盖的电器的能耗（单位：瓦时/分钟）。

---

### 2. REFIT Dataset

**基本信息**：
- **来源**: REFIT: Electrical Load Measurements (英国)
- **数据量**: 5,733,526 条记录
- **采样频率**: 8秒
- **特点**: 更多电器维度，粒度更细

---

## 📥 如何获取UCI数据集

### 方法1: 直接下载（推荐）

**下载链接**：
```
https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
```

**步骤**：
1. 访问上述链接直接下载ZIP文件（126.8 MB）
2. 解压得到 `household_power_consumption.txt`
3. 文件格式：分号分隔的文本文件

### 方法2: 使用Python API

安装ucimlrepo包：
```bash
pip install ucimlrepo
```

在代码中导入：
```python
from ucimlrepo import fetch_ucirepo 

# 获取数据集
dataset = fetch_ucirepo(id=235) 

# 提取特征和目标
X = dataset.data.features 
y = dataset.data.targets 

# 查看元数据
print(dataset.metadata) 
print(dataset.variables)
```

### 方法3: 使用我们提供的下载脚本

创建 `scripts/download_uci_data.py`:

```python
"""
下载UCI Household数据集
"""
import os
import urllib.request
import zipfile
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_uci_dataset(output_dir='data/raw'):
    """下载并解压UCI数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载URL
    url = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    zip_path = os.path.join(output_dir, 'uci_household.zip')
    txt_path = os.path.join(output_dir, 'household_power_consumption.txt')
    
    # 如果已存在，跳过下载
    if os.path.exists(txt_path):
        logger.info(f"✅ 数据文件已存在: {txt_path}")
        return txt_path
    
    # 下载
    logger.info(f"开始下载UCI数据集...")
    logger.info(f"URL: {url}")
    urllib.request.urlretrieve(url, zip_path)
    logger.info(f"✅ 下载完成: {zip_path}")
    
    # 解压
    logger.info("解压文件...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    logger.info(f"✅ 解压完成: {output_dir}")
    
    # 删除zip文件
    os.remove(zip_path)
    logger.info("清理临时文件")
    
    return txt_path

def load_uci_dataset(filepath):
    """加载UCI数据集为Pandas DataFrame"""
    
    logger.info(f"加载数据: {filepath}")
    
    # 读取数据（分号分隔，包含缺失值）
    df = pd.read_csv(
        filepath,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        na_values=['?'],
        infer_datetime_format=True,
        low_memory=False
    )
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"列名: {df.columns.tolist()}")
    logger.info(f"缺失值: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.2f}%)")
    
    return df

if __name__ == "__main__":
    # 下载数据
    filepath = download_uci_dataset()
    
    # 加载数据
    df = load_uci_dataset(filepath)
    
    # 显示基本信息
    print("\n" + "="*70)
    print("UCI Household Electric Power Consumption Dataset")
    print("="*70)
    print(f"\n样本数: {len(df):,}")
    print(f"特征数: {len(df.columns)}")
    print(f"时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe())
```

**使用方法**：
```bash
python scripts/download_uci_data.py
```

---

## 🔄 数据预处理步骤（论文中的做法）

根据论文，UCI数据集的预处理包括：

### 1. 时间重采样
- 原始：1分钟采样
- 重采样：15分钟平均（论文表3中使用）
- 方法：使用滑动窗口平均

### 2. 特征选择
论文主要使用以下特征：
- `Global_active_power` (主要目标变量，即EDP)
- 其他特征用于因果分析

### 3. 缺失值处理
- 方法：前向填充 (forward fill) 或线性插值
- 缺失率：约1.25%

### 4. 序列划分
- 训练集：前80%
- 测试集：后20%
- 序列长度：根据预测时间窗口设定

---

## 📈 论文中的使用方式

### 第一阶段：预测模型训练
- **数据集**: UCI + REFIT
- **目的**: 训练Parallel CNN-LSTM-Attention模型
- **输入**: 历史时间序列
- **输出**: EDP预测值

### 第二阶段：因果推断
- **数据集**: 使用训练好的模型在测试集上的预测
- **目的**: 构建因果贝叶斯网络
- **方法**: 不需要重新训练，使用统计方法和规则挖掘

---

## 📊 论文中报告的数据统计

**UCI数据集**:
- 样本数：2,075,259
- 预测任务：15分钟分辨率下的EDP预测
- 性能提升：相比串联CNN-LSTM提升 **34.84%**

**REFIT数据集**:
- 样本数：5,733,526
- 预测任务：多电器用电预测
- 性能提升：相比串联CNN-LSTM提升 **13.63%**

---

## ⚠️ 为什么我们使用合成数据

### 原因

1. **UCI数据集很大**（126.8 MB，200万+条记录）
   - 下载需要时间
   - 处理需要较大内存
   - 训练周期长

2. **项目是实现论文方法，不是复现论文结果**
   - 重点是验证方法的可行性
   - 不需要完全一致的数值结果
   - 合成数据更灵活可控

3. **合成数据的优势**
   - ✅ 快速生成任意规模
   - ✅ 可控的数据分布
   - ✅ 方便调试和验证
   - ✅ 不依赖外部下载

### 何时使用真实数据

如果你想要：
- 复现论文的精确数值结果
- 发表学术论文
- 做真实场景的应用部署

则应该使用UCI真实数据集。

---

## 🚀 快速开始

### 使用合成数据（当前方案）
```bash
# 生成训练数据
python scripts/generate_synthetic_data.py --mode training --n-samples 2000

# 训练模型
python scripts/run_training.py
```

### 使用UCI真实数据（完整流程）
```bash
# 1. 下载UCI数据
python scripts/download_uci_data.py

# 2. 预处理为训练格式
python scripts/preprocess_uci_data.py \
    --input data/raw/household_power_consumption.txt \
    --output data/processed/uci_processed.csv \
    --resample 15min

# 3. 训练模型
python scripts/run_training.py \
    --data data/processed/uci_processed.csv
```

---

## 📚 引用信息

如果使用UCI数据集，请引用：

```bibtex
@misc{hebrail2012individual,
  title={Individual Household Electric Power Consumption},
  author={Hebrail, Georges and Berard, Alice},
  year={2012},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C58K54},
  url={https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption}
}
```

**许可证**: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🔗 相关链接

- **UCI数据集页面**: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **直接下载**: https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
- **原始论文**: 见 `doc/能源预测--基于深度学习模型的因果可解释人工智能在能源需求预测中的应用.pdf`
- **UCI ML Repository**: https://archive.ics.uci.edu/

---

**最后更新**: 2026-01-16
