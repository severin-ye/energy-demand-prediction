# 数据文件夹结构说明

本文件夹包含项目使用的所有数据集，分为UCI真实数据集和合成数据集两大类。

## 📁 文件夹结构

```
data/
├── uci/                          # UCI真实数据集
│   ├── raw/                      # 原始数据（127 MB，不提交到git）
│   │   └── household_power_consumption.txt
│   ├── processed/                # 预处理后的数据（16 MB，不提交到git）
│   │   └── uci_household_clean.csv
│   └── splits/                   # 训练/测试集划分
│       ├── train.csv
│       └── test.csv
│
├── synthetic/                    # 合成数据集
│   ├── raw/                      # 原始合成数据
│   │   └── training_data.csv
│   └── scenarios/                # 测试场景数据
│       ├── heatwave.csv
│       ├── coldwave.csv
│       ├── high_temp_humid.csv
│       ├── low_temp_humid.csv
│       ├── moderate.csv
│       ├── peak_hour.csv
│       └── valley_hour.csv
│
└── processed/                    # 其他处理后的数据
    └── synthetic_energy_data.csv
```

## 📊 数据集说明

### 1. UCI真实数据集

**来源**: UCI Machine Learning Repository  
**数据集ID**: 235  
**名称**: Individual Household Electric Power Consumption

**基本信息**:
- **样本数**: 2,075,259 (原始) → 138,352 (15分钟重采样)
- **时间跨度**: 2006年12月 - 2010年11月（47个月）
- **采样频率**: 1分钟 → 15分钟
- **特征数**: 9个原始特征 + 4个时间特征

**特征说明**:
- `Global_active_power`: 全局有功功率 (kW)
- `Global_reactive_power`: 全局无功功率 (kW)
- `Voltage`: 电压 (V)
- `Global_intensity`: 全局电流强度 (A)
- `Sub_metering_1`: 厨房用电 (Wh)
- `Sub_metering_2`: 洗衣房用电 (Wh)
- `Sub_metering_3`: 热水器和空调用电 (Wh)
- `hour`, `day_of_week`, `month`, `is_weekend`: 时间特征

**文件大小**:
- 原始数据: 127 MB (`uci/raw/household_power_consumption.txt`)
- 清洗数据: 16 MB (`uci/processed/uci_household_clean.csv`)

### 2. 合成数据集

**生成工具**: `scripts/generate_synthetic_data.py`

**基本信息**:
- **样本数**: 2,000 (训练数据)
- **特征数**: 7个
- **特点**: 快速生成，可控参数，适合开发测试

**特征说明**:
- `Temperature`: 温度 (°C)
- `Humidity`: 湿度 (%)
- `WindSpeed`: 风速 (m/s)
- `EDP`: 能源需求预测目标 (kWh)
- `Hour`, `DayOfWeek`, `Month`: 时间特征

**测试场景**:
1. `heatwave.csv`: 热浪场景 (38°C极端高温)
2. `coldwave.csv`: 寒潮场景 (5°C极端低温)
3. `high_temp_humid.csv`: 高温高湿 (32°C, 75%)
4. `low_temp_humid.csv`: 低温低湿 (12°C, 40%)
5. `moderate.csv`: 适中温度 (20°C, 55%)
6. `peak_hour.csv`: 用电高峰 (晚间)
7. `valley_hour.csv`: 用电低谷 (深夜)

## 🔧 数据处理脚本

### 下载和预处理UCI数据
```bash
# 下载原始数据
python scripts/download_uci_data.py --method direct

# 预处理（重采样、特征工程）
python scripts/download_uci_data.py --method direct --preprocess
```

### 生成合成数据
```bash
# 生成训练数据
python scripts/generate_synthetic_data.py --mode training --n-samples 2000

# 生成测试场景
python scripts/generate_synthetic_data.py --mode scenario --scenario-type heatwave

# 批量生成所有场景
python scripts/generate_synthetic_data.py --mode batch
```

### 划分训练/测试集
```bash
# 划分UCI数据集（95% 训练，5% 测试）
python scripts/split_dataset.py \
    --input data/uci/processed/uci_household_clean.csv \
    --output-dir data/uci/splits \
    --test-ratio 0.05
```

## 📏 数据集对比

| 指标 | UCI真实数据 | 合成数据 |
|------|------------|---------|
| 样本数 | 138,352 | 2,000 |
| 时间跨度 | 47个月 | 可配置 |
| 功率均值 | 1.09 kW | 120 kWh |
| 功率范围 | 0.08-8.57 | 63-185 |
| 气象特征 | ❌ 无 | ✅ 有 |
| 文件大小 | 16 MB | 158 KB |
| 适用场景 | 论文发表、实际部署 | 开发测试、快速迭代 |

## ⚠️ 注意事项

### Git管理
- ✅ **提交到git**: 合成数据（<1MB）
- ❌ **不提交git**: UCI原始数据和处理后数据（>100MB）
- 配置在 `.gitignore` 中

### 文件大小限制
- `data/uci/raw/*.txt`: 127 MB（排除）
- `data/uci/processed/*.csv`: 16 MB（排除）
- `data/synthetic/**/*.csv`: <5 MB（可提交）

### 数据使用建议
1. **开发阶段**: 使用合成数据快速迭代
2. **测试阶段**: 使用UCI数据验证模型
3. **论文发表**: 必须使用UCI真实数据
4. **实际部署**: 使用真实业务数据

## 🔗 相关文档

- UCI数据集详细说明: [doc/数据集说明-UCI_Household.md](../doc/数据集说明-UCI_Household.md)
- 合成数据生成器文档: [scripts/README_synthetic_data.md](../scripts/README_synthetic_data.md)
- 数据处理代码: [src/data_processing/](../src/data_processing/)

## 📚 引用

如果使用UCI数据集，请引用：
```
Hebrail, G. & Berard, A. (2012). Individual Household Electric Power Consumption.
UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
```

---

**最后更新**: 2026-01-16
