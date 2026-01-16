# 合成数据生成工具

灵活的能源数据合成生成器，支持生成训练数据和各种测试场景。

## 功能特点

- ✅ **训练数据生成**: 生成多样化的正常能源数据用于模型训练
- ✅ **场景数据生成**: 生成特定场景的测试数据（高温、低温、极端天气等）
- ✅ **批量生成**: 一次性生成多个预定义场景
- ✅ **自定义参数**: 支持完全自定义的场景参数
- ✅ **真实模拟**: 考虑季节、日夜、时间等多种因素的能源消耗模型

## 安装

无需额外安装，使用项目环境即可：

```bash
source .venv/bin/activate  # 激活虚拟环境
```

## 使用方法

### 1. 生成训练数据

生成2000个样本用于模型训练：

```bash
python scripts/generate_synthetic_data.py \
    --mode training \
    --n-samples 2000 \
    --output data/synthetic
```

参数说明：
- `--n-samples`: 样本数量（默认2000）
- `--start-date`: 起始日期（默认2024-01-01）
- `--output`: 输出目录

输出文件：`data/synthetic/training_data.csv`

### 2. 生成单个场景

生成特定场景的测试数据：

```bash
# 高温高湿场景
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type high_temp_humid \
    --duration 30 \
    --output data/synthetic

# 热浪场景（极端高温）
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type heatwave \
    --duration 48 \
    --output data/synthetic
```

**预定义场景类型**：
- `high_temp_humid`: 高温高湿场景（夏季午后，32°C, 75%湿度）
- `low_temp_humid`: 低温低湿场景（冬季清晨，12°C, 40%湿度）
- `moderate`: 适中温度场景（春秋季，20°C, 55%湿度）
- `peak_hour`: 高峰时段场景（傍晚用电高峰，28°C）
- `valley_hour`: 低谷时段场景（深夜，18°C）
- `heatwave`: 热浪场景（极端高温，38°C, 80%湿度）
- `coldwave`: 寒潮场景（极端低温，5°C, 35%湿度）

参数说明：
- `--scenario-type`: 场景类型
- `--duration`: 时长（小时数，默认30）
- `--start-hour`: 起始小时（0-23，默认0）

输出文件：`data/synthetic/scenario_{场景类型}.csv`

### 3. 批量生成所有预定义场景

一次性生成所有7种预定义场景：

```bash
python scripts/generate_synthetic_data.py \
    --mode batch \
    --output data/synthetic
```

生成的场景：
- `high_temp_humid.csv` (30小时)
- `low_temp_humid.csv` (30小时)
- `moderate.csv` (30小时)
- `peak_hour.csv` (24小时)
- `valley_hour.csv` (24小时)
- `heatwave.csv` (48小时)
- `coldwave.csv` (48小时)

### 4. 自定义场景

完全自定义场景参数：

```bash
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type custom \
    --temp-base 25 \
    --humid-base 60 \
    --wind-base 4 \
    --duration 24 \
    --output data/synthetic
```

参数说明：
- `--temp-base`: 温度基准值（°C）
- `--humid-base`: 湿度基准值（%）
- `--wind-base`: 风速基准值（m/s）

输出文件：`data/synthetic/scenario_custom.csv`

## 数据格式

生成的CSV文件包含以下列：

| 列名 | 说明 | 单位 | 范围 |
|------|------|------|------|
| Temperature | 温度 | °C | 0-50 |
| Humidity | 湿度 | % | 20-95 |
| WindSpeed | 风速 | m/s | 0-25 |
| EDP | 能源消耗（占位符） | kWh | - |
| Hour | 小时 | - | 0-23 |
| DayOfWeek | 星期几 | - | 0-6 |
| Month | 月份 | - | 1-12 |

**注意**: 场景数据中的`EDP`列是占位符(0.0)，需要通过模型预测得到真实值。

## 示例

### 示例1：准备训练数据

```bash
# 生成3000个样本用于训练
python scripts/generate_synthetic_data.py \
    --mode training \
    --n-samples 3000 \
    --start-date 2023-01-01 \
    --output data/synthetic \
    --seed 123

# 输出
# INFO: 训练数据统计:
# INFO:   Temperature: 26.15 ± 5.07
# INFO:   Humidity: 65.02 ± 5.94
# INFO:   WindSpeed: 9.96 ± 4.69
# INFO:   EDP: 120.12 ± 20.88
# INFO: ✅ 训练数据已保存: data/synthetic/training_data.csv
```

### 示例2：准备推理测试场景

```bash
# 方案A：使用批量模式生成所有场景
python scripts/generate_synthetic_data.py --mode batch --output data/test_scenarios

# 方案B：单独生成感兴趣的场景
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type heatwave \
    --duration 72 \
    --output data/test_scenarios
```

### 示例3：在Python代码中使用

```python
from scripts.generate_synthetic_data import EnergyDataGenerator

# 初始化生成器
generator = EnergyDataGenerator(seed=42)

# 生成训练数据
train_data = generator.generate_training_data(n_samples=2000)

# 生成测试场景
test_scenario = generator.generate_scenario(
    scenario_type='heatwave',
    duration=48
)

# 批量生成多个场景
scenarios = generator.generate_multiple_scenarios([
    {'name': 'scene1', 'type': 'high_temp_humid', 'duration': 30},
    {'name': 'scene2', 'type': 'low_temp_humid', 'duration': 30}
])
```

## 更新现有脚本使用新工具

### 更新训练脚本

将 `scripts/run_training.py` 中的数据加载改为：

```python
# 旧方式
data_path = 'data/processed/synthetic_energy_data.csv'

# 新方式
data_path = 'data/synthetic/training_data.csv'
```

### 更新推理脚本

将 `scripts/run_inference.py` 改为从文件加载场景：

```python
import pandas as pd

# 加载预生成的场景
scenarios = [
    ('高温高湿场景', pd.read_csv('data/synthetic/high_temp_humid.csv')),
    ('低温低湿场景', pd.read_csv('data/synthetic/low_temp_humid.csv')),
    ('热浪场景', pd.read_csv('data/synthetic/heatwave.csv'))
]
```

## 高级用法

### 控制随机种子

```bash
python scripts/generate_synthetic_data.py \
    --mode training \
    --seed 12345 \
    --output data/synthetic
```

### 生成长时间序列

```bash
python scripts/generate_synthetic_data.py \
    --mode training \
    --n-samples 10000 \
    --start-date 2023-01-01 \
    --output data/synthetic
```

### 自定义时间段

```bash
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --scenario-type peak_hour \
    --start-hour 17 \
    --duration 6 \
    --output data/synthetic
```

## 数据质量

生成的数据具有以下特性：

1. **时间相关性**: 考虑日变化、周变化、季节变化
2. **特征相关性**: 温度与湿度负相关，风速影响能耗
3. **真实分布**: 基于物理规律的能源消耗模型
4. **可重现性**: 使用随机种子保证结果可复现

## 故障排除

### 问题：生成的数据为空

**解决**: 检查输出目录是否有写权限

```bash
mkdir -p data/synthetic
chmod 755 data/synthetic
```

### 问题：数据统计异常

**解决**: 检查自定义参数是否合理

```bash
# 温度范围: 0-50°C
# 湿度范围: 20-95%
# 风速范围: 0-25 m/s
```

### 问题：场景时长不足

**解决**: 至少需要30个时间步才能生成有效序列

```bash
python scripts/generate_synthetic_data.py \
    --mode scenario \
    --duration 30  # 最少30
```

## 许可证

本工具是能源因果AI系统的一部分，遵循项目许可证。
