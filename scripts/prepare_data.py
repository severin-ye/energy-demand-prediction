"""
UCI Household数据集下载和预处理脚本
数据源: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_uci_household_data(output_path='data/raw/household_power_consumption.txt'):
    """下载UCI Household数据集"""
    if os.path.exists(output_path):
        logger.info(f"数据文件已存在: {output_path}")
        return output_path
    
    logger.info("开始下载UCI Household数据集...")
    
    import urllib.request
    import zipfile
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
    zip_path = 'data/raw/household_power_consumption.zip'
    
    try:
        logger.info(f"从 {url} 下载...")
        urllib.request.urlretrieve(url, zip_path)
        
        logger.info(f"解压到 {os.path.dirname(output_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/raw')
        
        os.remove(zip_path)
        logger.info("下载完成！")
        return output_path
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("将使用合成数据集")
        return None


def create_synthetic_dataset(
    n_samples=2000,
    output_path='data/processed/synthetic_energy_data.csv'
):
    """创建合成能源数据集"""
    logger.info(f"创建合成能源数据集 (n={n_samples})...")
    
    np.random.seed(42)
    
    hours = np.arange(n_samples)
    
    # 温度：有日周期和季节周期
    temperature = (
        20 +
        5 * np.sin(2 * np.pi * hours / 24) +
        3 * np.sin(2 * np.pi * hours / (24 * 30)) +
        np.random.randn(n_samples) * 2
    )
    
    # 湿度：与温度负相关
    humidity = (
        70 -
        0.8 * temperature +
        np.random.randn(n_samples) * 5
    )
    
    # 风速
    windspeed = np.abs(
        10 +
        3 * np.sin(2 * np.pi * hours / (24 * 7)) +
        np.random.randn(n_samples) * 3
    )
    
    # EDP：受多因素影响
    hour_of_day = hours % 24
    peak_hours = ((hour_of_day >= 18) & (hour_of_day <= 22)).astype(float)
    
    edp = (
        100 +
        1.5 * temperature +
        0.3 * humidity +
        -0.5 * windspeed +
        30 * peak_hours +
        np.random.randn(n_samples) * 10
    )
    
    data = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'WindSpeed': windspeed,
        'EDP': edp,
        'Hour': hour_of_day,
        'DayOfWeek': (hours // 24) % 7,
        'Month': ((hours // 24 // 30) % 12) + 1
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    
    logger.info(f"合成数据已保存到: {output_path}")
    logger.info(f"数据形状: {data.shape}")
    logger.info(f"数据统计:\n{data.describe()}")
    
    return data


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("数据准备脚本")
    logger.info("="*60)
    
    # 创建合成数据集
    synthetic_data = create_synthetic_dataset(
        n_samples=2000,
        output_path='data/processed/synthetic_energy_data.csv'
    )
    
    logger.info("\n✅ 数据准备完成！")
    logger.info("数据路径: data/processed/synthetic_energy_data.csv")
    logger.info("\n" + "="*60)
    logger.info("可以开始训练模型！")
    logger.info("="*60)
