"""
使用ucimlrepo Python API下载UCI Household数据集

数据集信息:
- 名称: Individual Household Electric Power Consumption
- ID: 235
- 样本数: 2,075,259
- 时间范围: 2006/12/16 - 2010/11/26
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_with_progress(url, output_path):
    """带进度条的下载"""
    import urllib.request
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 / total_size)
            downloaded = count * block_size / (1024 * 1024)
            total = total_size / (1024 * 1024)
            print(f"\r下载进度: {percent:.1f}% ({downloaded:.1f}/{total:.1f} MB)", end='', flush=True)
        else:
            downloaded = count * block_size / (1024 * 1024)
            print(f"\r已下载: {downloaded:.1f} MB", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # 换行
    except Exception as e:
        print()  # 换行
        raise


def download_uci_dataset(output_dir='data/raw', method='direct'):
    """
    下载UCI数据集
    
    Args:
        output_dir: 输出目录
        method: 'api' 使用ucimlrepo, 'direct' 直接下载ZIP
    
    Returns:
        DataFrame: 加载的数据集
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if method == 'api':
        logger.info("使用ucimlrepo Python API下载...")
        
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            logger.error("❌ ucimlrepo未安装")
            logger.info("安装命令: pip install ucimlrepo")
            sys.exit(1)
        
        # 获取数据集
        logger.info("正在从UCI仓库获取数据集 ID=235...")
        dataset = fetch_ucirepo(id=235)
        
        # 提取数据
        logger.info("✅ 数据集下载成功")
        
        # 显示元数据
        logger.info("\n" + "="*70)
        logger.info("数据集元数据")
        logger.info("="*70)
        if hasattr(dataset, 'metadata'):
            for key, value in dataset.metadata.items():
                if key in ['name', 'num_instances', 'num_features', 'area', 'task']:
                    logger.info(f"{key}: {value}")
        
        # 组合特征和目标（如果有）
        X = dataset.data.features
        
        logger.info(f"\n特征数据形状: {X.shape}")
        logger.info(f"列名: {X.columns.tolist()}")
        
        # 保存原始数据
        output_path = os.path.join(output_dir, 'uci_household_raw.csv')
        X.to_csv(output_path, index=False)
        logger.info(f"✅ 保存原始数据: {output_path}")
        
        return X
        
    elif method == 'direct':
        logger.info("使用直接下载方式...")
        import zipfile
        
        # 下载URL
        url = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
        zip_path = os.path.join(output_dir, 'uci_household.zip')
        txt_path = os.path.join(output_dir, 'household_power_consumption.txt')
        
        # 检查是否已存在
        if os.path.exists(txt_path):
            logger.info(f"✅ 数据文件已存在: {txt_path}")
            return load_txt_dataset(txt_path)
        
        # 下载
        logger.info(f"开始下载: {url}")
        logger.info("文件大小: ~126.8 MB，请耐心等待...")
        
        try:
            download_with_progress(url, zip_path)
            logger.info(f"✅ 下载完成: {zip_path}")
        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            logger.info("提示: 如果下载超时，可以手动从以下地址下载:")
            logger.info(f"  {url}")
            logger.info(f"  然后放到: {zip_path}")
            raise
        
        # 解压
        logger.info("解压文件...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"✅ 解压完成")
        
        # 删除zip
        os.remove(zip_path)
        logger.info("清理临时文件")
        
        # 加载数据
        return load_txt_dataset(txt_path)


def load_txt_dataset(filepath):
    """加载TXT格式的UCI数据集"""
    logger.info(f"加载数据: {filepath}")
    
    # 读取数据（分号分隔，问号表示缺失值）
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?'],
        low_memory=False
    )
    
    # 解析日期时间
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )
    
    # 删除原始Date和Time列
    df = df.drop(['Date', 'Time'], axis=1)
    
    # 重新排序列（datetime放第一列）
    cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
    df = df[cols]
    
    logger.info(f"✅ 数据加载完成: {df.shape}")
    
    return df


def analyze_dataset(df):
    """分析数据集基本信息"""
    logger.info("\n" + "="*70)
    logger.info("数据集分析")
    logger.info("="*70)
    
    logger.info(f"\n📊 基本信息:")
    logger.info(f"  样本数: {len(df):,}")
    logger.info(f"  特征数: {len(df.columns)}")
    
    # 检查时间范围
    if 'datetime' in df.columns:
        logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        time_span = df['datetime'].max() - df['datetime'].min()
        logger.info(f"  时长: {time_span.days} 天 ({time_span.days/30.5:.1f} 个月)")
    
    # 缺失值统计
    missing = df.isnull().sum()
    total_missing = missing.sum()
    missing_pct = total_missing / df.size * 100
    
    logger.info(f"\n⚠️  缺失值:")
    logger.info(f"  总计: {total_missing:,} ({missing_pct:.2f}%)")
    if total_missing > 0:
        logger.info(f"  各列缺失:")
        for col, count in missing[missing > 0].items():
            pct = count / len(df) * 100
            logger.info(f"    {col}: {count:,} ({pct:.2f}%)")
    
    # 数值型特征统计
    logger.info(f"\n📈 数值统计:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()
    logger.info(f"\n{stats.to_string()}")
    
    # 显示前几行
    logger.info(f"\n📋 前5行数据:")
    logger.info(f"\n{df.head().to_string()}")


def preprocess_for_training(df, output_path='data/processed/uci_household_clean.csv'):
    """
    预处理数据用于训练
    
    处理步骤:
    1. 处理缺失值
    2. 重采样到15分钟（论文中使用的频率）
    3. 特征工程
    4. 保存清洗后的数据
    """
    logger.info("\n" + "="*70)
    logger.info("数据预处理")
    logger.info("="*70)
    
    df_clean = df.copy()
    
    # 1. 处理缺失值
    logger.info("\n1️⃣ 处理缺失值...")
    initial_missing = df_clean.isnull().sum().sum()
    
    # 前向填充
    df_clean = df_clean.fillna(method='ffill')
    # 后向填充（处理开头的缺失）
    df_clean = df_clean.fillna(method='bfill')
    
    remaining_missing = df_clean.isnull().sum().sum()
    logger.info(f"  处理前: {initial_missing:,} 缺失值")
    logger.info(f"  处理后: {remaining_missing:,} 缺失值")
    
    # 2. 设置时间索引并重采样
    if 'datetime' in df_clean.columns:
        logger.info("\n2️⃣ 重采样到15分钟...")
        df_clean = df_clean.set_index('datetime')
        
        # 重采样（论文中使用15分钟）
        df_clean = df_clean.resample('15T').mean()
        logger.info(f"  重采样后样本数: {len(df_clean):,}")
        
        # 重置索引
        df_clean = df_clean.reset_index()
    
    # 3. 特征工程
    logger.info("\n3️⃣ 特征工程...")
    if 'datetime' in df_clean.columns:
        df_clean['hour'] = df_clean['datetime'].dt.hour
        df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek
        df_clean['month'] = df_clean['datetime'].dt.month
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
        logger.info(f"  添加时间特征: hour, day_of_week, month, is_weekend")
    
    # 4. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"\n✅ 保存清洗数据: {output_path}")
    logger.info(f"  最终形状: {df_clean.shape}")
    
    return df_clean


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载UCI Household数据集')
    parser.add_argument(
        '--method',
        choices=['api', 'direct'],
        default='api',
        help='下载方式: api=ucimlrepo包, direct=直接下载ZIP'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='原始数据输出目录'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='是否进行预处理'
    )
    parser.add_argument(
        '--processed-output',
        default='data/processed/uci_household_clean.csv',
        help='预处理后的数据输出路径'
    )
    
    args = parser.parse_args()
    
    try:
        # 下载数据
        df = download_uci_dataset(
            output_dir=args.output_dir,
            method=args.method
        )
        
        # 分析数据
        analyze_dataset(df)
        
        # 预处理
        if args.preprocess:
            df_clean = preprocess_for_training(
                df,
                output_path=args.processed_output
            )
            
            logger.info("\n" + "="*70)
            logger.info("✅ 全部完成！")
            logger.info("="*70)
            logger.info(f"原始数据: {args.output_dir}/uci_household_raw.csv")
            logger.info(f"清洗数据: {args.processed_output}")
            logger.info("\n可以使用清洗后的数据进行训练:")
            logger.info(f"  python scripts/run_training.py --data {args.processed_output}")
        
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
