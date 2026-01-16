"""
对比UCI真实数据集和合成数据集

展示两个数据集的统计差异
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_analyze(filepath, name):
    """加载并分析数据集"""
    logger.info(f"\n{'='*70}")
    logger.info(f"分析 {name}")
    logger.info(f"{'='*70}")
    
    df = pd.read_csv(filepath)
    
    logger.info(f"📊 基本信息:")
    logger.info(f"  文件路径: {filepath}")
    logger.info(f"  样本数: {len(df):,}")
    logger.info(f"  特征数: {len(df.columns)}")
    logger.info(f"  列名: {df.columns.tolist()}")
    
    # 数值型列的统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"\n📈 数值特征统计:")
    
    for col in numeric_cols:
        if col in df.columns:
            stats = df[col].describe()
            logger.info(f"\n  {col}:")
            logger.info(f"    均值: {stats['mean']:.2f}")
            logger.info(f"    标准差: {stats['std']:.2f}")
            logger.info(f"    最小值: {stats['min']:.2f}")
            logger.info(f"    最大值: {stats['max']:.2f}")
    
    return df


def main():
    """主函数"""
    
    # 1. 合成数据
    synthetic_df = load_and_analyze(
        'data/synthetic/training_data.csv',
        '合成数据集 (Synthetic Data)'
    )
    
    # 2. UCI真实数据
    uci_df = load_and_analyze(
        'data/processed/uci_household_clean.csv',
        'UCI真实数据集 (Real Data)'
    )
    
    # 3. 对比
    logger.info(f"\n{'='*70}")
    logger.info(f"数据集对比")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n📊 规模对比:")
    logger.info(f"  合成数据: {len(synthetic_df):,} 样本")
    logger.info(f"  UCI数据: {len(uci_df):,} 样本")
    logger.info(f"  UCI / 合成: {len(uci_df) / len(synthetic_df):.1f}x")
    
    # 对比功率特征
    if 'EDP' in synthetic_df.columns and 'Global_active_power' in uci_df.columns:
        logger.info(f"\n⚡ 功率对比:")
        logger.info(f"  合成数据 EDP:")
        logger.info(f"    均值: {synthetic_df['EDP'].mean():.2f} kWh")
        logger.info(f"    标准差: {synthetic_df['EDP'].std():.2f}")
        logger.info(f"    范围: [{synthetic_df['EDP'].min():.2f}, {synthetic_df['EDP'].max():.2f}]")
        
        logger.info(f"\n  UCI数据 Global_active_power:")
        logger.info(f"    均值: {uci_df['Global_active_power'].mean():.2f} kW")
        logger.info(f"    标准差: {uci_df['Global_active_power'].std():.2f}")
        logger.info(f"    范围: [{uci_df['Global_active_power'].min():.2f}, {uci_df['Global_active_power'].max():.2f}]")
    
    # 对比温度（如果有）
    if 'temperature' in synthetic_df.columns:
        logger.info(f"\n🌡️  温度对比:")
        logger.info(f"  合成数据:")
        logger.info(f"    均值: {synthetic_df['temperature'].mean():.2f}°C")
        logger.info(f"    范围: [{synthetic_df['temperature'].min():.2f}, {synthetic_df['temperature'].max():.2f}]")
        logger.info(f"\n  注: UCI数据集不包含温度信息")
    
    logger.info(f"\n💡 总结:")
    logger.info(f"  ✅ 合成数据: 快速生成，可控参数，适合开发测试")
    logger.info(f"  ✅ UCI数据: 真实世界数据，适合发表论文和实际部署")
    logger.info(f"\n推荐:")
    logger.info(f"  - 开发阶段: 使用合成数据 (快速迭代)")
    logger.info(f"  - 最终评估: 使用UCI数据 (可信结果)")


if __name__ == "__main__":
    main()
