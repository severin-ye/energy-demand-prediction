"""
灵活的合成数据生成脚本
支持生成训练数据和各种测试场景数据
"""

import numpy as np
import pandas as pd
import argparse
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EnergyDataGenerator:
    """能源数据合成器"""
    
    def __init__(self, seed=42):
        """
        参数:
        - seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_training_data(
        self,
        n_samples=2000,
        start_date='2024-01-01',
        freq='H'
    ):
        """
        生成训练数据集（多样化的正常数据）
        
        参数:
        - n_samples: 样本数量
        - start_date: 起始日期
        - freq: 采样频率 ('H'=小时, 'D'=天)
        
        返回:
        - DataFrame with columns: Temperature, Humidity, WindSpeed, EDP, Hour, DayOfWeek, Month
        """
        logger.info(f"生成训练数据: {n_samples} 样本")
        
        # 生成时间序列
        dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
        
        data = []
        for i, date in enumerate(dates):
            # 时间特征
            hour = date.hour
            day_of_week = date.dayofweek
            month = date.month
            day_of_year = date.dayofyear
            
            # 温度：基准20°C + 季节变化 + 日变化 + 噪声
            seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365)
            daily_temp = 5 * np.sin(2 * np.pi * hour / 24)
            temperature = 20 + seasonal_temp + daily_temp + np.random.randn() * 2
            
            # 湿度：与温度负相关
            humidity = 70 - 0.8 * seasonal_temp - 0.5 * daily_temp + np.random.randn() * 5
            humidity = np.clip(humidity, 20, 95)
            
            # 风速：周期性变化 + 随机
            wind_speed = 10 + 5 * np.sin(2 * np.pi * day_of_week / 7) + np.random.randn() * 3
            wind_speed = np.clip(wind_speed, 0, 25)
            
            # EDP：多因素模型
            # 基础负荷
            base_load = 100
            
            # 温度影响（高温高能耗，低温低能耗）
            temp_effect = 2 * abs(temperature - 20)
            
            # 湿度影响
            humidity_effect = 0.5 * (humidity - 50)
            
            # 风速影响（负相关）
            wind_effect = -1.5 * (wind_speed - 10)
            
            # 时间影响（峰谷）
            if 9 <= hour <= 11 or 18 <= hour <= 20:
                time_effect = 30  # 峰值
            elif 1 <= hour <= 5:
                time_effect = -20  # 低谷
            else:
                time_effect = 0
            
            # 周末影响
            weekend_effect = -15 if day_of_week >= 5 else 0
            
            # 综合EDP
            edp = (base_load + temp_effect + humidity_effect + 
                   wind_effect + time_effect + weekend_effect + 
                   np.random.randn() * 10)
            edp = max(50, edp)  # 最小值
            
            data.append({
                'Temperature': temperature,
                'Humidity': humidity,
                'WindSpeed': wind_speed,
                'EDP': edp,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'Month': month
            })
        
        df = pd.DataFrame(data)
        logger.info(f"训练数据统计:")
        logger.info(f"  Temperature: {df['Temperature'].mean():.2f} ± {df['Temperature'].std():.2f}")
        logger.info(f"  Humidity: {df['Humidity'].mean():.2f} ± {df['Humidity'].std():.2f}")
        logger.info(f"  WindSpeed: {df['WindSpeed'].mean():.2f} ± {df['WindSpeed'].std():.2f}")
        logger.info(f"  EDP: {df['EDP'].mean():.2f} ± {df['EDP'].std():.2f}")
        
        return df
    
    def generate_scenario(
        self,
        scenario_type='normal',
        duration=30,
        start_hour=0,
        **kwargs
    ):
        """
        生成特定场景的测试数据
        
        参数:
        - scenario_type: 场景类型
          * 'high_temp_humid': 高温高湿
          * 'low_temp_humid': 低温低湿
          * 'moderate': 适中温度
          * 'peak_hour': 高峰时段
          * 'valley_hour': 低谷时段
          * 'heatwave': 热浪
          * 'coldwave': 寒潮
          * 'custom': 自定义（需要提供参数）
        - duration: 时间步数（小时）
        - start_hour: 起始小时
        - **kwargs: 自定义参数
          * temp_base, temp_variation
          * humid_base, humid_variation
          * wind_base, wind_variation
        
        返回:
        - DataFrame
        """
        logger.info(f"生成场景: {scenario_type}, 时长={duration}小时")
        
        # 预定义场景参数
        scenarios = {
            'high_temp_humid': {
                'temp_base': 32, 'temp_variation': 2,
                'humid_base': 75, 'humid_variation': 5,
                'wind_base': 3, 'wind_variation': 1,
                'description': '高温高湿场景（夏季午后）'
            },
            'low_temp_humid': {
                'temp_base': 12, 'temp_variation': 2,
                'humid_base': 40, 'humid_variation': 5,
                'wind_base': 8, 'wind_variation': 2,
                'description': '低温低湿场景（冬季清晨）'
            },
            'moderate': {
                'temp_base': 20, 'temp_variation': 3,
                'humid_base': 55, 'humid_variation': 5,
                'wind_base': 5, 'wind_variation': 2,
                'description': '适中温度场景（春秋季）'
            },
            'peak_hour': {
                'temp_base': 28, 'temp_variation': 2,
                'humid_base': 65, 'humid_variation': 5,
                'wind_base': 2, 'wind_variation': 1,
                'description': '高峰时段场景（傍晚用电高峰）'
            },
            'valley_hour': {
                'temp_base': 18, 'temp_variation': 2,
                'humid_base': 50, 'humid_variation': 5,
                'wind_base': 6, 'wind_variation': 2,
                'description': '低谷时段场景（深夜）'
            },
            'heatwave': {
                'temp_base': 38, 'temp_variation': 3,
                'humid_base': 80, 'humid_variation': 5,
                'wind_base': 1, 'wind_variation': 0.5,
                'description': '热浪场景（极端高温）'
            },
            'coldwave': {
                'temp_base': 5, 'temp_variation': 2,
                'humid_base': 35, 'humid_variation': 5,
                'wind_base': 12, 'wind_variation': 3,
                'description': '寒潮场景（极端低温）'
            }
        }
        
        # 获取场景参数
        if scenario_type == 'custom':
            params = kwargs
            description = '自定义场景'
        elif scenario_type in scenarios:
            params = scenarios[scenario_type]
            description = params.pop('description', scenario_type)
        else:
            raise ValueError(f"未知场景类型: {scenario_type}")
        
        logger.info(f"  {description}")
        
        # 提取参数
        temp_base = params.get('temp_base', 20)
        temp_var = params.get('temp_variation', 3)
        humid_base = params.get('humid_base', 55)
        humid_var = params.get('humid_variation', 5)
        wind_base = params.get('wind_base', 5)
        wind_var = params.get('wind_variation', 2)
        
        data = []
        for h in range(duration):
            hour = (start_hour + h) % 24
            
            # 添加日变化
            daily_cycle = np.sin(2 * np.pi * h / 24)
            
            temperature = temp_base + temp_var * daily_cycle + np.random.randn() * 0.5
            humidity = humid_base + humid_var * np.cos(2 * np.pi * h / 24) + np.random.randn() * 1
            humidity = np.clip(humidity, 20, 95)
            wind_speed = wind_base + wind_var * np.random.randn() * 0.5
            wind_speed = np.clip(wind_speed, 0, 25)
            
            data.append({
                'Temperature': temperature,
                'Humidity': humidity,
                'WindSpeed': wind_speed,
                'EDP': 0.0,  # 占位符，由模型预测
                'Hour': hour,
                'DayOfWeek': kwargs.get('day_of_week', 2),
                'Month': kwargs.get('month', 7)
            })
        
        df = pd.DataFrame(data)
        logger.info(f"  Temperature: {df['Temperature'].min():.1f} ~ {df['Temperature'].max():.1f}°C")
        logger.info(f"  Humidity: {df['Humidity'].min():.1f} ~ {df['Humidity'].max():.1f}%")
        logger.info(f"  WindSpeed: {df['WindSpeed'].min():.1f} ~ {df['WindSpeed'].max():.1f}m/s")
        
        return df
    
    def generate_multiple_scenarios(self, scenarios_config):
        """
        批量生成多个场景
        
        参数:
        - scenarios_config: List[Dict] 场景配置列表
          例: [{'name': 'scene1', 'type': 'high_temp_humid', 'duration': 30}, ...]
        
        返回:
        - Dict[str, DataFrame]
        """
        results = {}
        for config in scenarios_config:
            name = config.pop('name')
            scenario_type = config.pop('type', 'moderate')
            df = self.generate_scenario(scenario_type=scenario_type, **config)
            results[name] = df
        
        return results


def main():
    parser = argparse.ArgumentParser(description='灵活的能源数据合成生成器')
    parser.add_argument('--mode', type=str, default='training',
                       choices=['training', 'scenario', 'batch'],
                       help='生成模式: training(训练数据), scenario(单场景), batch(多场景)')
    parser.add_argument('--output', type=str, default='data/synthetic',
                       help='输出目录')
    
    # 训练数据参数
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='训练样本数量')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='起始日期')
    
    # 场景参数
    parser.add_argument('--scenario-type', type=str, default='moderate',
                       choices=['high_temp_humid', 'low_temp_humid', 'moderate',
                               'peak_hour', 'valley_hour', 'heatwave', 'coldwave', 'custom'],
                       help='场景类型')
    parser.add_argument('--duration', type=int, default=30,
                       help='场景时长（小时）')
    parser.add_argument('--start-hour', type=int, default=0,
                       help='起始小时')
    
    # 自定义参数
    parser.add_argument('--temp-base', type=float, help='温度基准值')
    parser.add_argument('--humid-base', type=float, help='湿度基准值')
    parser.add_argument('--wind-base', type=float, help='风速基准值')
    
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化生成器
    generator = EnergyDataGenerator(seed=args.seed)
    
    logger.info("=" * 70)
    logger.info("能源数据合成生成器")
    logger.info("=" * 70)
    logger.info("")
    
    if args.mode == 'training':
        # 生成训练数据
        logger.info(f"模式: 训练数据生成")
        df = generator.generate_training_data(
            n_samples=args.n_samples,
            start_date=args.start_date
        )
        
        output_path = os.path.join(args.output, 'training_data.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"✅ 训练数据已保存: {output_path}")
        logger.info(f"   样本数: {len(df)}, 特征数: {len(df.columns)}")
    
    elif args.mode == 'scenario':
        # 生成单个场景
        logger.info(f"模式: 场景数据生成")
        
        kwargs = {}
        if args.temp_base is not None:
            kwargs['temp_base'] = args.temp_base
        if args.humid_base is not None:
            kwargs['humid_base'] = args.humid_base
        if args.wind_base is not None:
            kwargs['wind_base'] = args.wind_base
        
        df = generator.generate_scenario(
            scenario_type=args.scenario_type,
            duration=args.duration,
            start_hour=args.start_hour,
            **kwargs
        )
        
        output_path = os.path.join(args.output, f'scenario_{args.scenario_type}.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"✅ 场景数据已保存: {output_path}")
        logger.info(f"   时间步: {len(df)}")
    
    elif args.mode == 'batch':
        # 生成多个预定义场景
        logger.info(f"模式: 批量场景生成")
        
        scenarios_config = [
            {'name': 'high_temp_humid', 'type': 'high_temp_humid', 'duration': 30},
            {'name': 'low_temp_humid', 'type': 'low_temp_humid', 'duration': 30},
            {'name': 'moderate', 'type': 'moderate', 'duration': 30},
            {'name': 'peak_hour', 'type': 'peak_hour', 'duration': 24, 'start_hour': 17},
            {'name': 'valley_hour', 'type': 'valley_hour', 'duration': 24, 'start_hour': 1},
            {'name': 'heatwave', 'type': 'heatwave', 'duration': 48},
            {'name': 'coldwave', 'type': 'coldwave', 'duration': 48}
        ]
        
        results = generator.generate_multiple_scenarios(scenarios_config)
        
        for name, df in results.items():
            output_path = os.path.join(args.output, f'{name}.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"✅ {name}: {output_path} ({len(df)} 时间步)")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("生成完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
