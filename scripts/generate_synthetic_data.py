"""
Flexible Synthetic Data Generation Script
Supports generating training data and various test scenario data
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
    """Energy Data Synthesizer"""
    
    def __init__(self, seed=42):
        """
        Parameters:
        - seed: Random seed
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
        Generates a training dataset (diverse normal data)
        
        Parameters:
        - n_samples: Number of samples
        - start_date: Starting date
        - freq: Sampling frequency ('H'=Hour, 'D'=Day)
        
        Returns:
        - DataFrame with columns: Temperature, Humidity, WindSpeed, EDP, Hour, DayOfWeek, Month
        """
        logger.info(f"Generating training data: {n_samples} samples")
        
        # Generate time series
        dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
        
        data = []
        for i, date in enumerate(dates):
            # Temporal features
            hour = date.hour
            day_of_week = date.dayofweek
            month = date.month
            day_of_year = date.dayofyear
            
            # Temperature: Base 20°C + Seasonal variation + Diurnal variation + Noise
            seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365)
            daily_temp = 5 * np.sin(2 * np.pi * hour / 24)
            temperature = 20 + seasonal_temp + daily_temp + np.random.randn() * 2
            
            # Humidity: Negatively correlated with temperature
            humidity = 70 - 0.8 * seasonal_temp - 0.5 * daily_temp + np.random.randn() * 5
            humidity = np.clip(humidity, 20, 95)
            
            # Wind Speed: Periodic variation + Random noise
            wind_speed = 10 + 5 * np.sin(2 * np.pi * day_of_week / 7) + np.random.randn() * 3
            wind_speed = np.clip(wind_speed, 0, 25)
            
            # EDP: Multi-factor model
            # Baseload
            base_load = 100
            
            # Temperature effect (High temp = high consumption, Low temp = low consumption)
            temp_effect = 2 * abs(temperature - 20)
            
            # Humidity effect
            humidity_effect = 0.5 * (humidity - 50)
            
            # Wind speed effect (Negative correlation)
            wind_effect = -1.5 * (wind_speed - 10)
            
            # Time of day effect (Peaks and Valleys)
            if 9 <= hour <= 11 or 18 <= hour <= 20:
                time_effect = 30  # Peak
            elif 1 <= hour <= 5:
                time_effect = -20  # Valley
            else:
                time_effect = 0
            
            # Weekend effect
            weekend_effect = -15 if day_of_week >= 5 else 0
            
            # Comprehensive EDP
            edp = (base_load + temp_effect + humidity_effect + 
                   wind_effect + time_effect + weekend_effect + 
                   np.random.randn() * 10)
            edp = max(50, edp)  # Minimum value constraint
            
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
        logger.info(f"Training Data Statistics:")
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
        Generates test data for specific scenarios
        
        Parameters:
        - scenario_type: Scenario Type
          * 'high_temp_humid': High Temp & High Humidity
          * 'low_temp_humid': Low Temp & Low Humidity
          * 'moderate': Moderate Temperature
          * 'peak_hour': Peak Demand Hours
          * 'valley_hour': Off-peak Valley Hours
          * 'heatwave': Heatwave
          * 'coldwave': Coldwave
          * 'custom': Custom (requires extra parameters)
        - duration: Number of time steps (hours)
        - start_hour: Starting hour
        - **kwargs: Custom parameters
          * temp_base, temp_variation
          * humid_base, humid_variation
          * wind_base, wind_variation
        
        Returns:
        - DataFrame
        """
        logger.info(f"Generating Scenario: {scenario_type}, Duration={duration} hours")
        
        # Predefined scenario parameters
        scenarios = {
            'high_temp_humid': {
                'temp_base': 32, 'temp_variation': 2,
                'humid_base': 75, 'humid_variation': 5,
                'wind_base': 3, 'wind_variation': 1,
                'description': 'High Temp & Humid (Summer Afternoon)'
            },
            'low_temp_humid': {
                'temp_base': 12, 'temp_variation': 2,
                'humid_base': 40, 'humid_variation': 5,
                'wind_base': 8, 'wind_variation': 2,
                'description': 'Low Temp & Low Humidity (Winter Early Morning)'
            },
            'moderate': {
                'temp_base': 20, 'temp_variation': 3,
                'humid_base': 55, 'humid_variation': 5,
                'wind_base': 5, 'wind_variation': 2,
                'description': 'Moderate Temperature (Spring/Autumn)'
            },
            'peak_hour': {
                'temp_base': 28, 'temp_variation': 2,
                'humid_base': 65, 'humid_variation': 5,
                'wind_base': 2, 'wind_variation': 1,
                'description': 'Peak Hour (Evening Electricity Peak)'
            },
            'valley_hour': {
                'temp_base': 18, 'temp_variation': 2,
                'humid_base': 50, 'humid_variation': 5,
                'wind_base': 6, 'wind_variation': 2,
                'description': 'Valley Hour (Late Night)'
            },
            'heatwave': {
                'temp_base': 38, 'temp_variation': 3,
                'humid_base': 80, 'humid_variation': 5,
                'wind_base': 1, 'wind_variation': 0.5,
                'description': 'Heatwave (Extreme High Temp)'
            },
            'coldwave': {
                'temp_base': 5, 'temp_variation': 2,
                'humid_base': 35, 'humid_variation': 5,
                'wind_base': 12, 'wind_variation': 3,
                'description': 'Coldwave (Extreme Low Temp)'
            }
        }
        
        # Fetch scenario parameters
        if scenario_type == 'custom':
            params = kwargs
            description = 'Custom Scenario'
        elif scenario_type in scenarios:
            params = scenarios[scenario_type]
            description = params.pop('description', scenario_type)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        logger.info(f"  {description}")
        
        # Extract parameters
        temp_base = params.get('temp_base', 20)
        temp_var = params.get('temp_variation', 3)
        humid_base = params.get('humid_base', 55)
        humid_var = params.get('humid_variation', 5)
        wind_base = params.get('wind_base', 5)
        wind_var = params.get('wind_variation', 2)
        
        data = []
        for h in range(duration):
            hour = (start_hour + h) % 24
            
            # Add diurnal cycle
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
                'EDP': 0.0,  # Placeholder, predicted by model
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
        Batch generate multiple scenarios
        
        Parameters:
        - scenarios_config: List[Dict] List of scenario configurations
          Example: [{'name': 'scene1', 'type': 'high_temp_humid', 'duration': 30}, ...]
        
        Returns:
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
    parser = argparse.ArgumentParser(description='Flexible Energy Data Synthetic Generator')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'scenario', 'batch'],
                        help='Generation Mode: training (train data), scenario (single), batch (multiple)')
    parser.add_argument('--output', type=str, default='data/synthetic',
                        help='Output directory')
    
    # Training Data Parameters
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of training samples')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                        help='Start date')
    
    # Scenario Parameters
    parser.add_argument('--scenario-type', type=str, default='moderate',
                        choices=['high_temp_humid', 'low_temp_humid', 'moderate',
                                 'peak_hour', 'valley_hour', 'heatwave', 'coldwave', 'custom'],
                        help='Scenario Type')
    parser.add_argument('--duration', type=int, default=30,
                        help='Scenario duration (hours)')
    parser.add_argument('--start-hour', type=int, default=0,
                        help='Start hour')
    
    # Custom Parameters
    parser.add_argument('--temp-base', type=float, help='Baseline temperature')
    parser.add_argument('--humid-base', type=float, help='Baseline humidity')
    parser.add_argument('--wind-base', type=float, help='Baseline wind speed')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize generator
    generator = EnergyDataGenerator(seed=args.seed)
    
    logger.info("=" * 70)
    logger.info("Energy Data Synthetic Generator")
    logger.info("=" * 70)
    logger.info("")
    
    if args.mode == 'training':
        # Generate training data
        logger.info(f"Mode: Training Data Generation")
        df = generator.generate_training_data(
            n_samples=args.n_samples,
            start_date=args.start_date
        )
        
        output_path = os.path.join(args.output, 'training_data.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Training data saved: {output_path}")
        logger.info(f"   Samples: {len(df)}, Features: {len(df.columns)}")
    
    elif args.mode == 'scenario':
        # Generate single scenario
        logger.info(f"Mode: Scenario Data Generation")
        
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
        logger.info(f"✅ Scenario data saved: {output_path}")
        logger.info(f"   Time steps: {len(df)}")
    
    elif args.mode == 'batch':
        # Generate multiple predefined scenarios
        logger.info(f"Mode: Batch Scenario Generation")
        
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
            logger.info(f"✅ {name}: {output_path} ({len(df)} time steps)")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Generation Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()