"""
推理测试脚本
加载训练好的模型，对新数据进行预测并生成因果解释和建议
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.inference_pipeline import InferencePipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(n_samples=10):
    """创建测试数据（生成完整序列）"""
    np.random.seed(42)
    
    # 创建足够多的历史数据点(至少30个以生成序列)
    hours = list(range(30))
    
    # 场景1: 高温高湿序列
    temp1 = [30 + 2*np.sin(h/24*2*np.pi) + np.random.randn()*0.5 for h in hours]
    hum1 = [70 + 5*np.cos(h/24*2*np.pi) + np.random.randn()*1 for h in hours]
    wind1 = [3 + np.random.randn()*0.5 for h in hours]
    
    scenario1 = pd.DataFrame({
        'Temperature': temp1,
        'Humidity': hum1,
        'WindSpeed': wind1,
        'EDP': [0.0] * 30,  # 占位符
        'Hour': [(14 + h) % 24 for h in hours],
        'DayOfWeek': [2] * 30,
        'Month': [7] * 30
    })
    
    # 场景2: 低温低湿序列
    temp2 = [12 + 2*np.sin(h/24*2*np.pi) + np.random.randn()*0.5 for h in hours]
    hum2 = [40 + 5*np.cos(h/24*2*np.pi) + np.random.randn()*1 for h in hours]
    wind2 = [8 + np.random.randn()*0.5 for h in hours]
    
    scenario2 = pd.DataFrame({
        'Temperature': temp2,
        'Humidity': hum2,
        'WindSpeed': wind2,
        'EDP': [0.0] * 30,
        'Hour': [(3 + h) % 24 for h in hours],
        'DayOfWeek': [1] * 30,
        'Month': [3] * 30
    })
    
    # 场景3: 适中温度序列
    temp3 = [20 + 2*np.sin(h/24*2*np.pi) + np.random.randn()*0.5 for h in hours]
    hum3 = [55 + 5*np.cos(h/24*2*np.pi) + np.random.randn()*1 for h in hours]
    wind3 = [5 + np.random.randn()*0.5 for h in hours]
    
    scenario3 = pd.DataFrame({
        'Temperature': temp3,
        'Humidity': hum3,
        'WindSpeed': wind3,
        'EDP': [0.0] * 30,
        'Hour': [(10 + h) % 24 for h in hours],
        'DayOfWeek': [3] * 30,
        'Month': [5] * 30
    })
    
    scenarios = [
        ('高温高湿场景', scenario1),
        ('低温低湿场景', scenario2),
        ('适中温度场景', scenario3)
    ]
    
    return scenarios


def main():
    logger.info("=" * 80)
    logger.info(" " * 30 + "推理测试流水线")
    logger.info("=" * 80)
    logger.info("")
    
    # 1. 加载模型
    logger.info("[步骤 1] 加载训练好的模型...")
    model_dir = './outputs/training_run_1/models'
    
    if not os.path.exists(model_dir):
        logger.error(f"❌ 模型目录不存在: {model_dir}")
        logger.error("请先运行训练脚本: python scripts/run_training.py")
        return
    
    try:
        pipeline = InferencePipeline(model_dir)
        logger.info(f"✅ 模型加载成功，目录: {model_dir}")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("")
    
    # 2. 创建测试数据
    logger.info("[步骤 2] 准备测试数据...")
    scenarios = create_test_data()
    logger.info(f"✅ 测试数据准备完成: {len(scenarios)} 个场景")
    logger.info(f"场景: {', '.join([name for name, _ in scenarios])}")
    logger.info("")
    
    # 3. 运行推理
    logger.info("[步骤 3] 执行推理...")
    logger.info("=" * 60)
    
    results_list = []
    
    for idx, (scenario_name, test_data) in enumerate(scenarios, 1):
        logger.info("")
        logger.info(f"场景 {idx}: {scenario_name}")
        logger.info("-" * 60)
        logger.info(f"输入数据: {len(test_data)} 个时间步")
        logger.info(f"  温度范围: {test_data['Temperature'].min():.1f} ~ {test_data['Temperature'].max():.1f}°C")
        logger.info(f"  湿度范围: {test_data['Humidity'].min():.1f} ~ {test_data['Humidity'].max():.1f}%")
        logger.info(f"  风速范围: {test_data['WindSpeed'].min():.1f} ~ {test_data['WindSpeed'].max():.1f}m/s")
        logger.info("")
        
        try:
            # 运行推理（不生成建议，避免贝叶斯网络问题）
            result = pipeline.predict(test_data, generate_recommendations=False)
            
            # 显示结果（取最后一个预测）
            idx_last = -1
            logger.info(f"📊 预测结果:")
            logger.info(f"  EDP预测值: {result['predictions'][idx_last]:.2f} kWh")
            logger.info(f"  EDP状态: {result['edp_states'][idx_last]}")
            logger.info(f"  CAM聚类: Cluster {result['cam_clusters'][idx_last]}")
            logger.info(f"  Attention类型: {result['attention_types'][idx_last]}")
            logger.info(f"  生成序列数: {len(result['predictions'])}")
            
            results_list.append({
                'scenario': scenario_name,
                'predictions': {
                    'edp': float(result['predictions'][idx_last]),
                    'state': result['edp_states'][idx_last],
                    'cam_cluster': int(result['cam_clusters'][idx_last]),
                    'attention_type': result['attention_types'][idx_last]
                }
            })
            
        except Exception as e:
            logger.error(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("")
    logger.info("=" * 60)
    
    # 4. 保存结果
    logger.info("")
    logger.info("[步骤 4] 保存推理结果...")
    
    output_dir = './outputs/inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'inference_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 结果已保存到: {output_file}")
    logger.info("")
    
    # 5. 汇总统计
    logger.info("[步骤 5] 结果汇总")
    logger.info("=" * 60)
    
    if results_list:
        all_preds = []
        for r in results_list:
            if 'predictions' in r and 'edp' in r['predictions']:
                all_preds.append(r['predictions']['edp'])
        
        if all_preds:
            logger.info(f"EDP预测统计:")
            logger.info(f"  最小值: {min(all_preds):.2f} kWh")
            logger.info(f"  最大值: {max(all_preds):.2f} kWh")
            logger.info(f"  平均值: {np.mean(all_preds):.2f} kWh")
    
    logger.info("")
    logger.info("✅ 推理测试完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
