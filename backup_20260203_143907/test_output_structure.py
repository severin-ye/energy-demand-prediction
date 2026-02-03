#!/usr/bin/env python3
"""
测试输出目录结构
"""

from pathlib import Path
from datetime import datetime

def test_training_dir():
    """测试训练目录生成"""
    date_suffix = datetime.now().strftime('%y-%m-%d')
    output_dir = f'./outputs/training/{date_suffix}'
    print(f"训练目录: {output_dir}")
    return output_dir

def test_inference_dir(model_dir):
    """测试推理目录生成"""
    model_dir_path = Path(model_dir)
    
    # 从模型目录提取名称
    if model_dir_path.parent.name == 'models':
        # 如果是 xxx/models，取上一级目录名
        model_name = model_dir_path.parent.parent.name
    else:
        model_name = model_dir_path.parent.name
    
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M')
    output_dir = f'outputs/inference/{model_name}/{timestamp}'
    
    print(f"\n模型目录: {model_dir}")
    print(f"提取模型名: {model_name}")
    print(f"推理目录: {output_dir}")
    return output_dir

if __name__ == '__main__':
    print("=" * 60)
    print("输出目录结构测试")
    print("=" * 60)
    
    # 测试训练目录
    train_dir = test_training_dir()
    
    # 测试推理目录（多种情况）
    test_cases = [
        'outputs/training/26-01-16/models',
        './outputs/training/26-01-17/models',
        'custom_models/my_model/models',
        'another/path/to/models',
    ]
    
    for model_dir in test_cases:
        test_inference_dir(model_dir)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
