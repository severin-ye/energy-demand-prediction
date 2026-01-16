#!/usr/bin/env python3
"""
演示：为什么样本数太少会导致推理错误

问题根源：序列长度要求
"""

import numpy as np

def create_sequences_demo(data, sequence_length=20):
    """
    模拟 create_sequences 函数的行为
    
    参数:
        data: 输入数据 [样本数,]
        sequence_length: 序列长度（滑动窗口大小）
    
    返回:
        序列数组和对应的目标值
    """
    print(f"\n{'='*60}")
    print(f"输入数据: {len(data)} 个样本")
    print(f"序列长度: {sequence_length}")
    print(f"{'='*60}\n")
    
    X, y = [], []
    
    # 这是关键循环：range(len(data) - sequence_length)
    max_index = len(data) - sequence_length
    print(f"循环范围: range(0, {max_index})")
    print(f"  → 可生成的序列数: {max_index}\n")
    
    if max_index <= 0:
        print(f"❌ 错误！无法生成序列")
        print(f"   原因: 样本数({len(data)}) ≤ 序列长度({sequence_length})")
        print(f"   需要: 样本数 > {sequence_length}")
        return None, None
    
    print(f"✅ 可以生成 {max_index} 个序列\n")
    print("序列构建过程:")
    
    for i in range(max_index):
        # 取窗口 [i : i+sequence_length]
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        
        X.append(sequence)
        y.append(target)
        
        if i < 3:  # 只显示前3个示例
            print(f"  序列 {i}: data[{i}:{i+sequence_length}] → 目标 data[{i+sequence_length}]")
            print(f"          值: {sequence} → {target}")
    
    if max_index > 3:
        print(f"  ... ({max_index - 3} 个序列省略)")
    
    return np.array(X), np.array(y)


def test_cases():
    """测试不同样本数的情况"""
    
    print("\n" + "="*70)
    print("测试案例：不同样本数的序列生成")
    print("="*70)
    
    sequence_length = 20
    
    # 测试案例1: 样本数太少（10个）
    print("\n【案例 1】样本数太少")
    data1 = np.arange(10)
    X1, y1 = create_sequences_demo(data1, sequence_length)
    
    # 测试案例2: 样本数刚好等于序列长度（20个）
    print("\n\n【案例 2】样本数 = 序列长度")
    data2 = np.arange(20)
    X2, y2 = create_sequences_demo(data2, sequence_length)
    
    # 测试案例3: 样本数略多于序列长度（25个）
    print("\n\n【案例 3】样本数略多（25个）")
    data3 = np.arange(25)
    X3, y3 = create_sequences_demo(data3, sequence_length)
    
    # 测试案例4: 样本数充足（50个）
    print("\n\n【案例 4】样本数充足（50个）")
    data4 = np.arange(50)
    X4, y4 = create_sequences_demo(data4, sequence_length)
    
    # 总结
    print("\n\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"\n给定序列长度 = {sequence_length}:\n")
    print(f"  • 样本数 = 10  → 序列数 = 0  ❌ (太少，无法生成)")
    print(f"  • 样本数 = 20  → 序列数 = 0  ❌ (刚好，但仍无法生成)")
    print(f"  • 样本数 = 21  → 序列数 = 1  ⚠️  (勉强可用)")
    print(f"  • 样本数 = 25  → 序列数 = 5  ⚠️  (太少)")
    print(f"  • 样本数 = 30  → 序列数 = 10 ✅ (可用)")
    print(f"  • 样本数 = 50  → 序列数 = 30 ✅ (推荐)")
    print(f"\n公式: 序列数 = max(0, 样本数 - 序列长度)")
    print(f"\n推荐: 样本数 ≥ {sequence_length + 30} (至少能生成30个序列)")


def explain_batch_outputs_error():
    """解释 batch_outputs 错误"""
    
    print("\n\n" + "="*70)
    print("为什么会出现 'batch_outputs' 错误？")
    print("="*70)
    
    print("""
当样本数太少时的执行流程：

1️⃣  用户输入: 20 个样本
2️⃣  预处理器创建序列:
    - sequence_length = 20
    - 可生成序列数 = 20 - 20 = 0
    - 返回: X=[], y=[]  (空数组！)

3️⃣  模型预测:
    - model.predict(X) 其中 X.shape = (0, 20, 3)
    - Keras 发现没有数据可预测
    - 循环体从未执行
    - batch_outputs 变量从未被赋值

4️⃣  访问变量:
    - 代码尝试访问 batch_outputs
    - UnboundLocalError: cannot access local variable 'batch_outputs'

解决方案:
  ✅ 确保样本数 > sequence_length
  ✅ 推荐: 样本数 ≥ sequence_length + 30
  ✅ 对于 sequence_length=20，建议至少 50 个样本
""")


if __name__ == '__main__':
    test_cases()
    explain_batch_outputs_error()
    
    print("\n" + "="*70)
    print("实际项目中的最小样本数要求")
    print("="*70)
    print("""
训练配置: sequence_length = 20

推理时的样本数要求:
  • 最小值: 21 个样本 (生成 1 个序列)
  • 安全值: 50 个样本 (生成 30 个序列) ✅
  • 推荐值: 100+ 个样本 (生成 80+ 个序列) ✅✅

为什么需要更多样本？
  1. 统计意义: 更多样本 → 更可靠的性能评估
  2. HTML报告: 默认生成前10个样本的详细报告
  3. 分布分析: 需要足够样本来分析状态分布
  4. 可视化: 更多数据点 → 更有意义的可视化

建议:
  python scripts/run_inference_uci.py \\
    --model-dir outputs/training/26-01-16/models \\
    --test-data data/uci/splits/test.csv \\
    --n-samples 50   # 或更多 ✅
""")
    
    print("\n" + "="*70 + "\n")
