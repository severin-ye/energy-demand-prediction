"""
UCI数据推理测试
使用训练好的模型对测试集进行预测，并生成易读的结果报告
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path

from src.pipeline.inference_pipeline import InferencePipeline
from src.visualization.inference_visualizer import InferenceVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_file, n_samples=100):
    """
    加载测试集数据
    
    Args:
        test_file: 测试集文件路径
        n_samples: 使用多少样本进行测试
    """
    logger.info(f"加载测试数据: {test_file}")
    df = pd.read_csv(test_file)
    
    # 只取前n_samples个样本以加快推理速度
    if len(df) > n_samples:
        df = df.iloc[:n_samples].copy()
        logger.info(f"使用前{n_samples}个样本进行推理")
    
    # 准备特征
    feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity']
    target_col = 'Global_active_power'
    
    # 重命名目标列为EDP（兼容训练时的命名）
    df_test = df[feature_cols + [target_col]].copy()
    df_test = df_test.rename(columns={target_col: 'EDP'})
    
    logger.info(f"测试数据形状: {df_test.shape}")
    logger.info(f"EDP范围: [{df_test['EDP'].min():.2f}, {df_test['EDP'].max():.2f}]")
    
    return df_test


def format_prediction_results(results):
    """
    将预测结果格式化为易读的报告
    
    Args:
        results: 推理结果字典
    """
    report = []
    report.append("=" * 80)
    report.append(" " * 25 + "📊 推理结果报告")
    report.append("=" * 80)
    
    # 1. 基本信息
    report.append("\n【1. 基本信息】")
    report.append(f"  测试样本数: {len(results['predictions'])} 个")
    report.append(f"  预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. 预测统计
    predictions = results['predictions']
    true_values = results.get('true_values', [])
    
    report.append("\n【2. 预测统计】")
    report.append(f"  预测EDP范围: [{predictions.min():.3f}, {predictions.max():.3f}] kW")
    report.append(f"  预测EDP均值: {predictions.mean():.3f} kW")
    report.append(f"  预测EDP标准差: {predictions.std():.3f} kW")
    
    if len(true_values) > 0:
        # 确保 true_values 与 predictions 长度匹配
        true_values = np.array(true_values[:len(predictions)])
        report.append(f"\n  真实EDP范围: [{true_values.min():.3f}, {true_values.max():.3f}] kW")
        report.append(f"  真实EDP均值: {true_values.mean():.3f} kW")
        
        # 计算误差指标
        errors = predictions - true_values
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = np.mean(np.abs(errors / (true_values + 1e-8))) * 100
        
        report.append(f"\n  【性能指标】")
        report.append(f"    MAE (平均绝对误差): {mae:.4f} kW")
        report.append(f"    RMSE (均方根误差): {rmse:.4f} kW")
        report.append(f"    MAPE (平均绝对百分比误差): {mape:.2f}%")
    
    # 3. 状态分布
    if 'edp_states' in results:
        states = results['edp_states']
        state_counts = pd.Series(states).value_counts()
        
        report.append("\n【3. EDP状态分布】")
        total = len(states)
        for state, count in state_counts.items():
            percentage = count / total * 100
            bar = "█" * int(percentage / 2)
            report.append(f"  {state:8s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # 4. DLP特征分布
    report.append("\n【4. 深度学习参数(DLP)特征】")
    
    if 'cam_clusters' in results:
        cam_clusters = results['cam_clusters']
        cam_counts = pd.Series(cam_clusters).value_counts().sort_index()
        
        report.append("  CAM聚类分布:")
        for cluster, count in cam_counts.items():
            percentage = count / len(cam_clusters) * 100
            report.append(f"    Cluster {cluster}: {count:4d} ({percentage:5.1f}%)")
    
    if 'attention_types' in results:
        attention_types = results['attention_types']
        attention_counts = pd.Series(attention_types).value_counts()
        
        report.append("\n  Attention类型分布:")
        for att_type, count in attention_counts.items():
            percentage = count / len(attention_types) * 100
            report.append(f"    {att_type:10s}: {count:4d} ({percentage:5.1f}%)")
    
    # 5. 样本案例展示
    report.append("\n【5. 典型样本案例】")
    
    # 选择3个代表性样本
    n_samples = len(predictions)
    indices = [0, n_samples // 2, n_samples - 1]
    
    for i, idx in enumerate(indices):
        if idx >= n_samples:
            continue
            
        report.append(f"\n  样本 {i+1} (索引 {idx}):")
        report.append(f"    预测EDP: {predictions[idx]:.3f} kW")
        
        if len(true_values) > 0 and idx < len(true_values):
            error = predictions[idx] - true_values[idx]
            report.append(f"    真实EDP: {true_values[idx]:.3f} kW")
            report.append(f"    误差: {error:+.3f} kW ({error/true_values[idx]*100:+.1f}%)")
        
        if 'edp_states' in results and idx < len(results['edp_states']):
            report.append(f"    状态: {results['edp_states'][idx]}")
        
        if 'cam_clusters' in results and idx < len(results['cam_clusters']):
            report.append(f"    CAM聚类: {results['cam_clusters'][idx]}")
        
        if 'attention_types' in results and idx < len(results['attention_types']):
            report.append(f"    Attention: {results['attention_types'][idx]}")
    
    # 6. 预测区间统计
    report.append("\n【6. 预测区间统计】")
    
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '>3.0']
    
    pred_binned = pd.cut(predictions, bins=bins, labels=labels, include_lowest=True)
    bin_counts = pred_binned.value_counts().sort_index()
    
    report.append("  预测EDP分布 (kW):")
    for label, count in bin_counts.items():
        percentage = count / len(predictions) * 100
        bar = "▓" * int(percentage / 2)
        report.append(f"    {label:10s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def save_detailed_results(results, output_file):
    """
    保存详细结果到CSV文件
    
    Args:
        results: 推理结果字典
        output_file: 输出文件路径
    """
    # 构建详细结果DataFrame
    data = {
        'index': range(len(results['predictions'])),
        'predicted_edp': results['predictions'],
    }
    
    if 'true_values' in results and len(results['true_values']) > 0:
        # 确保长度匹配
        true_vals = np.array(results['true_values'][:len(results['predictions'])])
        data['true_edp'] = true_vals
        data['error'] = results['predictions'] - true_vals
        data['abs_error'] = np.abs(data['error'])
        data['relative_error'] = data['error'] / (true_vals + 1e-8)
    
    if 'edp_states' in results:
        data['state'] = results['edp_states']
    
    if 'cam_clusters' in results:
        data['cam_cluster'] = results['cam_clusters']
    
    if 'attention_types' in results:
        data['attention_type'] = results['attention_types']
    
    df_results = pd.DataFrame(data)
    
    # 保存到CSV
    df_results.to_csv(output_file, index=False)
    logger.info(f"详细结果已保存到: {output_file}")
    
    return df_results


def generate_html_reports(results: dict, test_data: pd.DataFrame, output_dir: Path):
    """
    为每个样本生成HTML可视化报告
    
    Args:
        results: 推理结果字典
        test_data: 测试数据
        output_dir: 输出目录
    """
    logger.info("\n生成HTML可视化报告...")
    
    visualizer = InferenceVisualizer()
    html_dir = output_dir / 'html_reports'
    html_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = results['predictions']
    true_values = results.get('true_values', [])
    edp_states = results.get('edp_states', [])
    cam_clusters = results.get('cam_clusters', [])
    attention_types = results.get('attention_types', [])
    
    # 计算中位数用于偏离判断
    median_value = np.median(predictions)
    
    # 为每个样本生成HTML
    num_samples = min(len(predictions), 10)  # 只生成前10个样本的HTML
    logger.info(f"将为前{num_samples}个样本生成HTML报告")
    
    for idx in range(num_samples):
        # 准备样本数据
        sample_data = {
            'sample_id': idx,
            'window_size': 'N/A',  # 实际应该从数据获取
            'target_name': 'Global Active Power (EDP)',
            
            # 输入特征（取当前时刻的值）
            'input_features': {
                'Global Reactive Power': test_data.iloc[idx]['Global_reactive_power'],
                'Voltage': test_data.iloc[idx]['Voltage'],
                'Global Intensity': test_data.iloc[idx]['Global_intensity'],
            },
            
            # CAM和Attention
            'cam_cluster': int(cam_clusters[idx]) if idx < len(cam_clusters) else 0,
            'attention_type': attention_types[idx] if idx < len(attention_types) else 'Unknown',
            
            # 预测结果
            'prediction': float(predictions[idx]),
            'true_value': float(true_values[idx]) if idx < len(true_values) else 0,
            'error': float(predictions[idx] - true_values[idx]) if idx < len(true_values) else 0,
            'error_percent': float((predictions[idx] - true_values[idx]) / (true_values[idx] + 1e-8) * 100) if idx < len(true_values) else 0,
            
            # 状态
            'state': edp_states[idx] if idx < len(edp_states) else 'Unknown',
            'median_value': float(median_value),
            
            # 离散化特征（示例）
            'discrete_features': {
                'Global Reactive Power': _discretize_value(test_data.iloc[idx]['Global_reactive_power'], 'reactive'),
                'Voltage': _discretize_value(test_data.iloc[idx]['Voltage'], 'voltage'),
                'Global Intensity': _discretize_value(test_data.iloc[idx]['Global_intensity'], 'intensity'),
            }
        }
        
        # 生成HTML
        html_file = html_dir / f'sample_{idx:03d}.html'
        visualizer.generate_html(sample_data, html_file)
        
        if idx == 0:
            logger.info(f"✅ 示例报告: {html_file}")
    
    logger.info(f"✅ 已生成 {num_samples} 个HTML报告到: {html_dir}")
    
    # 生成索引页面
    _generate_index_page(html_dir, num_samples)
    logger.info(f"✅ 索引页面: {html_dir}/index.html")


def _discretize_value(value: float, feature_type: str) -> str:
    """离散化数值"""
    if feature_type == 'reactive':
        if value < 0.05:
            return '很低'
        elif value < 0.15:
            return '中等'
        else:
            return '偏高'
    elif feature_type == 'voltage':
        if value < 230:
            return '偏低'
        elif value < 245:
            return '正常'
        else:
            return '偏高'
    elif feature_type == 'intensity':
        if value < 5:
            return '低'
        elif value < 15:
            return '中等'
        else:
            return '高'
    return '未知'


def _generate_index_page(html_dir: Path, num_samples: int):
    """生成索引页面"""
    index_html = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>推理结果索引</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            margin: 0;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            text-decoration: none;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .card-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .card-subtitle {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 电力负荷智能预测可视化</h1>
        <p class="subtitle">UCI家庭电力消耗数据集 - 推理结果报告</p>
        
        <div class="grid">
'''
    
    for i in range(num_samples):
        index_html += f'''
            <a href="sample_{i:03d}.html" class="card">
                <div class="card-title">样本 #{i}</div>
                <div class="card-subtitle">查看详细推理流程</div>
            </a>
'''
    
    index_html += '''
        </div>
        
        <div class="footer">
            <p><strong>Parallel CNN-LSTM-Attention + Causal Inference</strong></p>
            <p>生成时间: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        </div>
    </div>
</body>
</html>
'''
    
    with open(html_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='UCI数据集推理测试')
    parser.add_argument(
        '--model-dir',
        default='outputs/training_uci/models',
        help='模型目录'
    )
    parser.add_argument(
        '--test-data',
        default='data/uci/splits/test.csv',
        help='测试数据文件'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='使用多少测试样本（默认100）'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/inference_uci',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("=" * 80)
        logger.info(" " * 30 + "推理测试开始")
        logger.info("=" * 80)
        
        # 1. 加载测试数据
        test_data = load_test_data(args.test_data, args.n_samples)
        true_values = test_data['EDP'].values
        
        # 2. 初始化推理流水线
        logger.info(f"\n加载模型: {args.model_dir}")
        pipeline = InferencePipeline(models_dir=args.model_dir)
        
        # 3. 执行推理
        logger.info(f"\n开始推理...")
        results = pipeline.predict(test_data)
        
        # 添加真实值到结果中
        results['true_values'] = true_values
        
        # 4. 生成易读报告
        logger.info(f"\n生成推理报告...")
        report = format_prediction_results(results)
        
        # 打印到控制台
        print("\n" + report)
        
        # 5. 保存报告到文件
        report_file = output_dir / 'inference_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"\n报告已保存到: {report_file}")
        
        # 6. 保存详细结果
        detail_file = output_dir / 'inference_details.csv'
        df_results = save_detailed_results(results, detail_file)
        
        # 7. 保存JSON格式结果
        json_file = output_dir / 'inference_results.json'
        # 确保长度匹配
        true_vals_matched = true_values[:len(results['predictions'])]
        json_results = {
            'predictions': results['predictions'].tolist(),
            'true_values': true_vals_matched.tolist() if isinstance(true_vals_matched, np.ndarray) else list(true_vals_matched),
            'edp_states': list(results.get('edp_states', [])),
            'cam_clusters': [int(x) for x in results.get('cam_clusters', [])],
            'attention_types': list(results.get('attention_types', [])),
            'statistics': {
                'mae': float(np.abs(results['predictions'] - true_vals_matched).mean()),
                'rmse': float(np.sqrt(((results['predictions'] - true_vals_matched) ** 2).mean())),
                'n_samples': len(results['predictions'])
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON结果已保存到: {json_file}")
        
        # 8. 生成HTML可视化报告
        logger.info("\n" + "=" * 80)
        logger.info("🎨 生成HTML可视化报告")
        logger.info("=" * 80)
        generate_html_reports(results, test_data, output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info(" " * 30 + "推理测试完成")
        logger.info("=" * 80)
        
        # 9. 返回性能摘要
        print("\n" + "=" * 80)
        print("📊 性能摘要")
        print("=" * 80)
        print(f"MAE:  {json_results['statistics']['mae']:.4f} kW")
        print(f"RMSE: {json_results['statistics']['rmse']:.4f} kW")
        print(f"样本数: {json_results['statistics']['n_samples']}")
        print("=" * 80)
        print(f"\n💡 查看HTML可视化: {output_dir}/html_reports/index.html")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
