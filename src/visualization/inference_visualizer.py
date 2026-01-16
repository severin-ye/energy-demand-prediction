"""
推理结果可视化模块 - 简洁版
专注于展示核心信息，清晰易读
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class InferenceVisualizer:
    """推理结果可视化器 - 简洁版"""
    
    def __init__(self):
        """初始化可视化器"""
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """加载简洁HTML模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析报告 #{sample_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .header {{
            background: #2c3e50;
            color: white;
            padding: 25px 30px;
            border-bottom: 3px solid #3498db;
        }}
        
        .header h1 {{
            font-size: 1.6em;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        
        .header .meta {{
            font-size: 0.9em;
            opacity: 0.85;
        }}
        
        .content {{
            padding: 25px 30px;
        }}
        
        /* 核心结果区域 */
        .summary {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin-bottom: 25px;
        }}
        
        .summary h2 {{
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        
        .result-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }}
        
        .result-item {{
            background: white;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        
        .result-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 4px;
        }}
        
        .result-value {{
            font-size: 1.5em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .result-value.peak {{ color: #e74c3c; }}
        .result-value.normal {{ color: #27ae60; }}
        .result-value.lower {{ color: #3498db; }}
        
        /* 分析区域 */
        .section {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .section:last-child {{ border-bottom: none; }}
        
        .section-title {{
            font-size: 1.15em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title .num {{
            width: 26px;
            height: 26px;
            background: #3498db;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 26px;
            font-size: 0.85em;
        }}
        
        .section-content {{
            margin-left: 36px;
        }}
        
        /* 数据表格 */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 0.95em;
        }}
        
        th {{
            background: #ecf0f1;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #bdc3c7;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{ background: #f8f9fa; }}
        
        /* 数据行 */
        .data-row {{
            padding: 8px 0;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #f5f5f5;
        }}
        
        .data-row:last-child {{ border-bottom: none; }}
        
        .data-label {{
            color: #7f8c8d;
            font-size: 0.95em;
        }}
        
        .data-value {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        /* 信息框 */
        .info-box {{
            background: #f8f9fa;
            border-left: 3px solid #3498db;
            padding: 12px;
            margin: 12px 0;
            font-size: 0.95em;
        }}
        
        .info-box.warning {{
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }}
        
        .info-box.success {{
            border-left-color: #27ae60;
            background: #f0f9f4;
        }}
        
        /* 建议卡片 */
        .recommendation {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12px;
            margin: 10px 0;
        }}
        
        .rec-title {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 6px;
        }}
        
        .rec-content {{
            color: #555;
            line-height: 1.5;
            font-size: 0.95em;
        }}
        
        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>电力负荷预测分析报告</h1>
            <p class="meta">样本 #{sample_id} · {timestamp}</p>
        </div>
        
        <div class="content">
            <!-- 核心结果 -->
            <div class="summary">
                <h2>预测结果</h2>
                <div class="result-grid">
                    <div class="result-item">
                        <div class="result-label">预测负荷</div>
                        <div class="result-value">{prediction} kW</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">实际负荷</div>
                        <div class="result-value">{actual} kW</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">预测误差</div>
                        <div class="result-value">{error}%</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">负荷状态</div>
                        <div class="result-value {state_class}">{state}</div>
                    </div>
                </div>
            </div>
            
            <!-- 1. 输入数据 -->
            <div class="section">
                <h3 class="section-title">
                    <span class="num">1</span>
                    输入数据
                </h3>
                <div class="section-content">
                    <table>
                        <tr>
                            <th>特征</th>
                            <th>当前值</th>
                            <th>离散化等级</th>
                        </tr>
                        {input_table}
                    </table>
                </div>
            </div>
            
            <!-- 2. 模型分析 -->
            <div class="section">
                <h3 class="section-title">
                    <span class="num">2</span>
                    模型分析
                </h3>
                <div class="section-content">
                    <div class="data-row">
                        <span class="data-label">CAM 聚类</span>
                        <span class="data-value">Cluster {cam_cluster}</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">注意力模式</span>
                        <span class="data-value">{attention_type}</span>
                    </div>
                    <div class="info-box">
                        {pattern_explanation}
                    </div>
                </div>
            </div>
            
            <!-- 3. 因果分析 -->
            <div class="section">
                <h3 class="section-title">
                    <span class="num">3</span>
                    因果关系
                </h3>
                <div class="section-content">
                    {causal_analysis}
                </div>
            </div>
            
            <!-- 4. 优化建议 -->
            <div class="section">
                <h3 class="section-title">
                    <span class="num">4</span>
                    优化建议
                </h3>
                <div class="section-content">
                    {recommendations}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_html(
        self,
        result: Dict[str, Any],
        sample_id: int,
        output_path: Path
    ) -> Path:
        """
        生成简洁的HTML报告
        
        参数:
            result: 推理结果字典
            sample_id: 样本ID
            output_path: 输出文件路径
        
        返回:
            Path: 生成的HTML文件路径
        """
        # 提取核心数据
        prediction = result.get('prediction', 0)
        actual = result.get('actual_value', 0)
        error = ((prediction - actual) / actual * 100) if actual != 0 else 0
        state = result.get('state', 'Unknown')
        
        # 状态对应的CSS类
        state_class_map = {
            'Peak': 'peak',
            'Normal': 'normal',
            'Lower': 'lower'
        }
        state_class = state_class_map.get(state, '')
        
        # 生成输入数据表格
        input_features = result.get('input_features', {})
        discrete_features = result.get('discrete_features', {})
        
        input_rows = []
        for key, value in input_features.items():
            discrete_value = discrete_features.get(key, '-')
            input_rows.append(f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value:.3f}</td>
                            <td>{discrete_value}</td>
                        </tr>""")
        input_table = ''.join(input_rows)
        
        # 模式解释
        cam_cluster = result.get('cam_cluster', 0)
        attention_type = result.get('attention_type', 'Unknown')
        pattern_explanation = f"模型检测到 {attention_type} 注意力模式，CAM聚类为第 {cam_cluster} 类"
        
        # 因果分析
        causal_text = result.get('causal_explanation', '暂无因果分析数据')
        causal_analysis = f'<div class="info-box">{causal_text}</div>'
        
        # 生成建议列表
        recommendations_list = result.get('recommendations', [])
        if recommendations_list:
            recs_html = []
            for i, rec in enumerate(recommendations_list[:5], 1):  # 最多显示5条
                title = rec.get('action', f'建议 {i}')
                content = rec.get('explanation', '')
                impact = rec.get('expected_impact', '')
                
                full_content = content
                if impact:
                    full_content += f"<br><small>预期效果: {impact}</small>"
                
                recs_html.append(f"""
                    <div class="recommendation">
                        <div class="rec-title">{i}. {title}</div>
                        <div class="rec-content">{full_content}</div>
                    </div>""")
            recommendations = ''.join(recs_html)
        else:
            recommendations = '<div class="info-box">当前状态良好，暂无优化建议</div>'
        
        # 填充模板
        html_content = self.template.format(
            sample_id=sample_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            prediction=f"{prediction:.3f}",
            actual=f"{actual:.3f}",
            error=f"{error:+.1f}",
            state=state,
            state_class=state_class,
            input_table=input_table,
            cam_cluster=cam_cluster,
            attention_type=attention_type,
            pattern_explanation=pattern_explanation,
            causal_analysis=causal_analysis,
            recommendations=recommendations
        )
        
        # 写入文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')
        
        return output_path
    
    def generate_index_html(
        self,
        n_samples: int,
        output_dir: Path
    ) -> Path:
        """生成简洁的索引页面"""
        index_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析报告索引</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }}
        .sample-link {{
            display: block;
            background: #3498db;
            color: white;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
            text-decoration: none;
            font-weight: 600;
            transition: background 0.2s;
        }}
        .sample-link:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>电力负荷预测分析报告</h1>
        <p>共 {n_samples} 个样本</p>
        <div class="grid">
            {links}
        </div>
    </div>
</body>
</html>
"""
        links = []
        for i in range(n_samples):
            links.append(f'<a href="sample_{i:03d}.html" class="sample-link">样本 #{i}</a>')
        
        html = index_template.format(
            n_samples=n_samples,
            links='\n            '.join(links)
        )
        
        index_path = output_dir / 'index.html'
        index_path.write_text(html, encoding='utf-8')
        
        return index_path
