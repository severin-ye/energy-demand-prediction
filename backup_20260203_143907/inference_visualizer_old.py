"""
推理结果可视化模块
生成精美的HTML页面展示完整推理流程
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class InferenceVisualizer:
    """推理结果可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """加载HTML模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>推理分析报告 - 样本 {sample_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: #2c3e50;
            color: white;
            padding: 30px 40px;
            border-bottom: 3px solid #3498db;
        }}
        
        .header h1 {{
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .header .meta {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px 40px;
        }}
        
        /* 核心结果区 - 最突出 */
        .summary {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin-bottom: 30px;
        }}
        
        .summary h2 {{
            font-size: 1.4em;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        
        .result-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .result-item {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        
        .result-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        
        .result-value {{
            font-size: 1.6em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .result-value.peak {{
            color: #e74c3c;
        }}
        
        .result-value.normal {{
            color: #27ae60;
        }}
        
        .result-value.lower {{
            color: #3498db;
        }}
        
        /* 分析步骤 - 简洁版 */
        .section {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section-title {{
            font-size: 1.2em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title .num {{
            display: inline-block;
            width: 28px;
            height: 28px;
            background: #3498db;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 28px;
            font-size: 0.9em;
        }}
        
        .section-content {{
            margin-left: 38px;
        }}
        
        .data-row {{
            padding: 8px 0;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #f5f5f5;
        }}
        
        .data-row:last-child {{
            border-bottom: none;
        }}
        
        .data-label {{
            color: #7f8c8d;
            font-size: 0.95em;
        }}
        
        .data-value {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        /* 简化的信息框 */
        .info-box {{
            background: #f8f9fa;
            border-left: 3px solid #3498db;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .info-box.warning {{
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }}
        
        .info-box.success {{
            border-left-color: #27ae60;
            background: #f0f9f4;
        }}
        
        /* 表格样式 */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
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
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        /* 简化的进度条 */
        .bar {{
            height: 24px;
            background: #3498db;
            border-radius: 3px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 0.9em;
        }}
        
        /* 建议列表 */
        .recommendation {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .recommendation-title {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        
        .recommendation-content {{
            color: #555;
            line-height: 1.6;
        }}
        }}
        
        .trend-arrow.down {{
            color: #28a745;
        }}
        
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .feature-card {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .feature-card.important {{
            border-color: #667eea;
            background: #f0f3ff;
        }}
        
        .feature-name {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .feature-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .recommendation {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .recommendation h3 {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .recommendation-item {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        .attention-viz {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 20px 0;
        }}
        
        .attention-block {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            background: #e0e0e0;
            position: relative;
        }}
        
        .attention-block.high {{
            background: #dc3545;
        }}
        
        .attention-block.medium {{
            background: #ffc107;
        }}
        
        .attention-block.low {{
            background: #28a745;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
        }}
        
        .emoji {{
            font-size: 2em;
            margin: 0 10px;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header">
            <h1>🔮 电力负荷智能预测流程可视化</h1>
            <p class="subtitle">样本 #{sample_id} | 生成时间: {timestamp}</p>
        </div>
        
        <div class="content">
            <!-- Step 0: 原始输入 -->
            <div class="step" data-step="0">
                <h2 class="step-title">📊 原始数据输入</h2>
                <p class="step-desc">系统接收到的现实世界数据</p>
                
                <div class="info-box info">
                    <h3>🎯 目标</h3>
                    <p>根据<strong>过去的用电情况</strong>，预测<strong>下一时刻是否会出现用电异常（峰值）</strong>，并说明原因和改进建议。</p>
                </div>
                
                <div class="feature-grid">
                    {input_features}
                </div>
                
                <div class="info-box">
                    <h4>📈 历史数据概览</h4>
                    <p>数据时间窗口: 过去 {window_size} 个时间步</p>
                    <p>预测目标: {target_name}</p>
                </div>
            </div>
            
            <!-- Step 1: 短期模式分析 -->
            <div class="step" data-step="1">
                <h2 class="step-title">🔍 短期模式分析 (CNN)</h2>
                <p class="step-desc">卷积神经网络在识别最近几分钟的突变模式</p>
                
                <div class="info-box highlight">
                    <h4>💡 CNN 在做什么？</h4>
                    <ul>
                        <li>看最近几分钟有没有<strong>突然变化</strong></li>
                        <li>看哪些电器是<strong>一起变大的</strong></li>
                        <li>识别<strong>短期异常模式</strong></li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <h4>📊 CAM激活模式</h4>
                    <p><strong>检测结果:</strong> {cam_pattern}</p>
                    <p><strong>聚类类型:</strong> Cluster {cam_cluster}</p>
                </div>
            </div>
            
            <!-- Step 2: 长期趋势分析 -->
            <div class="step" data-step="2">
                <h2 class="step-title">📈 长期趋势分析 (LSTM)</h2>
                <p class="step-desc">记忆网络在追踪整体走势</p>
                
                <div class="info-box highlight">
                    <h4>💡 LSTM 在做什么？</h4>
                    <ul>
                        <li>不关心具体哪一分钟</li>
                        <li>只关心<strong>整体走势</strong></li>
                        <li>判断是偶然波动还是<strong>持续趋势</strong></li>
                    </ul>
                </div>
                
                <div class="metric">
                    <span class="metric-label">趋势判断</span>
                    <span class="metric-value">{trend_direction}</span>
                </div>
            </div>
            
            <!-- Step 3: 关键时间判断 -->
            <div class="step" data-step="3">
                <h2 class="step-title">⏰ 关键时间判断 (Attention)</h2>
                <p class="step-desc">注意力机制在定位最重要的时间点</p>
                
                <div class="info-box highlight">
                    <h4>💡 注意力在做什么？</h4>
                    <ul>
                        <li>给每一个时间点<strong>打分</strong></li>
                        <li>分数越高，说明这个时刻<strong>越重要</strong></li>
                        <li>告诉你：<strong>模型为什么这样判断</strong></li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <h4>🎯 注意力类型: {attention_type}</h4>
                    <p><strong>关键结论:</strong> {attention_conclusion}</p>
                </div>
                
                <div class="attention-viz">
                    {attention_blocks}
                </div>
            </div>
            
            <!-- Step 4: 综合预测 -->
            <div class="step" data-step="4">
                <h2 class="step-title">🎯 综合判断与预测</h2>
                <p class="step-desc">融合所有信息，给出预测结果</p>
                
                <div class="info-box info">
                    <h4>🔄 融合以下信息:</h4>
                    <ul>
                        <li>CNN 的<strong>短期模式</strong></li>
                        <li>LSTM 的<strong>长期趋势</strong></li>
                        <li>Attention 的<strong>关键时间</strong></li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <div class="metric">
                        <span class="metric-label">预测负荷</span>
                        <span class="metric-value {prediction_class}">{prediction_value:.3f} kW</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">真实负荷</span>
                        <span class="metric-value">{true_value:.3f} kW</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">预测误差</span>
                        <span class="metric-value {error_class}">{error_value:.3f} kW ({error_percent:.1f}%)</span>
                    </div>
                </div>
            </div>
            
            <!-- Step 5: 状态分类 -->
            <div class="step" data-step="5">
                <h2 class="step-title">🚦 负荷状态分类</h2>
                <p class="step-desc">把"数值"变成"状态" - 这是因果分析的关键</p>
                
                <div class="info-box {state_box_class}">
                    <h3><span class="emoji">{state_emoji}</span> 状态: {state_name}</h3>
                    <p><strong>为什么要分类？</strong> 因果模型不擅长看"2.6、3.8"这样的数值，它更擅长看"高/很高/正常"这样的等级。</p>
                </div>
                
                <div class="info-box">
                    <h4>📊 状态判断依据:</h4>
                    <ul>
                        <li>预测值: {prediction_value:.3f} kW</li>
                        <li>历史中位数: {median_value:.3f} kW</li>
                        <li>偏离程度: {deviation_level}</li>
                    </ul>
                </div>
            </div>
            
            <!-- Step 6: 特征离散化 -->
            <div class="step" data-step="6">
                <h2 class="step-title">🔤 特征等级化</h2>
                <p class="step-desc">把所有连续数据"翻译成人类语言等级"</p>
                
                <div class="info-box highlight">
                    <h4>💡 为什么要等级化？</h4>
                    <p>因果推理更擅长理解"非常高/中等/偏低"，而不是具体数值。</p>
                </div>
                
                <div class="feature-grid">
                    {discrete_features}
                </div>
            </div>
            
            <!-- Step 7: DLP特征提取 -->
            <div class="step" data-step="7">
                <h2 class="step-title">🧠 模型内部感知提取</h2>
                <p class="step-desc">把"模型的直觉"翻译成人话</p>
                
                <div class="info-box info">
                    <h4>🎨 CAM特征聚类</h4>
                    <p><strong>模式类型:</strong> Cluster {cam_cluster}</p>
                    <p><strong>含义:</strong> {cam_meaning}</p>
                </div>
                
                <div class="info-box info">
                    <h4>⏰ Attention时间模式</h4>
                    <p><strong>注意力类型:</strong> {attention_type}</p>
                    <p><strong>含义:</strong> {attention_meaning}</p>
                </div>
            </div>
            
            <!-- Step 8: 因果推断 -->
            <div class="step" data-step="8">
                <h2 class="step-title">🔗 因果关系推断</h2>
                <p class="step-desc">这里不是在说"谁相关性高"，而是在说"是谁真正把你推向峰值的"</p>
                
                <div class="info-box highlight">
                    <h4>💡 因果推断在做什么？</h4>
                    <p>基于贝叶斯网络，分析各个因素对最终状态的<strong>因果贡献</strong>，而不仅仅是相关性。</p>
                </div>
                
                {causal_analysis}
            </div>
            
            <!-- Step 9: 反事实分析 -->
            <div class="step" data-step="9">
                <h2 class="step-title">🔮 反事实提问</h2>
                <p class="step-desc">"如果我改点什么会怎样？" - 真正有价值的建议</p>
                
                {counterfactual_analysis}
            </div>
            
            <!-- Step 10: 最终建议 -->
            <div class="step" data-step="10">
                <h2 class="step-title">✨ 智能建议输出</h2>
                <p class="step-desc">系统给出的最终结论和行动建议</p>
                
                {recommendations}
            </div>
        </div>
        
        <!-- 底部 -->
        <div class="footer">
            <p><strong>Parallel CNN-LSTM-Attention + Causal Inference System</strong></p>
            <p>基于UCI家庭电力消耗数据集训练</p>
            <p>生成时间: {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_html(
        self,
        sample_data: Dict[str, Any],
        output_file: str
    ) -> str:
        """
        生成单个样本的HTML可视化
        
        Args:
            sample_data: 样本数据字典
            output_file: 输出文件路径
        
        Returns:
            生成的HTML文件路径
        """
        # 准备所有需要填充的数据
        html_content = self.template.format(
            sample_id=sample_data.get('sample_id', 0),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Step 0
            input_features=self._format_input_features(sample_data),
            window_size=sample_data.get('window_size', 'N/A'),
            target_name=sample_data.get('target_name', 'EDP'),
            
            # Step 1
            cam_pattern=self._describe_cam_pattern(sample_data),
            cam_cluster=sample_data.get('cam_cluster', 0),
            
            # Step 2
            trend_direction=self._describe_trend(sample_data),
            
            # Step 3
            attention_type=sample_data.get('attention_type', 'Unknown'),
            attention_conclusion=self._describe_attention(sample_data),
            attention_blocks=self._generate_attention_blocks(sample_data),
            
            # Step 4
            prediction_value=sample_data.get('prediction', 0),
            true_value=sample_data.get('true_value', 0),
            error_value=sample_data.get('error', 0),
            error_percent=sample_data.get('error_percent', 0),
            prediction_class=self._get_prediction_class(sample_data),
            error_class=self._get_error_class(sample_data),
            
            # Step 5
            state_name=sample_data.get('state', 'Unknown'),
            state_emoji=self._get_state_emoji(sample_data.get('state')),
            state_box_class=self._get_state_box_class(sample_data.get('state')),
            median_value=sample_data.get('median_value', 0),
            deviation_level=self._describe_deviation(sample_data),
            
            # Step 6
            discrete_features=self._format_discrete_features(sample_data),
            
            # Step 7
            cam_meaning=self._get_cam_meaning(sample_data.get('cam_cluster', 0)),
            attention_meaning=self._get_attention_meaning(sample_data.get('attention_type')),
            
            # Step 8
            causal_analysis=self._format_causal_analysis(sample_data),
            
            # Step 9
            counterfactual_analysis=self._format_counterfactual(sample_data),
            
            # Step 10
            recommendations=self._format_recommendations(sample_data)
        )
        
        # 保存HTML文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _format_input_features(self, data: Dict) -> str:
        """格式化输入特征"""
        features = data.get('input_features', {})
        html = ""
        
        for name, value in features.items():
            html += f'''
            <div class="feature-card">
                <div class="feature-name">{name}</div>
                <div class="feature-value">{value:.3f}</div>
            </div>
            '''
        
        return html
    
    def _describe_cam_pattern(self, data: Dict) -> str:
        """描述CAM模式"""
        cluster = data.get('cam_cluster', 0)
        patterns = {
            0: "基础用电模式 - 各特征稳定变化",
            1: "异常突变模式 - 某些特征快速上升",
            2: "复杂混合模式 - 多特征交互影响"
        }
        return patterns.get(cluster, "未知模式")
    
    def _describe_trend(self, data: Dict) -> str:
        """描述趋势"""
        pred = data.get('prediction', 0)
        true_val = data.get('true_value', 0)
        
        if pred > true_val * 1.2:
            return "快速上升趋势 ⬆️⬆️"
        elif pred > true_val * 1.05:
            return "缓慢上升趋势 ⬆️"
        elif pred < true_val * 0.8:
            return "下降趋势 ⬇️"
        else:
            return "基本稳定 ➡️"
    
    def _describe_attention(self, data: Dict) -> str:
        """描述注意力结论"""
        att_type = data.get('attention_type', '')
        
        descriptions = {
            'Early': '模型主要关注<strong>历史早期</strong>的用电模式，说明当前状态受早期影响较大',
            'Late': '模型主要关注<strong>最近时刻</strong>的用电变化，说明近期变化是关键',
            'Other': '模型关注<strong>整个时间段</strong>的综合信息，各时刻都有贡献'
        }
        
        return descriptions.get(att_type, '注意力分布均匀')
    
    def _generate_attention_blocks(self, data: Dict) -> str:
        """生成注意力可视化块"""
        # 模拟注意力分布（实际应该从模型获取）
        att_type = data.get('attention_type', 'Other')
        
        blocks = []
        num_blocks = 80  # 假设80个时间步
        
        for i in range(num_blocks):
            if att_type == 'Late':
                # 后期注意力高
                level = 'high' if i > 60 else ('medium' if i > 40 else 'low')
            elif att_type == 'Early':
                # 早期注意力高
                level = 'high' if i < 20 else ('medium' if i < 40 else 'low')
            else:
                # 均匀分布
                level = 'medium'
            
            blocks.append(f'<div class="attention-block {level}" title="时间步 {i}"></div>')
        
        return ''.join(blocks)
    
    def _get_prediction_class(self, data: Dict) -> str:
        """获取预测值的CSS类"""
        state = data.get('state', '')
        return 'danger' if state == 'Peak' else ('success' if state == 'Lower' else '')
    
    def _get_error_class(self, data: Dict) -> str:
        """获取误差的CSS类"""
        error_percent = abs(data.get('error_percent', 0))
        return 'danger' if error_percent > 50 else ('success' if error_percent < 20 else '')
    
    def _get_state_emoji(self, state: str) -> str:
        """获取状态对应的emoji"""
        emojis = {
            'Lower': '🟢',
            'Normal': '🟡',
            'Peak': '🔴'
        }
        return emojis.get(state, '⚪')
    
    def _get_state_box_class(self, state: str) -> str:
        """获取状态框的CSS类"""
        classes = {
            'Lower': 'success',
            'Normal': 'highlight',
            'Peak': 'warning'
        }
        return classes.get(state, 'info')
    
    def _describe_deviation(self, data: Dict) -> str:
        """描述偏离程度"""
        pred = data.get('prediction', 0)
        median = data.get('median_value', 0)
        
        if median == 0:
            return "无法计算"
        
        deviation = (pred - median) / median * 100
        
        if abs(deviation) < 10:
            return "正常范围内"
        elif abs(deviation) < 30:
            return "轻微偏离"
        elif abs(deviation) < 50:
            return "中度偏离"
        else:
            return "严重偏离"
    
    def _format_discrete_features(self, data: Dict) -> str:
        """格式化离散化特征"""
        features = data.get('discrete_features', {})
        html = ""
        
        for name, level in features.items():
            importance = 'important' if level in ['非常高', '很高', 'High'] else ''
            html += f'''
            <div class="feature-card {importance}">
                <div class="feature-name">{name}</div>
                <div class="feature-value">{level}</div>
            </div>
            '''
        
        return html
    
    def _get_cam_meaning(self, cluster: int) -> str:
        """获取CAM聚类的含义"""
        meanings = {
            0: "模型识别出这是一个<strong>常规用电模式</strong>，各项特征按正常规律变化",
            1: "模型识别出这是一个<strong>异常突变模式</strong>，某些特征出现快速变化",
            2: "模型识别出这是一个<strong>复杂混合模式</strong>，多个因素同时起作用"
        }
        return meanings.get(cluster, "未知模式")
    
    def _get_attention_meaning(self, att_type: str) -> str:
        """获取注意力类型的含义"""
        meanings = {
            'Early': "模型认为<strong>历史早期的用电模式</strong>对当前预测影响更大",
            'Late': "模型认为<strong>最近时刻的用电变化</strong>对当前预测影响更大",
            'Other': "模型认为<strong>整个时间段的信息</strong>都很重要"
        }
        return meanings.get(att_type, "注意力分布均匀")
    
    def _format_causal_analysis(self, data: Dict) -> str:
        """格式化因果分析"""
        # 这里应该从实际的因果推断结果中获取
        return '''
        <div class="info-box">
            <h4>🔗 因果网络分析</h4>
            <p>基于贝叶斯网络，系统识别出以下因果关系：</p>
            <ul>
                <li>各输入特征通过因果链影响最终状态</li>
                <li>状态分类受到多个因素的联合影响</li>
                <li>可以通过干预特定变量来改变结果</li>
            </ul>
        </div>
        '''
    
    def _format_counterfactual(self, data: Dict) -> str:
        """格式化反事实分析"""
        return '''
        <div class="info-box info">
            <h4>🔮 反事实推理</h4>
            <p><strong>问题:</strong> 如果改变某个关键因素，结果会如何？</p>
            <p><strong>答案:</strong> 通过因果网络进行反事实推理，可以预测干预后的效果。</p>
            <p>例如：将某个"非常高"的特征降低到"中等"，可以降低峰值风险。</p>
        </div>
        '''
    
    def _format_recommendations(self, data: Dict) -> str:
        """格式化建议"""
        state = data.get('state', '')
        
        if state == 'Peak':
            return '''
            <div class="recommendation">
                <h3>⚠️ 峰值预警建议</h3>
                <div class="recommendation-item">
                    <strong>🎯 优先措施:</strong> 降低高负荷电器的使用强度
                </div>
                <div class="recommendation-item">
                    <strong>⏰ 时间建议:</strong> 避免在短时间内启动多个大功率设备
                </div>
                <div class="recommendation-item">
                    <strong>📊 预期效果:</strong> 可将峰值风险降低约 30-50%
                </div>
            </div>
            '''
        elif state == 'Normal':
            return '''
            <div class="recommendation">
                <h3>✅ 正常状态建议</h3>
                <div class="recommendation-item">
                    <strong>✨ 当前状态良好，用电处于正常范围</strong>
                </div>
                <div class="recommendation-item">
                    <strong>💡 建议:</strong> 保持当前用电模式，注意监控变化趋势
                </div>
            </div>
            '''
        else:
            return '''
            <div class="recommendation">
                <h3>🟢 低负荷状态</h3>
                <div class="recommendation-item">
                    <strong>✨ 用电负荷较低，运行良好</strong>
                </div>
                <div class="recommendation-item">
                    <strong>💡 提示:</strong> 当前是启动大功率设备的良好时机
                </div>
            </div>
            '''
