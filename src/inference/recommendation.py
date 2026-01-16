"""
推荐生成模块
基于因果推断结果生成可操作的节能建议
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    推荐引擎
    
    功能:
    1. 基于因果推断识别关键影响因素
    2. 生成可操作的节能建议
    3. 量化每条建议的预期效果
    4. 支持多模板和上下文感知的自然语言生成
    
    参数:
    - causal_inference: 训练好的CausalInference对象
    - feature_mapping: 特征到可读名称的映射
    - action_templates: 行动建议模板
    """
    
    def __init__(
        self,
        causal_inference,
        feature_mapping: Optional[Dict[str, str]] = None,
        action_templates: Optional[Dict[str, Dict[str, str]]] = None
    ):
        self.ci = causal_inference
        self.feature_mapping = feature_mapping or self._default_mapping()
        self.action_templates = action_templates or self._default_templates()
        
        logger.info("RecommendationEngine initialized")
    
    def _default_mapping(self) -> Dict[str, str]:
        """默认特征映射（可读名称）"""
        return {
            'Temperature': '温度',
            'Humidity': '湿度',
            'WindSpeed': '风速',
            'Hour': '时段',
            'DayOfWeek': '星期',
            'Month': '月份',
            'Season': '季节',
            'IsWeekend': '周末标志',
            'EDP_State': '用电负荷'
        }
    
    def _default_templates(self) -> Dict[str, Dict[str, str]]:
        """默认行动建议模板"""
        return {
            'Temperature': {
                'Low': '建议降低室内温度设定，例如调低空调温度至{value}℃左右',
                'Medium': '建议保持适中温度，避免过度制冷或制热',
                'High': '建议提高室内温度设定，减少空调能耗',
                'VeryHigh': '建议大幅提高温度设定，或考虑关闭部分空调'
            },
            'Humidity': {
                'Low': '建议降低除湿设备使用，湿度控制在{value}%左右',
                'Medium': '建议保持当前湿度水平',
                'High': '建议提高除湿设备功率，但注意能耗平衡',
                'VeryHigh': '建议开启除湿模式，或使用自然通风'
            },
            'WindSpeed': {
                'Low': '建议降低通风设备功率',
                'Medium': '建议保持当前通风水平',
                'High': '建议利用自然风，减少机械通风',
                'VeryHigh': '建议充分利用自然通风，关闭风扇等设备'
            },
            'Hour': {
                'Peak': '当前为用电高峰时段，建议错峰使用高耗能设备',
                'Normal': '建议合理安排用电，避免集中使用',
                'Lower': '当前为用电低谷时段，可适当使用高耗能设备'
            },
            'Season': {
                'Winter': '冬季建议优化供暖策略，使用定时和分区控制',
                'Spring': '春季建议减少空调使用，多利用自然通风',
                'Summer': '夏季建议优化制冷策略，避免过度制冷',
                'Autumn': '秋季建议减少空调使用，多利用自然环境'
            }
        }
    
    def generate_recommendations(
        self,
        current_state: Dict[str, str],
        target_state: str = 'Peak',
        sensitivity_threshold: float = 0.15,
        top_k: int = 5
    ) -> List[Dict]:
        """
        生成节能建议
        
        参数:
        - current_state: 当前状态（特征取值）
        - target_state: 目标状态（通常是'Peak'，表示高负荷）
        - sensitivity_threshold: 敏感性阈值（只推荐影响大的特征）
        - top_k: 返回前k条建议
        
        返回:
        - 建议列表，每条包含：
          {
            'rank': 排名,
            'feature': 特征名,
            'feature_name': 可读特征名,
            'current_value': 当前值,
            'recommended_value': 推荐值,
            'current_prob': 当前概率,
            'expected_prob': 预期概率,
            'potential_reduction': 潜在降低幅度,
            'action': 行动建议文本
          }
        """
        # 确保已运行敏感性分析
        if self.ci.sensitivity_results is None:
            features = list(current_state.keys())
            self.ci.sensitivity_analysis(features, target_state=target_state)
        
        # 获取关键因素
        critical_factors = self.ci.identify_critical_factors(
            threshold=sensitivity_threshold,
            target_state=target_state
        )
        
        # 生成建议
        recommendations = []
        for idx, factor in enumerate(critical_factors[:top_k]):
            feature = factor['Feature']
            
            # 获取可读名称
            feature_name = self.feature_mapping.get(feature, feature)
            
            # 当前值和推荐值
            current_value = current_state.get(feature, 'Unknown')
            recommended_value = factor['Recommended_Value']
            
            # 概率信息
            current_prob = factor['Current_Max_Prob']
            expected_prob = factor['Expected_Prob']
            potential_reduction = factor['Potential_Reduction']
            
            # 生成行动建议
            action = self._generate_action_text(
                feature,
                current_value,
                recommended_value,
                potential_reduction
            )
            
            recommendations.append({
                'rank': idx + 1,
                'feature': feature,
                'feature_name': feature_name,
                'current_value': current_value,
                'recommended_value': recommended_value,
                'current_prob': current_prob,
                'expected_prob': expected_prob,
                'potential_reduction': potential_reduction,
                'action': action
            })
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def _generate_action_text(
        self,
        feature: str,
        current_value: str,
        recommended_value: str,
        impact: float
    ) -> str:
        """
        生成行动建议文本
        
        参数:
        - feature: 特征名
        - current_value: 当前值
        - recommended_value: 推荐值
        - impact: 影响程度
        
        返回:
        - 自然语言建议
        """
        # 获取模板
        if feature in self.action_templates:
            template = self.action_templates[feature].get(
                recommended_value,
                f"建议将{feature}调整为{recommended_value}"
            )
            action = template.format(value=recommended_value)
        else:
            # 通用模板
            action = f"建议将{self.feature_mapping.get(feature, feature)}从'{current_value}'调整为'{recommended_value}'"
        
        # 添加影响程度说明
        if impact > 0.3:
            magnitude = "显著"
        elif impact > 0.15:
            magnitude = "明显"
        else:
            magnitude = "一定程度"
        
        action += f"，预计可{magnitude}降低高峰负荷概率（约{impact:.1%}）"
        
        return action
    
    def format_report(
        self,
        recommendations: List[Dict],
        current_state: Dict[str, str],
        prediction: Optional[float] = None
    ) -> str:
        """
        生成格式化的推荐报告
        
        参数:
        - recommendations: 推荐列表
        - current_state: 当前状态
        - prediction: 预测的EDP值（可选）
        
        返回:
        - 格式化的报告文本
        """
        report = "=" * 60 + "\n"
        report += "           能源消耗预测与优化建议报告\n"
        report += "=" * 60 + "\n\n"
        
        # 当前状态
        report += "【当前状态】\n"
        for feature, value in current_state.items():
            feature_name = self.feature_mapping.get(feature, feature)
            report += f"  {feature_name}: {value}\n"
        
        # 预测值
        if prediction is not None:
            report += f"\n【预测负荷】\n  {prediction:.2f} kWh\n"
        
        # 建议
        if recommendations:
            report += f"\n【优化建议】（共{len(recommendations)}条）\n\n"
            for rec in recommendations:
                report += f"{rec['rank']}. {rec['action']}\n"
                report += f"   当前: {rec['feature_name']}={rec['current_value']} "
                report += f"→ 推荐: {rec['recommended_value']}\n"
                report += f"   预期效果: 高峰概率从 {rec['current_prob']:.1%} 降至 {rec['expected_prob']:.1%}\n\n"
        else:
            report += "\n【优化建议】\n  当前状态已较优，无需调整。\n"
        
        report += "=" * 60 + "\n"
        report += "注：以上建议基于因果贝叶斯网络推断，供参考。\n"
        
        return report
    
    def prioritize_recommendations(
        self,
        recommendations: List[Dict],
        cost_mapping: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        根据成本-效益比优先级排序建议
        
        参数:
        - recommendations: 原始建议列表
        - cost_mapping: 特征调整成本映射 {feature: cost}
        
        返回:
        - 重新排序的建议列表（增加cost_benefit_ratio字段）
        """
        if cost_mapping is None:
            # 默认成本（可根据实际情况调整）
            cost_mapping = {
                'Temperature': 0.5,  # 温度调整成本较低
                'Humidity': 0.7,
                'WindSpeed': 0.3,
                'Hour': 1.0,  # 时段无法调整，成本高
                'Season': 1.0
            }
        
        # 计算成本-效益比
        for rec in recommendations:
            cost = cost_mapping.get(rec['feature'], 0.5)
            benefit = rec['potential_reduction']
            rec['cost'] = cost
            rec['benefit'] = benefit
            rec['cost_benefit_ratio'] = benefit / cost if cost > 0 else 0
        
        # 按成本-效益比降序排序
        prioritized = sorted(
            recommendations,
            key=lambda x: x['cost_benefit_ratio'],
            reverse=True
        )
        
        # 更新排名
        for idx, rec in enumerate(prioritized):
            rec['rank'] = idx + 1
        
        logger.info("Recommendations prioritized by cost-benefit ratio")
        
        return prioritized
    
    def save_recommendations(
        self,
        recommendations: List[Dict],
        filepath: str
    ):
        """保存建议到CSV文件"""
        df = pd.DataFrame(recommendations)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {len(recommendations)} recommendations to {filepath}")


if __name__ == "__main__":
    # 示例（需要先训练因果推断模型）
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    from src.models.bayesian_net import CausalBayesianNetwork
    from src.inference.causal_inference import CausalInference
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # 模拟训练
    np.random.seed(42)
    data = pd.DataFrame({
        'Temperature': np.random.choice(['Low', 'Medium', 'High'], 200),
        'Humidity': np.random.choice(['Low', 'Medium', 'High'], 200),
        'WindSpeed': np.random.choice(['Low', 'Medium', 'High'], 200),
        'EDP_State': np.random.choice(['Lower', 'Normal', 'Peak'], 200)
    })
    
    bn = CausalBayesianNetwork(domain_edges=[
        ('Temperature', 'EDP_State'),
        ('Humidity', 'EDP_State')
    ])
    bn.learn_structure(data)
    bn.learn_parameters(data)
    
    ci = CausalInference(bn, target_var='EDP_State')
    
    # 创建推荐引擎
    engine = RecommendationEngine(ci)
    
    # 当前状态
    current_state = {
        'Temperature': 'High',
        'Humidity': 'Medium',
        'WindSpeed': 'Low'
    }
    
    # 生成建议
    recommendations = engine.generate_recommendations(
        current_state,
        target_state='Peak',
        top_k=3
    )
    
    print("\n" + "="*60)
    print("Generated Recommendations:")
    print("="*60)
    for rec in recommendations:
        print(f"{rec['rank']}. {rec['action']}")
    
    # 生成报告
    report = engine.format_report(recommendations, current_state, prediction=125.5)
    print("\n" + report)
