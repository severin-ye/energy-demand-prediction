"""
因果推断模块
提供基于贝叶斯网络的因果分析工具，包括敏感性分析和反事实推断
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CausalInference:
    """
    因果推断工具
    
    功能:
    1. 敏感性分析：分析每个特征变化对EDP的影响程度
    2. 龙卷风图可视化：展示特征重要性
    3. 反事实推断：计算"如果X=x会怎样"
    4. 因果效应量化：计算平均因果效应（ACE）
    
    参数:
    - bayesian_network: 训练好的CausalBayesianNetwork对象
    - target_var: 目标变量（通常是'EDP_State'）
    """
    
    def __init__(self, bayesian_network, target_var: str = 'EDP_State'):
        self.bn = bayesian_network
        self.target_var = target_var
        self.sensitivity_results = None
        
        logger.info(f"CausalInference initialized for target '{target_var}'")
    
    def sensitivity_analysis(
        self,
        features: List[str],
        baseline_evidence: Optional[Dict[str, str]] = None,
        target_state: str = 'Peak'
    ) -> pd.DataFrame:
        """
        敏感性分析：计算每个特征变化对目标状态概率的影响
        
        参数:
        - features: 要分析的特征列表
        - baseline_evidence: 基线证据（其他特征的固定值）
        - target_state: 目标状态（如'Peak'）
        
        返回:
        - DataFrame包含每个特征的敏感性指标
        """
        results = []
        
        for feature in features:
            # 获取该特征的所有可能取值
            feature_values = self._get_feature_values(feature)
            
            # 计算每个取值下的目标概率
            probs = []
            for value in feature_values:
                evidence = {feature: value}
                if baseline_evidence:
                    evidence.update(baseline_evidence)
                
                # 查询P(target_var | evidence)
                result = self.bn.query([self.target_var], evidence=evidence)
                prob = self._extract_prob(result, target_state)
                probs.append(prob)
            
            # 计算敏感性指标
            prob_range = max(probs) - min(probs)
            prob_std = np.std(probs)
            prob_max = max(probs)
            max_value = feature_values[np.argmax(probs)]
            
            results.append({
                'Feature': feature,
                'Prob_Range': prob_range,  # 概率变化范围
                'Prob_Std': prob_std,  # 概率标准差
                'Max_Prob': prob_max,  # 最大概率
                'Max_Value': max_value,  # 导致最大概率的取值
                'Values': feature_values,
                'Probs': probs
            })
        
        self.sensitivity_results = pd.DataFrame(results)
        
        # 按概率范围降序排序
        self.sensitivity_results = self.sensitivity_results.sort_values(
            'Prob_Range',
            ascending=False
        ).reset_index(drop=True)
        
        logger.info(f"Sensitivity analysis completed for {len(features)} features")
        
        return self.sensitivity_results
    
    def tornado_chart(
        self,
        top_k: int = 10,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        绘制龙卷风图（Tornado Chart）展示特征重要性
        
        参数:
        - top_k: 显示前k个重要特征
        - output_path: 输出路径（如果为None则显示）
        - figsize: 图片尺寸
        """
        if self.sensitivity_results is None:
            raise ValueError("Must run sensitivity_analysis() first")
        
        # 取前k个特征
        df_plot = self.sensitivity_results.head(top_k).copy()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制水平条形图
        y_pos = np.arange(len(df_plot))
        ax.barh(y_pos, df_plot['Prob_Range'], align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['Feature'])
        ax.invert_yaxis()  # 最重要的在顶部
        ax.set_xlabel('Probability Range', fontsize=12)
        ax.set_title(f'Tornado Chart: Feature Sensitivity on {self.target_var}', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            ax.text(
                row['Prob_Range'] + 0.01,
                i,
                f"{row['Prob_Range']:.3f}",
                va='center',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tornado chart saved to {output_path}")
        else:
            plt.show()
    
    def counterfactual_analysis(
        self,
        actual_evidence: Dict[str, str],
        intervention: Dict[str, str],
        target_state: str = 'Peak'
    ) -> Dict:
        """
        反事实推断：如果某些变量取不同值会怎样
        
        参数:
        - actual_evidence: 实际观测到的证据
        - intervention: 反事实干预
        - target_state: 目标状态
        
        返回:
        - {
            'actual_prob': 实际概率,
            'counterfactual_prob': 反事实概率,
            'causal_effect': 因果效应,
            'interpretation': 文字解释
          }
        """
        # 实际情况下的概率
        actual_result = self.bn.query([self.target_var], evidence=actual_evidence)
        actual_prob = self._extract_prob(actual_result, target_state)
        
        # 反事实情况下的概率（使用do-演算）
        counterfactual_result = self.bn.do_calculus(
            intervention=intervention,
            query_var=self.target_var,
            evidence=None  # do-演算不需要其他证据
        )
        counterfactual_prob = self._extract_prob(counterfactual_result, target_state)
        
        # 因果效应
        causal_effect = counterfactual_prob - actual_prob
        
        # 生成解释
        if causal_effect > 0:
            direction = "increase"
            magnitude = "significantly" if abs(causal_effect) > 0.2 else "moderately"
        elif causal_effect < 0:
            direction = "decrease"
            magnitude = "significantly" if abs(causal_effect) > 0.2 else "moderately"
        else:
            direction = "no change"
            magnitude = ""
        
        interpretation = (
            f"If {intervention} (instead of {actual_evidence}), "
            f"the probability of {self.target_var}={target_state} would "
            f"{magnitude} {direction} by {abs(causal_effect):.3f} "
            f"(from {actual_prob:.3f} to {counterfactual_prob:.3f})"
        )
        
        result = {
            'actual_prob': actual_prob,
            'counterfactual_prob': counterfactual_prob,
            'causal_effect': causal_effect,
            'interpretation': interpretation
        }
        
        logger.info(f"Counterfactual: {interpretation}")
        
        return result
    
    def average_causal_effect(
        self,
        treatment_var: str,
        treatment_value: str,
        control_value: str,
        target_state: str = 'Peak',
        data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        计算平均因果效应（ACE）：E[Y | do(X=x1)] - E[Y | do(X=x0)]
        
        参数:
        - treatment_var: 干预变量
        - treatment_value: 干预值（处理组）
        - control_value: 对照值（对照组）
        - target_state: 目标状态
        - data: 数据集（用于计算期望）
        
        返回:
        - ACE值
        """
        # 处理组概率
        treatment_result = self.bn.do_calculus(
            intervention={treatment_var: treatment_value},
            query_var=self.target_var
        )
        treatment_prob = self._extract_prob(treatment_result, target_state)
        
        # 对照组概率
        control_result = self.bn.do_calculus(
            intervention={treatment_var: control_value},
            query_var=self.target_var
        )
        control_prob = self._extract_prob(control_result, target_state)
        
        # ACE
        ace = treatment_prob - control_prob
        
        logger.info(
            f"ACE: P({self.target_var}={target_state} | do({treatment_var}={treatment_value})) - "
            f"P({self.target_var}={target_state} | do({treatment_var}={control_value})) = {ace:.3f}"
        )
        
        return ace
    
    def identify_critical_factors(
        self,
        threshold: float = 0.15,
        target_state: str = 'Peak'
    ) -> List[Dict]:
        """
        识别关键影响因素（敏感性分析结果中概率范围超过阈值的特征）
        
        参数:
        - threshold: 概率范围阈值
        - target_state: 目标状态
        
        返回:
        - 关键因素列表（包含推荐建议）
        """
        if self.sensitivity_results is None:
            raise ValueError("Must run sensitivity_analysis() first")
        
        critical = self.sensitivity_results[
            self.sensitivity_results['Prob_Range'] >= threshold
        ].copy()
        
        recommendations = []
        for _, row in critical.iterrows():
            # 找到最佳取值（最小化目标概率）
            min_prob_idx = np.argmin(row['Probs'])
            optimal_value = row['Values'][min_prob_idx]
            
            recommendations.append({
                'Feature': row['Feature'],
                'Impact': row['Prob_Range'],
                'Current_Max_Prob': row['Max_Prob'],
                'Recommended_Value': optimal_value,
                'Expected_Prob': row['Probs'][min_prob_idx],
                'Potential_Reduction': row['Max_Prob'] - row['Probs'][min_prob_idx]
            })
        
        logger.info(f"Identified {len(recommendations)} critical factors")
        
        return recommendations
    
    def _get_feature_values(self, feature: str) -> List[str]:
        """获取特征的所有可能取值"""
        if self.bn.model is None:
            raise ValueError("Bayesian network not trained")
        
        # 从CPD获取取值
        cpd = self.bn.model.get_cpds(feature)
        return list(cpd.state_names[feature])
    
    def _extract_prob(self, query_result, state: str) -> float:
        """从查询结果中提取特定状态的概率"""
        # pgmpy查询结果是DiscreteFactor对象
        values = query_result.values
        states = query_result.state_names[self.target_var]
        
        # 找到目标状态的索引
        if state in states:
            idx = states.index(state)
            return float(values[idx])
        else:
            logger.warning(f"State '{state}' not found in {states}")
            return 0.0
    
    def generate_explanation(
        self,
        current_state: Dict[str, str],
        target_state: str = 'Peak',
        top_k: int = 3
    ) -> str:
        """
        生成自然语言解释
        
        参数:
        - current_state: 当前状态（特征取值）
        - target_state: 目标状态
        - top_k: 解释前k个重要因素
        
        返回:
        - 自然语言解释文本
        """
        if self.sensitivity_results is None:
            raise ValueError("Must run sensitivity_analysis() first")
        
        # 获取前k个重要因素
        top_features = self.sensitivity_results.head(top_k)
        
        explanation = f"Analysis of {self.target_var}={target_state}:\n\n"
        
        for idx, row in top_features.iterrows():
            feature = row['Feature']
            impact = row['Prob_Range']
            
            # 当前值对应的概率
            if feature in current_state:
                current_value = current_state[feature]
                value_idx = row['Values'].index(current_value) if current_value in row['Values'] else 0
                current_prob = row['Probs'][value_idx]
                
                explanation += (
                    f"{idx + 1}. {feature}: "
                    f"Current value '{current_value}' leads to {current_prob:.2%} probability. "
                    f"This feature has a {impact:.2%} impact range.\n"
                )
        
        return explanation


if __name__ == "__main__":
    # 示例（需要先训练贝叶斯网络）
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    from src.models.bayesian_net import CausalBayesianNetwork
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据和训练网络
    np.random.seed(42)
    data = pd.DataFrame({
        'Temperature': np.random.choice(['Low', 'Medium', 'High'], 200),
        'Humidity': np.random.choice(['Low', 'Medium', 'High'], 200),
        'WindSpeed': np.random.choice(['Low', 'Medium', 'High'], 200),
        'EDP_State': np.random.choice(['Lower', 'Normal', 'Peak'], 200)
    })
    
    domain_edges = [
        ('Temperature', 'EDP_State'),
        ('Humidity', 'EDP_State')
    ]
    
    bn = CausalBayesianNetwork(domain_edges=domain_edges)
    bn.learn_structure(data)
    bn.learn_parameters(data)
    
    # 创建因果推断工具
    ci = CausalInference(bn, target_var='EDP_State')
    
    # 敏感性分析
    features = ['Temperature', 'Humidity', 'WindSpeed']
    sensitivity = ci.sensitivity_analysis(features, target_state='Peak')
    print("\nSensitivity Analysis:")
    print(sensitivity[['Feature', 'Prob_Range', 'Max_Value', 'Max_Prob']])
    
    # 龙卷风图（不显示，仅测试）
    # ci.tornado_chart(top_k=3)
    
    # 反事实推断
    counterfactual = ci.counterfactual_analysis(
        actual_evidence={'Temperature': 'High', 'Humidity': 'Medium'},
        intervention={'Temperature': 'Low'},
        target_state='Peak'
    )
    print(f"\nCounterfactual Analysis:")
    print(counterfactual['interpretation'])
    
    # 平均因果效应
    ace = ci.average_causal_effect(
        treatment_var='Temperature',
        treatment_value='High',
        control_value='Low',
        target_state='Peak'
    )
    print(f"\nACE: {ace:.3f}")
