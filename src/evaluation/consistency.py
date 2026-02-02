"""
解释一致性评估模块

用于评估贝叶斯网络在训练集和测试集上的解释一致性（论文表5）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ExplanationConsistencyEvaluator:
    """
    解释一致性评估器
    
    评估指标:
    - 余弦相似度: 训练集与测试集特征重要性的相似程度
    - 越高表示解释越稳定，不易过拟合
    
    论文基准（表5）:
    - 本方法: Peak=0.99940, Normal=0.99983, Lower=0.99974
    - SHAP: 0.95-0.96
    - LIME: 0.70-0.75
    - PD Variance: 0.81-0.96
    """
    
    def __init__(self, bayesian_network):
        """
        参数:
            bayesian_network: 训练好的贝叶斯网络
        """
        self.bn = bayesian_network
        self.train_importance = None
        self.test_importance = None
    
    def extract_feature_importance(
        self,
        data: pd.DataFrame,
        target_var: str = 'EDP_State',
        evidence_vars: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        从贝叶斯网络提取特征重要性
        
        方法: 计算每个特征对目标变量的条件概率影响
        
        参数:
            data: 数据集
            target_var: 目标变量（EDP_State）
            evidence_vars: 证据变量列表（如果为None则使用所有非目标变量）
        
        返回:
            特征重要性字典 {state: importance_vector}
        """
        if evidence_vars is None:
            evidence_vars = [col for col in data.columns if col != target_var]
        
        # 获取目标变量的可能取值
        target_states = data[target_var].unique()
        
        importance_dict = {}
        
        for state in target_states:
            feature_scores = []
            
            for feature in evidence_vars:
                # 计算特征的边际影响
                # P(target=state | feature) 的变化程度
                feature_values = data[feature].unique()
                
                probs = []
                for value in feature_values:
                    try:
                        result = self.bn.query(
                            variables=[target_var],
                            evidence={feature: value}
                        )
                        prob = result.values[list(result.state_names[target_var]).index(state)]
                        probs.append(prob)
                    except:
                        probs.append(0.0)
                
                # 使用概率的标准差作为重要性得分
                importance = np.std(probs) if len(probs) > 0 else 0.0
                feature_scores.append(importance)
            
            # 归一化
            feature_scores = np.array(feature_scores)
            if np.sum(feature_scores) > 0:
                feature_scores = feature_scores / np.sum(feature_scores)
            
            importance_dict[state] = feature_scores
        
        return importance_dict
    
    def compute_markov_blanket_importance(
        self,
        target_var: str = 'EDP_State'
    ) -> Dict[str, List[str]]:
        """
        基于马尔可夫毯计算特征重要性
        
        马尔可夫毯包含:
        - 父节点（直接原因）
        - 子节点（直接结果）
        - 子节点的其他父节点
        
        返回:
            每个状态的马尔可夫毯特征
        """
        mb = self.bn.get_markov_blanket(target_var)
        
        # 计算马尔可夫毯中每个特征的重要性
        importance = {}
        for feature in mb:
            # 检查是否是父节点（直接因果）
            parents = self.bn.get_parents(target_var)
            if feature in parents:
                importance[feature] = 1.0  # 最高重要性
            else:
                importance[feature] = 0.5  # 间接影响
        
        return importance
    
    def evaluate_consistency(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_var: str = 'EDP_State'
    ) -> Dict[str, float]:
        """
        评估训练集和测试集解释的一致性
        
        参数:
            train_data: 训练集
            test_data: 测试集
            target_var: 目标变量
        
        返回:
            每个状态的余弦相似度 {state: similarity}
        """
        logger.info("提取训练集特征重要性...")
        self.train_importance = self.extract_feature_importance(
            train_data,
            target_var
        )
        
        logger.info("提取测试集特征重要性...")
        self.test_importance = self.extract_feature_importance(
            test_data,
            target_var
        )
        
        # 计算每个状态的余弦相似度
        similarities = {}
        
        for state in self.train_importance.keys():
            if state in self.test_importance:
                train_vec = self.train_importance[state].reshape(1, -1)
                test_vec = self.test_importance[state].reshape(1, -1)
                
                sim = cosine_similarity(train_vec, test_vec)[0, 0]
                similarities[state] = sim
        
        logger.info("一致性评估完成:")
        for state, sim in similarities.items():
            logger.info(f"  {state}: {sim:.5f}")
        
        return similarities
    
    def compare_with_baselines(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_var: str = 'EDP_State'
    ) -> pd.DataFrame:
        """
        与baseline方法对比（SHAP、LIME等）
        
        注: 实际实现需要集成SHAP/LIME库
        这里提供框架
        
        返回:
            对比结果DataFrame
        """
        # 本方法的一致性
        bn_consistency = self.evaluate_consistency(
            train_data,
            test_data,
            target_var
        )
        
        results = []
        
        for state, sim in bn_consistency.items():
            results.append({
                'State': state,
                'Method': 'Bayesian Network (Ours)',
                'Cosine Similarity': sim
            })
        
        # TODO: 添加SHAP、LIME、PD Variance的计算
        # 这里使用论文报告的值作为参考
        paper_baselines = {
            'SHAP': {'Peak': 0.95, 'Normal': 0.96, 'Lower': 0.95},
            'LIME': {'Peak': 0.70, 'Normal': 0.75, 'Lower': 0.72},
            'PD Variance': {'Peak': 0.81, 'Normal': 0.96, 'Lower': 0.85}
        }
        
        for method, values in paper_baselines.items():
            for state, sim in values.items():
                results.append({
                    'State': state,
                    'Method': f'{method} (Paper)',
                    'Cosine Similarity': sim
                })
        
        df = pd.DataFrame(results)
        
        logger.info("\n=== 一致性对比 ===")
        logger.info(f"\n{df.to_string()}")
        
        return df
    
    def visualize_consistency(
        self,
        save_path: str = None
    ):
        """
        可视化一致性结果
        
        参数:
            save_path: 保存路径（如果为None则显示）
        """
        import matplotlib.pyplot as plt
        
        if self.train_importance is None or self.test_importance is None:
            raise ValueError("Must call evaluate_consistency() first")
        
        states = list(self.train_importance.keys())
        n_states = len(states)
        
        fig, axes = plt.subplots(1, n_states, figsize=(15, 5))
        
        if n_states == 1:
            axes = [axes]
        
        for i, state in enumerate(states):
            train_vec = self.train_importance[state]
            test_vec = self.test_importance[state]
            
            x = np.arange(len(train_vec))
            width = 0.35
            
            axes[i].bar(x - width/2, train_vec, width, label='Train', alpha=0.8)
            axes[i].bar(x + width/2, test_vec, width, label='Test', alpha=0.8)
            
            axes[i].set_title(f'State: {state}')
            axes[i].set_xlabel('Feature Index')
            axes[i].set_ylabel('Importance')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到 {save_path}")
        else:
            plt.show()


def evaluate_explanation_stability(
    bayesian_network,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_var: str = 'EDP_State',
    output_dir: str = None
) -> Dict[str, float]:
    """
    便捷函数: 评估解释稳定性
    
    参数:
        bayesian_network: 贝叶斯网络
        train_data: 训练集
        test_data: 测试集
        target_var: 目标变量
        output_dir: 输出目录
    
    返回:
        一致性得分字典
    """
    evaluator = ExplanationConsistencyEvaluator(bayesian_network)
    
    # 评估一致性
    consistency = evaluator.evaluate_consistency(
        train_data,
        test_data,
        target_var
    )
    
    # 与baseline对比
    comparison_df = evaluator.compare_with_baselines(
        train_data,
        test_data,
        target_var
    )
    
    # 保存结果
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存对比表
        comparison_df.to_csv(
            os.path.join(output_dir, 'consistency_comparison.csv'),
            index=False
        )
        
        # 可视化
        evaluator.visualize_consistency(
            os.path.join(output_dir, 'consistency_visualization.png')
        )
    
    return consistency


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例使用
    print("解释一致性评估模块已创建")
    print("使用方法:")
    print("1. 训练贝叶斯网络")
    print("2. 调用 evaluate_explanation_stability()")
    print("3. 获得各状态的一致性得分")
