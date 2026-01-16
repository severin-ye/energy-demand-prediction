"""
关联规则挖掘模块
使用Apriori算法挖掘频繁项集和关联规则，为贝叶斯网络提供候选边
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from mlxtend.frequent_patterns import apriori, association_rules
import logging

logger = logging.getLogger(__name__)


class AssociationRuleMiner:
    """
    关联规则挖掘器
    
    功能:
    1. 将连续特征离散化后编码为事务格式
    2. 使用Apriori算法挖掘频繁项集
    3. 生成关联规则并筛选与EDP相关的规则
    4. 提取候选因果关系边
    
    参数:
    - min_support: 最小支持度阈值（默认0.05）
    - min_confidence: 最小置信度阈值（默认0.6）
    - min_lift: 最小提升度阈值（默认1.2）
    """
    
    def __init__(
        self,
        min_support: float = 0.05,
        min_confidence: float = 0.6,
        min_lift: float = 1.2
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        
        self.frequent_itemsets = None
        self.rules = None
        self.edp_rules = None
        self.candidate_edges = None
        
        logger.info(
            f"AssociationRuleMiner initialized with "
            f"min_support={min_support}, min_confidence={min_confidence}, min_lift={min_lift}"
        )
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        edp_col: str = 'EDP_State'
    ) -> pd.DataFrame:
        """
        准备数据：将离散化后的数据转换为One-hot编码
        
        参数:
        - df: 离散化后的DataFrame（特征值为类别标签）
        - edp_col: EDP状态列名
        
        返回:
        - 布尔型DataFrame，适用于Apriori算法
        """
        # 验证EDP列存在
        if edp_col not in df.columns:
            raise ValueError(f"EDP column '{edp_col}' not found in DataFrame")
        
        # One-hot编码所有列
        df_encoded = pd.get_dummies(df, prefix_sep='=')
        
        # 转换为布尔型
        df_encoded = df_encoded.astype(bool)
        
        logger.info(
            f"Prepared data: {len(df)} transactions, {len(df_encoded.columns)} items"
        )
        
        return df_encoded
    
    def mine_frequent_itemsets(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """
        挖掘频繁项集
        
        参数:
        - df_encoded: One-hot编码后的布尔型DataFrame
        
        返回:
        - 频繁项集DataFrame（包含support列）
        """
        self.frequent_itemsets = apriori(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True
        )
        
        logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        
        return self.frequent_itemsets
    
    def generate_rules(
        self,
        metric: str = 'confidence',
        min_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        生成关联规则
        
        参数:
        - metric: 评估指标（'confidence', 'lift', 'leverage', 'conviction'）
        - min_threshold: 最小阈值（如果为None，使用类初始化时的阈值）
        
        返回:
        - 关联规则DataFrame
        """
        if self.frequent_itemsets is None:
            raise ValueError("Must call mine_frequent_itemsets() first")
        
        if min_threshold is None:
            min_threshold = self.min_confidence if metric == 'confidence' else self.min_lift
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # 过滤提升度
        if metric != 'lift':
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        logger.info(f"Generated {len(self.rules)} association rules")
        
        return self.rules
    
    def filter_edp_rules(self, edp_prefix: str = 'EDP_State=') -> pd.DataFrame:
        """
        筛选与EDP相关的规则（EDP在后件）
        
        参数:
        - edp_prefix: EDP列的前缀（One-hot编码后）
        
        返回:
        - 包含EDP的关联规则
        """
        if self.rules is None:
            raise ValueError("Must call generate_rules() first")
        
        def contains_edp(consequents):
            """检查后件是否包含EDP"""
            return any(edp_prefix in str(item) for item in consequents)
        
        self.edp_rules = self.rules[
            self.rules['consequents'].apply(contains_edp)
        ].copy()
        
        # 按置信度降序排序
        self.edp_rules = self.edp_rules.sort_values(
            'confidence',
            ascending=False
        ).reset_index(drop=True)
        
        logger.info(f"Filtered to {len(self.edp_rules)} EDP-related rules")
        
        return self.edp_rules
    
    def rules_to_constraints(self, top_k: int = 50) -> List[Tuple[str, str]]:
        """
        将关联规则转换为贝叶斯网络的候选边约束
        
        参数:
        - top_k: 提取前k条规则
        
        返回:
        - [(前件特征, 后件特征), ...] 列表
        """
        if self.edp_rules is None:
            raise ValueError("Must call filter_edp_rules() first")
        
        edges = []
        edp_rules_subset = self.edp_rules.head(top_k)
        
        for _, rule in edp_rules_subset.iterrows():
            # 提取前件特征名（去除离散化标签）
            antecedents = [item.split('=')[0] for item in rule['antecedents']]
            
            # 提取后件特征名
            consequents = [item.split('=')[0] for item in rule['consequents']]
            
            # 创建边（前件 -> 后件）
            for ant in antecedents:
                for cons in consequents:
                    if ant != cons:  # 避免自环
                        edges.append((ant, cons))
        
        # 去重
        self.candidate_edges = list(set(edges))
        
        logger.info(f"Extracted {len(self.candidate_edges)} candidate edges from top {top_k} rules")
        
        return self.candidate_edges
    
    def get_rule_summary(self, n: int = 10) -> pd.DataFrame:
        """
        获取规则摘要（用于可视化和报告）
        
        参数:
        - n: 返回前n条规则
        
        返回:
        - 精简的规则DataFrame
        """
        if self.edp_rules is None:
            raise ValueError("Must call filter_edp_rules() first")
        
        summary = self.edp_rules.head(n)[[
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]].copy()
        
        # 格式化显示
        summary['antecedents'] = summary['antecedents'].apply(lambda x: ', '.join(x))
        summary['consequents'] = summary['consequents'].apply(lambda x: ', '.join(x))
        
        return summary
    
    def save_rules(self, filepath: str):
        """保存规则到CSV文件"""
        if self.edp_rules is None:
            raise ValueError("No rules to save")
        
        # 转换frozenset为字符串
        df_save = self.edp_rules.copy()
        df_save['antecedents'] = df_save['antecedents'].apply(lambda x: ', '.join(x))
        df_save['consequents'] = df_save['consequents'].apply(lambda x: ', '.join(x))
        
        df_save.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df_save)} rules to {filepath}")
    
    def load_rules(self, filepath: str):
        """从CSV文件加载规则"""
        df_load = pd.read_csv(filepath)
        
        # 转换字符串为frozenset
        df_load['antecedents'] = df_load['antecedents'].apply(
            lambda x: frozenset(x.split(', '))
        )
        df_load['consequents'] = df_load['consequents'].apply(
            lambda x: frozenset(x.split(', '))
        )
        
        self.edp_rules = df_load
        logger.info(f"Loaded {len(df_load)} rules from {filepath}")


if __name__ == "__main__":
    # 示例使用
    logging.basicConfig(level=logging.INFO)
    
    # 模拟离散化数据
    np.random.seed(42)
    df_discrete = pd.DataFrame({
        'Temperature': np.random.choice(['Low', 'Medium', 'High'], 100),
        'Humidity': np.random.choice(['Low', 'Medium', 'High'], 100),
        'WindSpeed': np.random.choice(['Low', 'Medium', 'High'], 100),
        'EDP_State': np.random.choice(['Lower', 'Normal', 'Peak'], 100, p=[0.2, 0.5, 0.3])
    })
    
    # 创建挖掘器
    miner = AssociationRuleMiner(min_support=0.05, min_confidence=0.6, min_lift=1.2)
    
    # 数据准备
    df_encoded = miner.prepare_data(df_discrete, edp_col='EDP_State')
    print(f"\nEncoded data shape: {df_encoded.shape}")
    
    # 挖掘频繁项集
    frequent = miner.mine_frequent_itemsets(df_encoded)
    print(f"\nTop 5 frequent itemsets:")
    print(frequent.head())
    
    # 生成规则
    rules = miner.generate_rules()
    print(f"\nGenerated {len(rules)} rules")
    
    # 筛选EDP规则
    edp_rules = miner.filter_edp_rules()
    print(f"\nTop 5 EDP rules:")
    print(miner.get_rule_summary(5))
    
    # 提取候选边
    edges = miner.rules_to_constraints(top_k=10)
    print(f"\nCandidate edges for Bayesian Network:")
    for edge in edges[:10]:
        print(f"  {edge[0]} -> {edge[1]}")
