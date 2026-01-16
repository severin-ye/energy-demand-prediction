"""
因果贝叶斯网络模块
结合领域知识约束和数据学习，构建因果解释的贝叶斯网络
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import (
    HillClimbSearch,
    BIC,
    K2,
    MaximumLikelihoodEstimator,
    BayesianEstimator
)
from pgmpy.inference import VariableElimination
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class CausalBayesianNetwork:
    """
    因果贝叶斯网络
    
    功能:
    1. 结合领域知识约束（白名单/黑名单边）
    2. 使用BIC评分进行结构学习
    3. 最大似然估计参数
    4. 支持因果推断（do-演算）
    
    参数:
    - domain_edges: 领域知识强制边（白名单）
    - forbidden_edges: 禁止边（黑名单，如防止时间倒流）
    - score_fn: 评分函数（'bic' 或 'k2'）
    """
    
    def __init__(
        self,
        domain_edges: Optional[List[Tuple[str, str]]] = None,
        forbidden_edges: Optional[List[Tuple[str, str]]] = None,
        score_fn: str = 'bic'
    ):
        self.domain_edges = domain_edges or []
        self.forbidden_edges = forbidden_edges or []
        self.score_fn = score_fn
        
        self.model = None
        self.inference_engine = None
        self.feature_names = None
        
        logger.info(
            f"CausalBayesianNetwork initialized with {len(self.domain_edges)} "
            f"domain edges and {len(self.forbidden_edges)} forbidden edges"
        )
    
    def _create_constraints(
        self,
        candidate_edges: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        创建结构学习约束
        
        参数:
        - candidate_edges: 从关联规则提取的候选边
        
        返回:
        - (白名单边, 黑名单边)
        """
        # 白名单：领域知识边 + 候选边
        white_list = self.domain_edges.copy()
        if candidate_edges:
            white_list.extend(candidate_edges)
        
        # 去重
        white_list = list(set(white_list))
        
        # 黑名单：禁止边
        black_list = self.forbidden_edges.copy()
        
        logger.info(f"Constraints: {len(white_list)} white-list, {len(black_list)} black-list edges")
        
        return white_list, black_list
    
    def learn_structure(
        self,
        data: pd.DataFrame,
        candidate_edges: Optional[List[Tuple[str, str]]] = None,
        max_iter: int = 100,
        max_indegree: Optional[int] = None
    ) -> DiscreteBayesianNetwork:
        """
        学习贝叶斯网络结构
        
        参数:
        - data: 离散化后的数据
        - candidate_edges: 候选边（来自关联规则）
        - max_iter: 爬山搜索最大迭代次数
        - max_indegree: 节点最大入度（限制复杂度）
        
        返回:
        - DiscreteBayesianNetwork模型
        """
        self.feature_names = list(data.columns)
        
        # 创建约束
        white_list, black_list = self._create_constraints(candidate_edges)
        
        # 选择评分函数
        if self.score_fn == 'bic':
            scoring = BIC(data)
        elif self.score_fn == 'k2':
            scoring = K2(data)
        else:
            raise ValueError(f"Unknown score function: {self.score_fn}")
        
        # 爬山搜索结构学习
        hc = HillClimbSearch(data)
        
        try:
            # pgmpy白名单/黑名单使用方式
            best_model = hc.estimate(
                scoring_method=scoring,
                max_iter=max_iter,
                max_indegree=max_indegree,
                white_list=white_list if white_list else None,
                black_list=black_list if black_list else None
            )
        except Exception as e:
            logger.warning(f"Structure learning with constraints failed: {e}")
            logger.info("Falling back to unconstrained structure learning")
            best_model = hc.estimate(
                scoring_method=scoring,
                max_iter=max_iter,
                max_indegree=max_indegree
            )
        
        self.model = DiscreteBayesianNetwork(best_model.edges())
        
        logger.info(
            f"Learned structure with {len(self.model.nodes())} nodes "
            f"and {len(self.model.edges())} edges"
        )
        
        return self.model
    
    def learn_parameters(
        self,
        data: pd.DataFrame,
        estimator: str = 'mle',
        prior_type: str = 'BDeu',
        equivalent_sample_size: int = 10
    ):
        """
        学习条件概率表（CPT）参数
        
        参数:
        - data: 训练数据
        - estimator: 'mle'（极大似然）或 'bayes'（贝叶斯估计）
        - prior_type: 先验类型（用于贝叶斯估计）
        - equivalent_sample_size: 等效样本大小（用于BDeu先验）
        """
        if self.model is None:
            raise ValueError("Must call learn_structure() first")
        
        if estimator == 'mle':
            est = MaximumLikelihoodEstimator(self.model, data)
        elif estimator == 'bayes':
            est = BayesianEstimator(
                self.model,
                data,
                prior_type=prior_type,
                equivalent_sample_size=equivalent_sample_size
            )
        else:
            raise ValueError(f"Unknown estimator: {estimator}")
        
        # 拟合所有CPT
        for node in self.model.nodes():
            cpd = est.estimate_cpd(node)
            self.model.add_cpds(cpd)
        
        # 验证模型
        assert self.model.check_model(), "Model CPDs are inconsistent"
        
        logger.info(f"Learned parameters using {estimator} estimator")
    
    def create_inference_engine(self):
        """创建变量消去推理引擎"""
        if self.model is None:
            raise ValueError("Must train the model first")
        
        self.inference_engine = VariableElimination(self.model)
        logger.info("Inference engine created")
    
    def query(
        self,
        variables: List[str],
        evidence: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        贝叶斯推断查询 P(variables | evidence)
        
        参数:
        - variables: 查询变量列表
        - evidence: 证据字典 {变量名: 取值}
        
        返回:
        - 查询结果字典
        """
        if self.inference_engine is None:
            self.create_inference_engine()
        
        result = self.inference_engine.query(
            variables=variables,
            evidence=evidence
        )
        
        return result
    
    def do_calculus(
        self,
        intervention: Dict[str, str],
        query_var: str,
        evidence: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Do-演算：P(query_var | do(intervention), evidence)
        
        参数:
        - intervention: 干预变量 {变量名: 干预值}
        - query_var: 查询变量
        - evidence: 其他证据
        
        返回:
        - 干预后的概率分布
        """
        if self.model is None:
            raise ValueError("Must train the model first")
        
        # 创建干预后的模型（移除指向干预变量的边）
        intervened_model = self.model.copy()
        
        for var in intervention.keys():
            # 移除所有指向干预变量的边
            parents = list(intervened_model.get_parents(var))
            for parent in parents:
                intervened_model.remove_edge(parent, var)
        
        # 在干预模型上推断
        intervened_inference = VariableElimination(intervened_model)
        
        # 合并干预和证据
        combined_evidence = intervention.copy()
        if evidence:
            combined_evidence.update(evidence)
        
        result = intervened_inference.query(
            variables=[query_var],
            evidence=combined_evidence
        )
        
        logger.info(f"Do-calculus: P({query_var} | do({intervention}), {evidence})")
        
        return result
    
    def get_parents(self, node: str) -> List[str]:
        """获取节点的父节点（直接原因）"""
        if self.model is None:
            raise ValueError("Model not trained")
        return list(self.model.get_parents(node))
    
    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        获取马尔可夫毯（父节点+子节点+子节点的其他父节点）
        这是与目标变量直接相关的所有变量
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        mb = set()
        
        # 父节点
        mb.update(self.model.get_parents(node))
        
        # 子节点
        children = self.model.get_children(node)
        mb.update(children)
        
        # 子节点的其他父节点
        for child in children:
            mb.update(self.model.get_parents(child))
        
        # 移除自身
        mb.discard(node)
        
        return mb
    
    def get_causal_paths(
        self,
        source: str,
        target: str
    ) -> List[List[str]]:
        """
        获取从source到target的所有有向路径（因果路径）
        
        参数:
        - source: 源节点
        - target: 目标节点
        
        返回:
        - 路径列表，每个路径是节点序列
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # 转换为networkx DiGraph
        G = nx.DiGraph(self.model.edges())
        
        try:
            paths = list(nx.all_simple_paths(G, source, target))
            logger.info(f"Found {len(paths)} causal paths from {source} to {target}")
            return paths
        except nx.NetworkXNoPath:
            logger.warning(f"No causal path from {source} to {target}")
            return []
    
    def save_model(self, filepath: str):
        """保存模型结构和参数"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # 保存为BIF格式（Bayesian Interchange Format）
        from pgmpy.readwrite import BIFWriter
        writer = BIFWriter(self.model)
        writer.write_bif(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        from pgmpy.readwrite import BIFReader
        reader = BIFReader(filepath)
        self.model = reader.get_model()
        
        logger.info(f"Model loaded from {filepath}")
    
    def visualize_structure(self, output_path: Optional[str] = None):
        """
        可视化网络结构
        
        参数:
        - output_path: 输出图片路径（如果为None，则显示）
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph(self.model.edges())
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=2000,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray'
        )
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network structure saved to {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # 示例使用
    logging.basicConfig(level=logging.INFO)
    
    # 模拟离散化数据
    np.random.seed(42)
    data = pd.DataFrame({
        'Temperature': np.random.choice(['Low', 'Medium', 'High'], 200),
        'Humidity': np.random.choice(['Low', 'Medium', 'High'], 200),
        'WindSpeed': np.random.choice(['Low', 'Medium', 'High'], 200),
        'EDP_State': np.random.choice(['Lower', 'Normal', 'Peak'], 200)
    })
    
    # 领域知识：温度、湿度、风速都可能影响EDP
    domain_edges = [
        ('Temperature', 'EDP_State'),
        ('Humidity', 'EDP_State'),
        ('WindSpeed', 'EDP_State')
    ]
    
    # 创建贝叶斯网络
    bn = CausalBayesianNetwork(domain_edges=domain_edges)
    
    # 学习结构
    model = bn.learn_structure(data)
    print(f"\nLearned structure edges: {model.edges()}")
    
    # 学习参数
    bn.learn_parameters(data, estimator='mle')
    
    # 推断示例
    bn.create_inference_engine()
    
    # 查询：P(EDP_State | Temperature=High)
    result = bn.query(['EDP_State'], evidence={'Temperature': 'High'})
    print(f"\nP(EDP_State | Temperature=High):")
    print(result)
    
    # Do-演算：P(EDP_State | do(Temperature=Low))
    result_do = bn.do_calculus(
        intervention={'Temperature': 'Low'},
        query_var='EDP_State'
    )
    print(f"\nP(EDP_State | do(Temperature=Low)):")
    print(result_do)
    
    # 获取EDP的马尔可夫毯
    mb = bn.get_markov_blanket('EDP_State')
    print(f"\nMarkov Blanket of EDP_State: {mb}")
