"""
Causal Bayesian Network Module
Combines domain knowledge constraints and data-driven learning to construct causal explainable Bayesian Networks.
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
    Causal Bayesian Network
    
    Functions:
    1. Incorporates domain knowledge constraints (white-list/black-list edges).
    2. Performs structure learning using the BIC score.
    3. Estimates parameters using Maximum Likelihood.
    4. Supports causal inference (do-calculus).
    
    Parameters:
    - domain_edges: Forced edges from domain knowledge (white-list).
    - forbidden_edges: Forbidden edges (black-list, e.g., to prevent reverse causality in time).
    - score_fn: Scoring function ('bic' or 'k2').
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
        Creates constraints for structure learning.
        
        Parameters:
        - candidate_edges: Candidate edges extracted from association rules.
        
        Returns:
        - (white_list_edges, black_list_edges)
        """
        # White-list: Domain knowledge edges + candidate edges
        white_list = self.domain_edges.copy()
        if candidate_edges:
            white_list.extend(candidate_edges)
        
        # Remove duplicates
        white_list = list(set(white_list))
        
        # Black-list: Forbidden edges
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
        Learns the Bayesian Network structure.
        
        Parameters:
        - data: Discretized data.
        - candidate_edges: Candidate edges (from association rules).
        - max_iter: Max iterations for Hill Climbing search.
        - max_indegree: Max in-degree per node (limits complexity).
        
        Returns:
        - DiscreteBayesianNetwork model.
        """
        self.feature_names = list(data.columns)
        
        # Create constraints
        white_list, black_list = self._create_constraints(candidate_edges)
        
        # Select scoring function
        if self.score_fn == 'bic':
            scoring = BIC(data)
        elif self.score_fn == 'k2':
            scoring = K2(data)
        else:
            raise ValueError(f"Unknown score function: {self.score_fn}")
        
        # Hill Climbing Structure Learning
        hc = HillClimbSearch(data)
        
        try:
            # pgmpy usage for white-list/black-list
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
        Learns the Conditional Probability Table (CPT) parameters.
        
        Parameters:
        - data: Training data.
        - estimator: 'mle' (Maximum Likelihood) or 'bayes' (Bayesian Estimator).
        - prior_type: Type of prior (for Bayesian estimation).
        - equivalent_sample_size: Equivalent sample size (for BDeu prior).
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
        
        # Fit all CPTs
        for node in self.model.nodes():
            cpd = est.estimate_cpd(node)
            self.model.add_cpds(cpd)
        
        # Validate model
        assert self.model.check_model(), "Model CPDs are inconsistent"
        
        logger.info(f"Learned parameters using {estimator} estimator")
    
    def create_inference_engine(self):
        """Creates a Variable Elimination inference engine."""
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
        Bayesian Inference Query P(variables | evidence).
        
        Parameters:
        - variables: List of query variables.
        - evidence: Evidence dictionary {variable_name: value}.
        
        Returns:
            Dictionary of query results.
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
        Do-calculus: P(query_var | do(intervention), evidence).
        
        Parameters:
        - intervention: Intervened variables {variable_name: intervention_value}.
        - query_var: Variable to query.
        - evidence: Other evidence.
        
        Returns:
            Probability distribution after intervention.
        """
        if self.model is None:
            raise ValueError("Must train the model first")
        
        # Create post-intervention model (removes edges pointing to intervened variables)
        intervened_model = self.model.copy()
        
        for var in intervention.keys():
            # Remove all edges pointing to the intervention variable
            parents = list(intervened_model.get_parents(var))
            for parent in parents:
                intervened_model.remove_edge(parent, var)
        
        # Perform inference on the intervened model
        intervened_inference = VariableElimination(intervened_model)
        
        # Combine intervention and evidence
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
        """Gets parent nodes (direct causes) of a node."""
        if self.model is None:
            raise ValueError("Model not trained")
        return list(self.model.get_parents(node))
    
    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Gets the Markov Blanket (parents + children + children's other parents).
        These are all variables directly relevant to the target variable.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        mb = set()
        
        # Parents
        mb.update(self.model.get_parents(node))
        
        # Children
        children = self.model.get_children(node)
        mb.update(children)
        
        # Other parents of the children
        for child in children:
            mb.update(self.model.get_parents(child))
        
        # Discard self
        mb.discard(node)
        
        return mb
    
    def get_causal_paths(
        self,
        source: str,
        target: str
    ) -> List[List[str]]:
        """
        Gets all directed paths (causal paths) from source to target.
        
        Parameters:
        - source: Source node.
        - target: Target node.
        
        Returns:
            List of paths, where each path is a node sequence.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Convert to networkx DiGraph
        G = nx.DiGraph(self.model.edges())
        
        try:
            paths = list(nx.all_simple_paths(G, source, target))
            logger.info(f"Found {len(paths)} causal paths from {source} to {target}")
            return paths
        except nx.NetworkXNoPath:
            logger.warning(f"No causal path from {source} to {target}")
            return []
    
    def save_model(self, filepath: str):
        """Saves model structure and parameters."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save in BIF format (Bayesian Interchange Format)
        from pgmpy.readwrite import BIFWriter
        writer = BIFWriter(self.model)
        writer.write_bif(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Loads a model."""
        from pgmpy.readwrite import BIFReader
        reader = BIFReader(filepath)
        self.model = reader.get_model()
        
        logger.info(f"Model loaded from {filepath}")
    
    def visualize_structure(self, output_path: Optional[str] = None):
        """
        Visualizes the network structure.
        
        Parameters:
        - output_path: Output image path (displays if None).
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
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock discretized data
    np.random.seed(42)
    data = pd.DataFrame({
        'Temperature': np.random.choice(['Low', 'Medium', 'High'], 200),
        'Humidity': np.random.choice(['Low', 'Medium', 'High'], 200),
        'WindSpeed': np.random.choice(['Low', 'Medium', 'High'], 200),
        'EDP_State': np.random.choice(['Lower', 'Normal', 'Peak'], 200)
    })
    
    # Domain knowledge: Temperature, Humidity, and WindSpeed may influence EDP
    domain_edges = [
        ('Temperature', 'EDP_State'),
        ('Humidity', 'EDP_State'),
        ('WindSpeed', 'EDP_State')
    ]
    
    # Create Bayesian Network
    bn = CausalBayesianNetwork(domain_edges=domain_edges)
    
    # Learn structure
    model = bn.learn_structure(data)
    print(f"\nLearned structure edges: {model.edges()}")
    
    # Learn parameters
    bn.learn_parameters(data, estimator='mle')
    
    # Inference example
    bn.create_inference_engine()
    
    # Query: P(EDP_State | Temperature=High)
    result = bn.query(['EDP_State'], evidence={'Temperature': 'High'})
    print(f"\nP(EDP_State | Temperature=High):")
    print(result)
    
    # Do-calculus: P(EDP_State | do(Temperature=Low))
    result_do = bn.do_calculus(
        intervention={'Temperature': 'Low'},
        query_var='EDP_State'
    )
    print(f"\nP(EDP_State | do(Temperature=Low)):")
    print(result_do)
    
    # Get Markov Blanket of EDP
    mb = bn.get_markov_blanket('EDP_State')
    print(f"\nMarkov Blanket of EDP_State: {mb}")