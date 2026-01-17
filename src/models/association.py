"""
Association Rule Mining Module
Uses the Apriori algorithm to mine frequent itemsets and association rules, 
providing candidate edges for the Bayesian Network.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from mlxtend.frequent_patterns import apriori, association_rules
import logging

logger = logging.getLogger(__name__)


class AssociationRuleMiner:
    """
    Association Rule Miner
    
    Functions:
    1. Encodes discretized continuous features into transaction format.
    2. Mines frequent itemsets using the Apriori algorithm.
    3. Generates and filters association rules related to EDP.
    4. Extracts candidate causal edges.
    
    Parameters:
    - min_support: Minimum support threshold (default 0.05).
    - min_confidence: Minimum confidence threshold (default 0.6).
    - min_lift: Minimum lift threshold (default 1.2).
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
        Prepares data: Converts discretized data into One-hot encoding.
        
        Parameters:
        - df: Discretized DataFrame (feature values are category labels).
        - edp_col: Name of the EDP state column.
        
        Returns:
        - Boolean DataFrame suitable for the Apriori algorithm.
        """
        # Validate that the EDP column exists
        if edp_col not in df.columns:
            raise ValueError(f"EDP column '{edp_col}' not found in DataFrame")
        
        # One-hot encode all columns
        df_encoded = pd.get_dummies(df, prefix_sep='=')
        
        # Convert to boolean type
        df_encoded = df_encoded.astype(bool)
        
        logger.info(
            f"Prepared data: {len(df)} transactions, {len(df_encoded.columns)} items"
        )
        
        return df_encoded
    
    def mine_frequent_itemsets(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """
        Mines frequent itemsets.
        
        Parameters:
        - df_encoded: One-hot encoded boolean DataFrame.
        
        Returns:
        - DataFrame of frequent itemsets (includes 'support' column).
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
        Generates association rules.
        
        Parameters:
        - metric: Evaluation metric ('confidence', 'lift', 'leverage', 'conviction').
        - min_threshold: Minimum threshold (if None, uses threshold from class initialization).
        
        Returns:
        - DataFrame of association rules.
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
        
        # Filter by lift
        if metric != 'lift':
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        logger.info(f"Generated {len(self.rules)} association rules")
        
        return self.rules
    
    def filter_edp_rules(self, edp_prefix: str = 'EDP_State=') -> pd.DataFrame:
        """
        Filters rules related to EDP (where EDP is in the consequent).
        
        Parameters:
        - edp_prefix: Prefix of the EDP column (after One-hot encoding).
        
        Returns:
        - Association rules containing EDP.
        """
        if self.rules is None:
            raise ValueError("Must call generate_rules() first")
        
        def contains_edp(consequents):
            """Checks if the consequent contains EDP"""
            return any(edp_prefix in str(item) for item in consequents)
        
        self.edp_rules = self.rules[
            self.rules['consequents'].apply(contains_edp)
        ].copy()
        
        # Sort by confidence in descending order
        self.edp_rules = self.edp_rules.sort_values(
            'confidence',
            ascending=False
        ).reset_index(drop=True)
        
        logger.info(f"Filtered to {len(self.edp_rules)} EDP-related rules")
        
        return self.edp_rules
    
    def rules_to_constraints(self, top_k: int = 50) -> List[Tuple[str, str]]:
        """
        Converts association rules into candidate edge constraints for the Bayesian Network.
        
        Parameters:
        - top_k: Extract from the top k rules.
        
        Returns:
        - A list of [(antecedent_feature, consequent_feature), ...].
        """
        if self.edp_rules is None:
            raise ValueError("Must call filter_edp_rules() first")
        
        edges = []
        edp_rules_subset = self.edp_rules.head(top_k)
        
        for _, rule in edp_rules_subset.iterrows():
            # Extract antecedent feature names (remove discretization labels)
            antecedents = [item.split('=')[0] for item in rule['antecedents']]
            
            # Extract consequent feature names
            consequents = [item.split('=')[0] for item in rule['consequents']]
            
            # Create edges (antecedent -> consequent)
            for ant in antecedents:
                for cons in consequents:
                    if ant != cons:  # Avoid self-loops
                        edges.append((ant, cons))
        
        # Remove duplicates
        self.candidate_edges = list(set(edges))
        
        logger.info(f"Extracted {len(self.candidate_edges)} candidate edges from top {top_k} rules")
        
        return self.candidate_edges
    
    def get_rule_summary(self, n: int = 10) -> pd.DataFrame:
        """
        Gets a summary of the rules (for visualization and reporting).
        
        Parameters:
        - n: Returns the top n rules.
        
        Returns:
        - A concise DataFrame of rules.
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
        
        # Format for display
        summary['antecedents'] = summary['antecedents'].apply(lambda x: ', '.join(x))
        summary['consequents'] = summary['consequents'].apply(lambda x: ', '.join(x))
        
        return summary
    
    def save_rules(self, filepath: str):
        """Saves rules to a CSV file"""
        if self.edp_rules is None:
            raise ValueError("No rules to save")
        
        # Convert frozenset to string for saving
        df_save = self.edp_rules.copy()
        df_save['antecedents'] = df_save['antecedents'].apply(lambda x: ', '.join(x))
        df_save['consequents'] = df_save['consequents'].apply(lambda x: ', '.join(x))
        
        df_save.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df_save)} rules to {filepath}")
    
    def load_rules(self, filepath: str):
        """Loads rules from a CSV file"""
        df_load = pd.read_csv(filepath)
        
        # Convert string back to frozenset
        df_load['antecedents'] = df_load['antecedents'].apply(
            lambda x: frozenset(x.split(', '))
        )
        df_load['consequents'] = df_load['consequents'].apply(
            lambda x: frozenset(x.split(', '))
        )
        
        self.edp_rules = df_load
        logger.info(f"Loaded {len(df_load)} rules from {filepath}")


if __name__ == "__main__":
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock discretized data
    np.random.seed(42)
    df_discrete = pd.DataFrame({
        'Temperature': np.random.choice(['Low', 'Medium', 'High'], 100),
        'Humidity': np.random.choice(['Low', 'Medium', 'High'], 100),
        'WindSpeed': np.random.choice(['Low', 'Medium', 'High'], 100),
        'EDP_State': np.random.choice(['Lower', 'Normal', 'Peak'], 100, p=[0.2, 0.5, 0.3])
    })
    
    # Create miner
    miner = AssociationRuleMiner(min_support=0.05, min_confidence=0.6, min_lift=1.2)
    
    # Data preparation
    df_encoded = miner.prepare_data(df_discrete, edp_col='EDP_State')
    print(f"\nEncoded data shape: {df_encoded.shape}")
    
    # Mine frequent itemsets
    frequent = miner.mine_frequent_itemsets(df_encoded)
    print(f"\nTop 5 frequent itemsets:")
    print(frequent.head())
    
    # Generate rules
    rules = miner.generate_rules()
    print(f"\nGenerated {len(rules)} rules")
    
    # Filter EDP rules
    edp_rules = miner.filter_edp_rules()
    print(f"\nTop 5 EDP rules:")
    print(miner.get_rule_summary(5))
    
    # Extract candidate edges
    edges = miner.rules_to_constraints(top_k=10)
    print(f"\nCandidate edges for Bayesian Network:")
    for edge in edges[:10]:
        print(f"  {edge[0]} -> {edge[1]}")