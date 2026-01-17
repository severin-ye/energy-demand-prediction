"""
Causal Inference Module
Provides causal analysis tools based on Bayesian Networks, including sensitivity analysis and counterfactual inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CausalInference:
    """
    Causal Inference Tool
    
    Functions:
    1. Sensitivity Analysis: Analyze the impact of feature changes on EDP.
    2. Tornado Chart Visualization: Display feature importance.
    3. Counterfactual Inference: Calculate "What if X=x".
    4. Causal Effect Quantification: Calculate Average Causal Effect (ACE).
    
    Parameters:
    - bayesian_network: Trained CausalBayesianNetwork object.
    - target_var: Target variable (typically 'EDP_State').
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
        Sensitivity Analysis: Calculates the impact of each feature change on the target state probability.
        
        Parameters:
        - features: List of features to analyze.
        - baseline_evidence: Baseline evidence (fixed values for other features).
        - target_state: Target state (e.g., 'Peak').
        
        Returns:
        - DataFrame containing sensitivity metrics for each feature.
        """
        results = []
        
        for feature in features:
            # Get all possible values for this feature
            feature_values = self._get_feature_values(feature)
            
            # Calculate target probability for each value
            probs = []
            for value in feature_values:
                evidence = {feature: value}
                if baseline_evidence:
                    evidence.update(baseline_evidence)
                
                # Query P(target_var | evidence)
                result = self.bn.query([self.target_var], evidence=evidence)
                prob = self._extract_prob(result, target_state)
                probs.append(prob)
            
            # Calculate sensitivity metrics
            prob_range = max(probs) - min(probs)
            prob_std = np.std(probs)
            prob_max = max(probs)
            max_value = feature_values[np.argmax(probs)]
            
            results.append({
                'Feature': feature,
                'Prob_Range': prob_range,  # Range of probability change
                'Prob_Std': prob_std,      # Probability standard deviation
                'Max_Prob': prob_max,      # Maximum probability
                'Max_Value': max_value,    # Value leading to maximum probability
                'Values': feature_values,
                'Probs': probs
            })
        
        self.sensitivity_results = pd.DataFrame(results)
        
        # Sort by probability range in descending order
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
        Plot Tornado Chart to display feature importance.
        
        Parameters:
        - top_k: Number of top important features to display.
        - output_path: Output path (displays if None).
        - figsize: Image size.
        """
        if self.sensitivity_results is None:
            raise ValueError("Must run sensitivity_analysis() first")
        
        # Take the top k features
        df_plot = self.sensitivity_results.head(top_k).copy()
        
        # Create chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(df_plot))
        ax.barh(y_pos, df_plot['Prob_Range'], align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['Feature'])
        ax.invert_yaxis()  # Most important at the top
        ax.set_xlabel('Probability Range', fontsize=12)
        ax.set_title(f'Tornado Chart: Feature Sensitivity on {self.target_var}', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # Add numeric labels
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
        Counterfactual Inference: What would happen if certain variables took different values.
        
        Parameters:
        - actual_evidence: Actually observed evidence.
        - intervention: Counterfactual intervention.
        - target_state: Target state.
        
        Returns:
        - {
            'actual_prob': Actual probability,
            'counterfactual_prob': Counterfactual probability,
            'causal_effect': Causal effect,
            'interpretation': Verbal explanation
          }
        """
        # Probability under actual conditions
        actual_result = self.bn.query([self.target_var], evidence=actual_evidence)
        actual_prob = self._extract_prob(actual_result, target_state)
        
        # Probability under counterfactual conditions (using do-calculus)
        counterfactual_result = self.bn.do_calculus(
            intervention=intervention,
            query_var=self.target_var,
            evidence=None  # do-calculus doesn't strictly require other evidence
        )
        counterfactual_prob = self._extract_prob(counterfactual_result, target_state)
        
        # Causal Effect
        causal_effect = counterfactual_prob - actual_prob
        
        # Generate interpretation
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
        Calculate Average Causal Effect (ACE): E[Y | do(X=x1)] - E[Y | do(X=x0)]
        
        Parameters:
        - treatment_var: Intervention variable.
        - treatment_value: Intervention value (treatment group).
        - control_value: Control value (control group).
        - target_state: Target state.
        - data: Dataset (used for computing expectations).
        
        Returns:
        - ACE value.
        """
        # Treatment group probability
        treatment_result = self.bn.do_calculus(
            intervention={treatment_var: treatment_value},
            query_var=self.target_var
        )
        treatment_prob = self._extract_prob(treatment_result, target_state)
        
        # Control group probability
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
        Identify critical factors (features in sensitivity results whose probability range exceeds the threshold).
        
        Parameters:
        - threshold: Probability range threshold.
        - target_state: Target state.
        
        Returns:
        - List of critical factors (including recommendations).
        """
        if self.sensitivity_results is None:
            raise ValueError("Must run sensitivity_analysis() first")
        
        critical = self.sensitivity_results[
            self.sensitivity_results['Prob_Range'] >= threshold
        ].copy()
        
        recommendations = []
        for _, row in critical.iterrows():
            # Find optimal value (minimizes target probability)
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
        """Get all possible values of a feature."""
        if self.bn.model is None:
            raise ValueError("Bayesian network not trained")
        
        # Get values from CPD
        cpd = self.bn.model.get_cpds(feature)
        return list(cpd.state_names[feature])
    
    def _extract_prob(self, query_result, state: str) -> float:
        """Extract the probability of a specific state from query results."""
        # pgmpy query result is a DiscreteFactor object
        values = query_result.values
        states = query_result.state_names[self.target_var]
        
        # Find index of target state
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
        Generate natural language explanation.
        
        Parameters:
        - current_state: Current state (feature values).
        - target_state: Target state.
        - top_k: Explain the top k important factors.
        
        Returns:
        - Natural language explanation text.
        """
        if self.sensitivity_results is None:
            raise ValueError("Must run sensitivity_analysis() first")
        
        # Get top k important factors
        top_features = self.sensitivity_results.head(top_k)
        
        explanation = f"Analysis of {self.target_var}={target_state}:\n\n"
        
        for idx, row in top_features.iterrows():
            feature = row['Feature']
            impact = row['Prob_Range']
            
            # Probability corresponding to current value
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
    # Example (requires a trained Bayesian Network)
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    from src.models.bayesian_net import CausalBayesianNetwork
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Mock data and train network
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
    
    # Create Causal Inference Tool
    ci = CausalInference(bn, target_var='EDP_State')
    
    # Sensitivity Analysis
    features = ['Temperature', 'Humidity', 'WindSpeed']
    sensitivity = ci.sensitivity_analysis(features, target_state='Peak')
    print("\nSensitivity Analysis:")
    print(sensitivity[['Feature', 'Prob_Range', 'Max_Value', 'Max_Prob']])
    
    # Counterfactual Analysis
    counterfactual = ci.counterfactual_analysis(
        actual_evidence={'Temperature': 'High', 'Humidity': 'Medium'},
        intervention={'Temperature': 'Low'},
        target_state='Peak'
    )
    print(f"\nCounterfactual Analysis:")
    print(counterfactual['interpretation'])
    
    # Average Causal Effect
    ace = ci.average_causal_effect(
        treatment_var='Temperature',
        treatment_value='High',
        control_value='Low',
        target_state='Peak'
    )
    print(f"\nACE: {ace:.3f}")