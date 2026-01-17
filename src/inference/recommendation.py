"""
Recommendation Generation Module
Generates actionable energy-saving suggestions based on causal inference results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Recommendation Engine
    
    Functions:
    1. Identify key influence factors based on causal inference.
    2. Generate actionable energy-saving suggestions.
    3. Quantify the expected effect of each suggestion.
    4. Support multi-template and context-aware natural language generation.
    
    Parameters:
    - causal_inference: Trained CausalInference object.
    - feature_mapping: Mapping from features to human-readable names.
    - action_templates: Templates for action suggestions.
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
        """Default feature mapping (Readable Names)"""
        return {
            'Temperature': 'Temperature',
            'Humidity': 'Humidity',
            'WindSpeed': 'Wind Speed',
            'Hour': 'Time Period',
            'DayOfWeek': 'Day of Week',
            'Month': 'Month',
            'Season': 'Season',
            'IsWeekend': 'Weekend Flag',
            'EDP_State': 'Power Load'
        }
    
    def _default_templates(self) -> Dict[str, Dict[str, str]]:
        """Default action suggestion templates"""
        return {
            'Temperature': {
                'Low': 'Suggest lowering the indoor temperature setting, e.g., adjusting AC to around {value}℃',
                'Medium': 'Suggest maintaining a moderate temperature, avoid excessive cooling or heating',
                'High': 'Suggest increasing the indoor temperature setting to reduce AC energy consumption',
                'VeryHigh': 'Suggest significantly increasing temperature settings or consider turning off some units'
            },
            'Humidity': {
                'Low': 'Suggest reducing dehumidifier usage, maintaining humidity around {value}%',
                'Medium': 'Suggest maintaining the current humidity level',
                'High': 'Suggest increasing dehumidifier power while being mindful of energy balance',
                'VeryHigh': 'Suggest enabling dehumidification mode or using natural ventilation'
            },
            'WindSpeed': {
                'Low': 'Suggest reducing ventilation equipment power',
                'Medium': 'Suggest maintaining current ventilation levels',
                'High': 'Suggest utilizing natural wind to reduce mechanical ventilation',
                'VeryHigh': 'Suggest making full use of natural ventilation and turning off fans, etc.'
            },
            'Hour': {
                'Peak': 'Current time is a peak demand period; suggest shifting high-energy tasks to off-peak',
                'Normal': 'Suggest arranging electricity use reasonably to avoid concentrated demand',
                'Lower': 'Current time is an off-peak period; high-energy equipment can be used appropriately'
            },
            'Season': {
                'Winter': 'In Winter, suggest optimizing heating strategies using timed and zone controls',
                'Spring': 'In Spring, suggest reducing AC use and utilizing natural ventilation',
                'Summer': 'In Summer, suggest optimizing cooling strategies to avoid over-cooling',
                'Autumn': 'In Autumn, suggest reducing AC use and utilizing the natural environment'
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
        Generates energy-saving recommendations.
        
        Parameters:
        - current_state: Current state (feature values).
        - target_state: Target state (usually 'Peak', indicating high load).
        - sensitivity_threshold: Threshold for sensitivity (only recommend high-impact features).
        - top_k: Return the top k suggestions.
        
        Returns:
        - List of suggestions, each containing:
          {
            'rank': Rank,
            'feature': Feature name,
            'feature_name': Readable feature name,
            'current_value': Current value,
            'recommended_value': Recommended value,
            'current_prob': Current probability,
            'expected_prob': Expected probability,
            'potential_reduction': Potential reduction magnitude,
            'action': Action suggestion text
          }
        """
        # Ensure sensitivity analysis has been run
        if self.ci.sensitivity_results is None:
            features = list(current_state.keys())
            self.ci.sensitivity_analysis(features, target_state=target_state)
        
        # Get critical factors
        critical_factors = self.ci.identify_critical_factors(
            threshold=sensitivity_threshold,
            target_state=target_state
        )
        
        # Generate suggestions
        recommendations = []
        for idx, factor in enumerate(critical_factors[:top_k]):
            feature = factor['Feature']
            
            # Get readable name
            feature_name = self.feature_mapping.get(feature, feature)
            
            # Current and recommended values
            current_value = current_state.get(feature, 'Unknown')
            recommended_value = factor['Recommended_Value']
            
            # Probability information
            current_prob = factor['Current_Max_Prob']
            expected_prob = factor['Expected_Prob']
            potential_reduction = factor['Potential_Reduction']
            
            # Generate action text
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
        Generates the action suggestion text.
        
        Parameters:
        - feature: Feature name.
        - current_value: Current value.
        - recommended_value: Recommended value.
        - impact: Magnitude of impact.
        
        Returns:
        - Natural language recommendation.
        """
        # Get template
        if feature in self.action_templates:
            template = self.action_templates[feature].get(
                recommended_value,
                f"Suggest adjusting {feature} to {recommended_value}"
            )
            action = template.format(value=recommended_value)
        else:
            # Generic template
            action = f"Suggest adjusting {self.feature_mapping.get(feature, feature)} from '{current_value}' to '{recommended_value}'"
        
        # Add impact magnitude description
        if impact > 0.3:
            magnitude = "significantly"
        elif impact > 0.15:
            magnitude = "noticeably"
        else:
            magnitude = "to some extent"
        
        action += f", which is expected to {magnitude} reduce the probability of peak load (by approx. {impact:.1%})"
        
        return action
    
    def format_report(
        self,
        recommendations: List[Dict],
        current_state: Dict[str, str],
        prediction: Optional[float] = None
    ) -> str:
        """
        Generates a formatted recommendation report.
        
        Parameters:
        - recommendations: List of suggestions.
        - current_state: Current state.
        - prediction: Predicted EDP value (optional).
        
        Returns:
        - Formatted report text.
        """
        report = "=" * 60 + "\n"
        report += "          Energy Consumption Prediction & Optimization Report\n"
        report += "=" * 60 + "\n\n"
        
        # Current State
        report += "[Current State]\n"
        for feature, value in current_state.items():
            feature_name = self.feature_mapping.get(feature, feature)
            report += f"  {feature_name}: {value}\n"
        
        # Predicted Value
        if prediction is not None:
            if hasattr(prediction, 'item'):
                pred_value = prediction.item()
            elif hasattr(prediction, '__getitem__'):
                pred_value = float(prediction[0])
            else:
                pred_value = float(prediction)
            report += f"\n[Predicted Load]\n  {pred_value:.2f} kWh\n"
        
        # Suggestions
        if recommendations:
            report += f"\n[Optimization Suggestions] ({len(recommendations)} total)\n\n"
            for rec in recommendations:
                report += f"{rec['rank']}. {rec['action']}\n"
                report += f"   Current: {rec['feature_name']}={rec['current_value']} "
                report += f"→ Recommended: {rec['recommended_value']}\n"
                report += f"   Expected Effect: Peak probability drops from {rec['current_prob']:.1%} to {rec['expected_prob']:.1%}\n\n"
        else:
            report += "\n[Optimization Suggestions]\n  Current state is optimal, no adjustment needed.\n"
        
        report += "=" * 60 + "\n"
        report += "Note: Suggestions are based on Causal Bayesian Network inference for reference.\n"
        
        return report
    
    def prioritize_recommendations(
        self,
        recommendations: List[Dict],
        cost_mapping: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Prioritizes suggestions based on cost-benefit ratio.
        
        Parameters:
        - recommendations: Original suggestion list.
        - cost_mapping: Mapping of adjustment costs per feature {feature: cost}.
        
        Returns:
        - Re-ordered suggestion list with added 'cost_benefit_ratio' field.
        """
        if cost_mapping is None:
            # Default costs (adjust based on reality)
            cost_mapping = {
                'Temperature': 0.5,  # Low adjustment cost
                'Humidity': 0.7,
                'WindSpeed': 0.3,
                'Hour': 1.0,  # Time cannot be adjusted, effectively high cost
                'Season': 1.0
            }
        
        # Calculate cost-benefit ratio
        for rec in recommendations:
            cost = cost_mapping.get(rec['feature'], 0.5)
            benefit = rec['potential_reduction']
            rec['cost'] = cost
            rec['benefit'] = benefit
            rec['cost_benefit_ratio'] = benefit / cost if cost > 0 else 0
        
        # Sort by cost-benefit ratio descending
        prioritized = sorted(
            recommendations,
            key=lambda x: x['cost_benefit_ratio'],
            reverse=True
        )
        
        # Update ranks
        for idx, rec in enumerate(prioritized):
            rec['rank'] = idx + 1
        
        logger.info("Recommendations prioritized by cost-benefit ratio")
        
        return prioritized
    
    def save_recommendations(
        self,
        recommendations: List[Dict],
        filepath: str
    ):
        """Saves recommendations to a CSV file"""
        df = pd.DataFrame(recommendations)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {len(recommendations)} recommendations to {filepath}")


if __name__ == "__main__":
    # Example (Requires a trained causal inference model)
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    from src.models.bayesian_net import CausalBayesianNetwork
    from src.inference.causal_inference import CausalInference
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
    
    bn = CausalBayesianNetwork(domain_edges=[
        ('Temperature', 'EDP_State'),
        ('Humidity', 'EDP_State')
    ])
    bn.learn_structure(data)
    bn.learn_parameters(data)
    
    ci = CausalInference(bn, target_var='EDP_State')
    
    # Create engine
    engine = RecommendationEngine(ci)
    
    # Current State
    current_state = {
        'Temperature': 'High',
        'Humidity': 'Medium',
        'WindSpeed': 'Low'
    }
    
    # Generate suggestions
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
    
    # Generate report
    report = engine.format_report(recommendations, current_state, prediction=125.5)
    print("\n" + report)