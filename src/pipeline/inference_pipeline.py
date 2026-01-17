"""
Inference Pipeline
Single sample prediction + Causal explanation + Optimization recommendation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import joblib

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Inference Pipeline
    
    Functions:
    1. Load all trained models and components.
    2. Perform prediction on a single sample (sequence).
    3. Extract DLP (Deep Learning Parameters) and perform clustering.
    4. Conduct causal inference analysis.
    5. Generate optimization recommendations.
    
    Parameters:
    - models_dir: Path to the directory containing trained models.
    """
    
    def __init__(self, models_dir: str = './outputs/models'):
        self.models_dir = models_dir
        
        # Model components
        self.preprocessor = None
        self.predictor = None
        self.state_classifier = None
        self.discretizer = None
        self.cam_clusterer = None
        self.attention_clusterer = None
        self.bayesian_network = None
        self.causal_inference = None
        self.recommendation_engine = None
        
        # Load models
        self.load_models()
        
        logger.info("InferencePipeline initialized")
    
    def load_models(self):
        """Loads all model components"""
        logger.info(f"Loading models from {self.models_dir}...")
        
        # Load preprocessor
        self.preprocessor = joblib.load(os.path.join(self.models_dir, 'preprocessor.pkl'))
        
        # Load prediction model
        from tensorflow import keras
        from src.models.predictor import AttentionLayer
        
        # Register custom layer and load model
        model_path = os.path.join(self.models_dir, 'predictor.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.models_dir, 'predictor.h5')
        
        with keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
            self.predictor_model = keras.models.load_model(model_path)
        
        # Models for CAM and Attention extraction
        self.cam_model = None
        self.attention_model = None
        
        # Load state classifier
        self.state_classifier = joblib.load(os.path.join(self.models_dir, 'state_classifier.pkl'))
        
        # Load discretizer
        self.discretizer = joblib.load(os.path.join(self.models_dir, 'discretizer.pkl'))
        
        # Load clusterers
        self.cam_clusterer = joblib.load(os.path.join(self.models_dir, 'cam_clusterer.pkl'))
        self.attention_clusterer = joblib.load(os.path.join(self.models_dir, 'attention_clusterer.pkl'))
        
        # Load Bayesian Network
        from src.models.bayesian_net import CausalBayesianNetwork
        self.bayesian_network = CausalBayesianNetwork()
        self.bayesian_network.load_model(os.path.join(self.models_dir, 'bayesian_network.bif'))
        
        # Create Causal Inference and Recommendation Engine
        from src.inference.causal_inference import CausalInference
        from src.inference.recommendation import RecommendationEngine
        
        self.causal_inference = CausalInference(self.bayesian_network, target_var='EDP_State')
        self.recommendation_engine = RecommendationEngine(self.causal_inference)
        
        logger.info("All models loaded successfully")
    
    def _extract_cam(self, X):
        """Extracts CAM features"""
        from tensorflow import keras
        import numpy as np
        if self.cam_model is None:
            cnn_conv2_output = self.predictor_model.get_layer('cnn_conv2').output
            self.cam_model = keras.Model(
                inputs=self.predictor_model.input,
                outputs=cnn_conv2_output
            )
        cam_output = self.cam_model.predict(X, verbose=0)
        return np.mean(cam_output, axis=-1)
    
    def _extract_attention(self, X):
        """Extracts Attention weights"""
        from tensorflow import keras
        if self.attention_model is None:
            attention_layer = self.predictor_model.get_layer('attention')
            lstm_output = self.predictor_model.get_layer('lstm').output
            _, attention_weights = attention_layer(lstm_output)
            self.attention_model = keras.Model(
                inputs=self.predictor_model.input,
                outputs=attention_weights
            )
        return self.attention_model.predict(X, verbose=0)
    
    def predict(
        self,
        data: pd.DataFrame,
        generate_recommendations: bool = True
    ) -> Dict:
        """
        Performs prediction and explanation for single or multiple samples
        
        Parameters:
        - data: Input data (must contain feature columns, at least sequence_length rows).
        - generate_recommendations: Whether to generate optimization recommendations.
        
        Returns:
        - Dictionary of prediction results.
        """
        logger.info("Starting inference...")
        
        results = {}
        
        # Step 1: Preprocessing
        X, _ = self.preprocessor.transform(data)
        results['n_samples'] = len(X)
        
        # Step 2: Prediction
        predictions = self.predictor_model.predict(X, verbose=0)
        results['predictions'] = predictions.flatten()
        
        # Step 3: Extract DLP
        cam_features = self._extract_cam(X)
        attention_features = self._extract_attention(X)
        
        # Step 4: DLP Clustering
        cam_clusters = self.cam_clusterer.predict(cam_features)
        attention_clusters = self.attention_clusterer.predict(attention_features)
        attention_types = [
            self.attention_clusterer.cluster_names_[c] for c in attention_clusters
        ]
        
        results['cam_clusters'] = cam_clusters
        results['attention_types'] = attention_types
        
        # Step 5: State Classification
        edp_states = self.state_classifier.predict(predictions)
        results['edp_states'] = edp_states
        
        # Step 6: Feature Discretization
        features_to_discretize = self.preprocessor.feature_cols
        discrete_data = self.discretizer.transform(
            data[features_to_discretize].iloc[:len(predictions)]
        )
        
        # Convert to DataFrame
        discrete_features = pd.DataFrame(
            discrete_data,
            columns=features_to_discretize
        )
        
        # Build current state dictionary (using the last sample)
        current_state = discrete_features.iloc[-1].to_dict()
        current_state['EDP_State'] = edp_states[-1]
        current_state['CAM_Cluster'] = str(cam_clusters[-1])
        current_state['Attention_Type'] = attention_types[-1]
        
        results['current_state'] = current_state
        
        # Step 7: Causal Inference (if recommendations are requested)
        if generate_recommendations:
            # Sensitivity Analysis
            features_to_analyze = list(discrete_features.columns)
            sensitivity = self.causal_inference.sensitivity_analysis(
                features_to_analyze,
                target_state='Peak'
            )
            results['sensitivity'] = sensitivity
            
            # Generate Recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                current_state,
                target_state='Peak',
                top_k=5
            )
            results['recommendations'] = recommendations
            
            # Generate Report
            report = self.recommendation_engine.format_report(
                recommendations,
                current_state,
                prediction=predictions[-1]
            )
            results['report'] = report
        
        logger.info(f"Inference completed for {len(X)} samples")
        
        return results
    
    def predict_single(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> str:
        """
        Predicts for a single sample and generates a full report
        
        Parameters:
        - data: Input data (at least sequence_length rows).
        - verbose: Whether to print detailed info.
        
        Returns:
        - Formatted report text.
        """
        results = self.predict(data, generate_recommendations=True)
        
        # Extract key info
        prediction = results['predictions'][-1]
        edp_state = results['edp_states'][-1]
        cam_cluster = results['cam_clusters'][-1]
        attention_type = results['attention_types'][-1]
        
        # Print summary
        if verbose:
            print("\n" + "="*60)
            print("           Prediction Result")
            print("="*60)
            print(f"Predicted Load: {prediction:.2f} kWh")
            print(f"Load State: {edp_state}")
            print(f"CAM Cluster: {cam_cluster}")
            print(f"Attention Type: {attention_type}")
            print("\n")
        
        # Return full report
        return results['report']
    
    def batch_predict(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Batch prediction (no recommendations, only predictions)
        
        Parameters:
        - data: Input data.
        - output_path: Path for output CSV (optional).
        
        Returns:
        - DataFrame containing prediction results.
        """
        results = self.predict(data, generate_recommendations=False)
        
        # Build results DataFrame
        df_results = pd.DataFrame({
            'Prediction': results['predictions'],
            'EDP_State': results['edp_states'],
            'CAM_Cluster': results['cam_clusters'],
            'Attention_Type': results['attention_types']
        })
        
        if output_path:
            df_results.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved to {output_path}")
        
        return df_results
    
    def explain_prediction(
        self,
        sample_idx: int,
        data: pd.DataFrame
    ) -> str:
        """
        Explains a single prediction result
        
        Parameters:
        - sample_idx: Index of the sample.
        - data: Complete data.
        
        Returns:
        - Explanation text.
        """
        results = self.predict(data, generate_recommendations=True)
        
        # Extract sample info
        prediction = results['predictions'][sample_idx]
        state = results['edp_states'][sample_idx]
        current_state = results['current_state']
        
        # Generate explanation
        explanation = self.causal_inference.generate_explanation(
            current_state,
            target_state=state,
            top_k=3
        )
        
        full_explanation = (
            f"Explanation for Sample #{sample_idx}\n"
            f"{'='*60}\n"
            f"Predicted Load: {prediction:.2f} kWh\n"
            f"Load State: {state}\n\n"
            f"{explanation}"
        )
        
        return full_explanation


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Note: Training pipeline must be run first to generate model files.
    # This is a demonstration of the API usage.
    
    try:
        # Create inference pipeline
        pipeline = InferencePipeline(models_dir='./outputs/models')
        
        # Simulate new data
        np.random.seed(42)
        new_data = pd.DataFrame({
            'Temperature': np.random.randn(25) * 10 + 20,
            'Humidity': np.random.randn(25) * 15 + 60,
            'WindSpeed': np.random.randn(25) * 5 + 10
        })
        
        # Single sample prediction
        report = pipeline.predict_single(new_data, verbose=True)
        print(report)
        
    except FileNotFoundError:
        print("Model files not found. Please run the training pipeline first.")