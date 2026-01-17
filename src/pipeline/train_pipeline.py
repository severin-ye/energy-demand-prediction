"""
Training Pipeline
End-to-end training flow, integrating all modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainPipeline:
    """
    Training Pipeline (9-step flow)
    
    Steps:
    1. Data Preprocessing (Sequence generation)
    2. Train Parallel CNN-LSTM-Attention model
    3. Extract CAM and Attention weights (DLP)
    4. DLP Clustering (CAM clustering + Attention clustering)
    5. Compute Sn robust scale estimation and perform state classification
    6. Feature Discretization (4-level quantization)
    7. Association Rule Mining (Apriori)
    8. Bayesian Network structure and parameter learning
    9. Save all models and results
    
    Parameters:
    - config: Configuration dictionary
    - output_dir: Output directory
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: str = './outputs'
    ):
        self.config = config or self._default_config()
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        # Model components
        self.preprocessor = None
        self.predictor = None
        self.state_classifier = None
        self.discretizer = None
        self.cam_clusterer = None
        self.attention_clusterer = None
        self.association_miner = None
        self.bayesian_network = None
        
        logger.info(f"TrainPipeline initialized with output_dir={output_dir}")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            # Data Preprocessing
            'sequence_length': 20,
            'feature_cols': ['Temperature', 'Humidity', 'WindSpeed'],
            'target_col': 'EDP',
            
            # Prediction Model
            'cnn_filters': [64, 32],
            'lstm_units': 64,
            'attention_units': 50,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            
            # State Classification
            'n_states': 3,
            'state_names': ['Lower', 'Normal', 'Peak'],
            
            # Discretization
            'n_bins': 4,
            'bin_labels': ['Low', 'Medium', 'High', 'VeryHigh'],
            
            # DLP Clustering
            'n_cam_clusters': 3,
            'n_attention_clusters': 3,
            
            # Association Rules
            'min_support': 0.05,
            'min_confidence': 0.6,
            'min_lift': 1.2,
            
            # Bayesian Network
            'bn_score_fn': 'bic',
            'bn_max_iter': 100,
            'bn_estimator': 'mle'
        }
    
    def run(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Executes the full training pipeline.
        
        Parameters:
        - train_data: Training data (including features and target)
        - val_data: Validation data (optional)
        
        Returns:
        - Dictionary of training results
        """
        logger.info("="*60)
        logger.info("Starting Training Pipeline")
        logger.info("="*60)
        
        results = {}
        
        # Step 1: Data Preprocessing
        logger.info("\n[Step 1/9] Data Preprocessing...")
        X_train, y_train, X_val, y_val = self._step1_preprocess(train_data, val_data)
        results['data_shapes'] = {
            'X_train': X_train.shape,
            'y_train': y_train.shape
        }
        
        # Step 2: Train prediction model
        logger.info("\n[Step 2/9] Training Parallel CNN-LSTM-Attention Model...")
        history = self._step2_train_predictor(X_train, y_train, X_val, y_val)
        results['training_history'] = {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', [])
        }
        
        # Step 3: Extract DLP
        logger.info("\n[Step 3/9] Extracting Deep Learning Parameters (CAM & Attention)...")
        cam_features, attention_features = self._step3_extract_dlp(X_train)
        results['dlp_shapes'] = {
            'cam': cam_features.shape,
            'attention': attention_features.shape
        }
        
        # Step 4: DLP Clustering
        logger.info("\n[Step 4/9] Clustering DLP Features...")
        cam_clusters, attention_types = self._step4_cluster_dlp(cam_features, attention_features)
        results['cluster_distributions'] = {
            'cam': np.bincount(cam_clusters),
            'attention': pd.Series(attention_types).value_counts().to_dict()
        }
        
        # Step 5: Sn State Classification
        logger.info("\n[Step 5/9] Sn State Classification...")
        edp_states = self._step5_classify_states(y_train)
        results['state_distribution'] = pd.Series(edp_states).value_counts().to_dict()
        
        # Step 6: Feature Discretization
        logger.info("\n[Step 6/9] Feature Discretization...")
        discrete_data = self._step6_discretize(train_data, edp_states, cam_clusters, attention_types)
        results['discrete_features'] = list(discrete_data.columns)
        
        # Step 7: Association Rule Mining
        logger.info("\n[Step 7/9] Association Rule Mining...")
        candidate_edges = self._step7_mine_rules(discrete_data)
        results['candidate_edges'] = candidate_edges
        
        # Step 8: Bayesian Network Learning
        logger.info("\n[Step 8/9] Bayesian Network Learning...")
        bn_edges = self._step8_learn_bayesian_network(discrete_data, candidate_edges)
        results['bn_edges'] = bn_edges
        
        # Step 9: Save models
        logger.info("\n[Step 9/9] Saving Models...")
        self._step9_save_models()
        
        logger.info("\n" + "="*60)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("="*60)
        
        return results
    
    def _step1_preprocess(self, train_data, val_data):
        """Step 1: Data Preprocessing"""
        from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
        
        self.preprocessor = EnergyDataPreprocessor(
            sequence_length=self.config['sequence_length'],
            feature_cols=self.config['feature_cols'],
            target_col=self.config['target_col']
        )
        
        X_train, y_train = self.preprocessor.fit_transform(train_data)
        
        if val_data is not None:
            X_val, y_val = self.preprocessor.transform(val_data)
        else:
            X_val, y_val = None, None
        
        logger.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
        if X_val is not None:
            logger.info(f"  Val: X={X_val.shape}, y={y_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def _step2_train_predictor(self, X_train, y_train, X_val, y_val):
        """Step 2: Train Prediction Model"""
        from src.models.predictor import ParallelCNNLSTMAttention
        
        self.predictor = ParallelCNNLSTMAttention(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            cnn_filters=self.config['cnn_filters'][0] if isinstance(self.config['cnn_filters'], list) else self.config['cnn_filters'],
            lstm_units=self.config['lstm_units'],
            attention_units=self.config['attention_units']
        )
        
        # Compile and Train
        self.predictor.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.predictor.model.fit(
            X_train,
            y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            verbose=1
        )
        
        logger.info(f"  Final train loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            logger.info(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
    def _step3_extract_dlp(self, X_train):
        """Step 3: Extract DLP (CAM and Attention weights)"""
        cam_features = self.predictor.extract_cam(X_train)
        attention_features = self.predictor.extract_attention_weights(X_train)
        
        logger.info(f"  CAM: {cam_features.shape}")
        logger.info(f"  Attention: {attention_features.shape}")
        
        return cam_features, attention_features
    
    def _step4_cluster_dlp(self, cam_features, attention_features):
        """Step 4: DLP Clustering"""
        from src.models.clustering import DLPClusterer, AttentionClusterer
        
        # CAM Clustering
        self.cam_clusterer = DLPClusterer(n_clusters=self.config['n_cam_clusters'])
        cam_clusters = self.cam_clusterer.fit_predict(cam_features)
        
        # Attention Clustering
        self.attention_clusterer = AttentionClusterer(n_clusters=self.config['n_attention_clusters'])
        attention_clusters = self.attention_clusterer.fit_predict(attention_features)
        attention_types = [self.attention_clusterer.cluster_names_[c] for c in attention_clusters]
        
        logger.info(f"  CAM clusters: {np.bincount(cam_clusters)}")
        logger.info(f"  Attention types: {pd.Series(attention_types).value_counts().to_dict()}")
        
        return cam_clusters, attention_types
    
    def _step5_classify_states(self, y_train):
        """Step 5: Sn State Classification"""
        from src.models.state_classifier import SnStateClassifier
        
        self.state_classifier = SnStateClassifier(
            n_states=self.config['n_states'],
            state_names=self.config['state_names']
        )
        
        edp_states = self.state_classifier.fit_predict(y_train)
        
        logger.info(f"  State distribution: {pd.Series(edp_states).value_counts().to_dict()}")
        
        return edp_states
    
    def _step6_discretize(self, train_data, edp_states, cam_clusters, attention_types):
        """Step 6: Feature Discretization"""
        from src.models.discretizer import QuantileDiscretizer
        
        # Discretize raw features
        self.discretizer = QuantileDiscretizer(
            n_bins=self.config['n_bins'],
            strategy='quantile'
        )
        
        features_to_discretize = self.config['feature_cols'].copy()
        # Ensure we take the same number of samples as edp_states
        n_samples = len(edp_states)
        discrete_data = self.discretizer.fit_transform(
            train_data[features_to_discretize].iloc[:n_samples]
        )
        
        # Convert to DataFrame if it's a numpy array
        if not isinstance(discrete_data, pd.DataFrame):
            discrete_data = pd.DataFrame(
                discrete_data,
                columns=features_to_discretize
            )
        
        # Add EDP State, CAM Cluster, and Attention Type
        discrete_data['EDP_State'] = edp_states
        discrete_data['CAM_Cluster'] = cam_clusters.astype(str)
        discrete_data['Attention_Type'] = attention_types
        
        logger.info(f"  Discretized features: {list(discrete_data.columns)}")
        
        return discrete_data
    
    def _step7_mine_rules(self, discrete_data):
        """Step 7: Association Rule Mining"""
        from src.models.association import AssociationRuleMiner
        
        self.association_miner = AssociationRuleMiner(
            min_support=self.config['min_support'],
            min_confidence=self.config['min_confidence'],
            min_lift=self.config['min_lift']
        )
        
        # Prepare data
        df_encoded = self.association_miner.prepare_data(discrete_data, edp_col='EDP_State')
        
        # Mine frequent itemsets
        self.association_miner.mine_frequent_itemsets(df_encoded)
        
        # Generate rules
        self.association_miner.generate_rules()
        
        # Filter EDP rules
        self.association_miner.filter_edp_rules()
        
        # Extract candidate edges
        candidate_edges = self.association_miner.rules_to_constraints(top_k=50)
        
        logger.info(f"  Found {len(candidate_edges)} candidate edges")
        
        # Save rules
        rules_path = os.path.join(self.output_dir, 'results', 'association_rules.csv')
        self.association_miner.save_rules(rules_path)
        
        return candidate_edges
    
    def _step8_learn_bayesian_network(self, discrete_data, candidate_edges):
        """Step 8: Bayesian Network Learning"""
        from src.models.bayesian_net import CausalBayesianNetwork
        
        # Domain knowledge constraints
        domain_edges = [
            ('Temperature', 'EDP_State'),
            ('Humidity', 'EDP_State'),
            ('CAM_Cluster', 'EDP_State'),
            ('Attention_Type', 'EDP_State')
        ]
        
        self.bayesian_network = CausalBayesianNetwork(
            domain_edges=domain_edges,
            score_fn=self.config['bn_score_fn']
        )
        
        # Structure Learning
        self.bayesian_network.learn_structure(
            discrete_data,
            candidate_edges=candidate_edges,
            max_iter=self.config['bn_max_iter']
        )
        
        # Parameter Learning
        self.bayesian_network.learn_parameters(
            discrete_data,
            estimator=self.config['bn_estimator']
        )
        
        bn_edges = list(self.bayesian_network.model.edges())
        logger.info(f"  Learned {len(bn_edges)} edges in Bayesian Network")
        
        # Save network visualization
        bn_viz_path = os.path.join(self.output_dir, 'results', 'bayesian_network.png')
        self.bayesian_network.visualize_structure(bn_viz_path)
        
        return bn_edges
    
    def _step9_save_models(self):
        """Step 9: Save All Models"""
        models_dir = os.path.join(self.output_dir, 'models')
        
        # Save preprocessor
        joblib.dump(self.preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
        
        # Save prediction model (using new format)
        self.predictor.model.save(os.path.join(models_dir, 'predictor.keras'))
        
        # Save state classifier
        joblib.dump(self.state_classifier, os.path.join(models_dir, 'state_classifier.pkl'))
        
        # Save discretizer
        joblib.dump(self.discretizer, os.path.join(models_dir, 'discretizer.pkl'))
        
        # Save clusterers
        joblib.dump(self.cam_clusterer, os.path.join(models_dir, 'cam_clusterer.pkl'))
        joblib.dump(self.attention_clusterer, os.path.join(models_dir, 'attention_clusterer.pkl'))
        
        # Save Bayesian Network
        self.bayesian_network.save_model(os.path.join(models_dir, 'bayesian_network.bif'))
        
        # Save configuration
        import json
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"  All models saved to {models_dir}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Mock Data
    np.random.seed(42)
    n_samples = 300
    
    train_data = pd.DataFrame({
        'Temperature': np.random.randn(n_samples) * 10 + 20,
        'Humidity': np.random.randn(n_samples) * 15 + 60,
        'WindSpeed': np.random.randn(n_samples) * 5 + 10,
        'EDP': np.random.randn(n_samples) * 30 + 100
    })
    
    # Create and run pipeline
    pipeline = TrainPipeline(output_dir='./outputs')
    results = pipeline.run(train_data)
    
    print("\n" + "="*60)
    print("Training Results Summary:")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")