"""
训练流水线
端到端训练流程，集成所有模块
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
    训练流水线（9步流程）
    
    步骤:
    1. 数据预处理（序列生成）
    2. 训练Parallel CNN-LSTM-Attention模型
    3. 提取CAM和Attention权重（DLP）
    4. DLP聚类（CAM聚类 + Attention聚类）
    5. 计算Sn鲁棒尺度估计并进行状态分类
    6. 特征离散化（4级量化）
    7. 关联规则挖掘（Apriori）
    8. 贝叶斯网络结构和参数学习
    9. 保存所有模型和结果
    
    参数:
    - config: 配置字典
    - output_dir: 输出目录
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: str = './outputs'
    ):
        self.config = config or self._default_config()
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        
        # 模型组件
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
        """默认配置"""
        return {
            # 数据预处理
            'sequence_length': 20,
            'feature_cols': ['Temperature', 'Humidity', 'WindSpeed'],
            'target_col': 'EDP',
            
            # 预测模型
            'cnn_filters': [64, 32],
            'lstm_units': 64,
            'attention_units': 50,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            
            # 状态分类
            'n_states': 3,
            'state_names': ['Lower', 'Normal', 'Peak'],
            
            # 离散化
            'n_bins': 4,
            'bin_labels': ['Low', 'Medium', 'High', 'VeryHigh'],
            
            # DLP聚类
            'n_cam_clusters': 3,
            'n_attention_clusters': 3,
            
            # 关联规则
            'min_support': 0.05,
            'min_confidence': 0.6,
            'min_lift': 1.2,
            
            # 贝叶斯网络
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
        执行完整训练流水线
        
        参数:
        - train_data: 训练数据（包含特征列和目标列）
        - val_data: 验证数据（可选）
        
        返回:
        - 训练结果字典
        """
        logger.info("="*60)
        logger.info("Starting Training Pipeline")
        logger.info("="*60)
        
        results = {}
        
        # Step 1: 数据预处理
        logger.info("\n[Step 1/9] Data Preprocessing...")
        X_train, y_train, X_val, y_val = self._step1_preprocess(train_data, val_data)
        results['data_shapes'] = {
            'X_train': X_train.shape,
            'y_train': y_train.shape
        }
        
        # Step 2: 训练预测模型
        logger.info("\n[Step 2/9] Training Parallel CNN-LSTM-Attention Model...")
        history = self._step2_train_predictor(X_train, y_train, X_val, y_val)
        results['training_history'] = {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', [])
        }
        
        # Step 3: 提取DLP
        logger.info("\n[Step 3/9] Extracting Deep Learning Parameters (CAM & Attention)...")
        cam_features, attention_features = self._step3_extract_dlp(X_train)
        results['dlp_shapes'] = {
            'cam': cam_features.shape,
            'attention': attention_features.shape
        }
        
        # Step 4: DLP聚类
        logger.info("\n[Step 4/9] Clustering DLP Features...")
        cam_clusters, attention_types = self._step4_cluster_dlp(cam_features, attention_features)
        results['cluster_distributions'] = {
            'cam': np.bincount(cam_clusters),
            'attention': pd.Series(attention_types).value_counts().to_dict()
        }
        
        # Step 5: Sn状态分类
        logger.info("\n[Step 5/9] Sn State Classification...")
        edp_states = self._step5_classify_states(y_train)
        results['state_distribution'] = pd.Series(edp_states).value_counts().to_dict()
        
        # Step 6: 特征离散化
        logger.info("\n[Step 6/9] Feature Discretization...")
        discrete_data = self._step6_discretize(train_data, edp_states, cam_clusters, attention_types)
        results['discrete_features'] = list(discrete_data.columns)
        
        # Step 7: 关联规则挖掘
        logger.info("\n[Step 7/9] Association Rule Mining...")
        candidate_edges = self._step7_mine_rules(discrete_data)
        results['candidate_edges'] = candidate_edges
        
        # Step 8: 贝叶斯网络学习
        logger.info("\n[Step 8/9] Bayesian Network Learning...")
        bn_edges = self._step8_learn_bayesian_network(discrete_data, candidate_edges)
        results['bn_edges'] = bn_edges
        
        # Step 9: 保存模型
        logger.info("\n[Step 9/9] Saving Models...")
        self._step9_save_models()
        
        logger.info("\n" + "="*60)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("="*60)
        
        return results
    
    def _step1_preprocess(self, train_data, val_data):
        """步骤1：数据预处理"""
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
        """步骤2：训练预测模型"""
        from src.models.predictor import ParallelCNNLSTMAttention
        
        self.predictor = ParallelCNNLSTMAttention(
            input_shape=(X_train.shape[1], X_train.shape[2]),  # (sequence_length, n_features)
            cnn_filters=self.config['cnn_filters'][0] if isinstance(self.config['cnn_filters'], list) else self.config['cnn_filters'],
            lstm_units=self.config['lstm_units'],
            attention_units=self.config['attention_units']
        )
        
        # 编译和训练
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
        """步骤3：提取DLP（CAM和Attention权重）"""
        cam_features = self.predictor.extract_cam(X_train)
        attention_features = self.predictor.extract_attention_weights(X_train)
        
        logger.info(f"  CAM: {cam_features.shape}")
        logger.info(f"  Attention: {attention_features.shape}")
        
        return cam_features, attention_features
    
    def _step4_cluster_dlp(self, cam_features, attention_features):
        """步骤4：DLP聚类"""
        from src.models.clustering import DLPClusterer, AttentionClusterer
        
        # CAM聚类
        self.cam_clusterer = DLPClusterer(n_clusters=self.config['n_cam_clusters'])
        cam_clusters = self.cam_clusterer.fit_predict(cam_features)
        
        # Attention聚类
        self.attention_clusterer = AttentionClusterer(n_clusters=self.config['n_attention_clusters'])
        attention_clusters = self.attention_clusterer.fit_predict(attention_features)
        attention_types = [self.attention_clusterer.cluster_names_[c] for c in attention_clusters]
        
        logger.info(f"  CAM clusters: {np.bincount(cam_clusters)}")
        logger.info(f"  Attention types: {pd.Series(attention_types).value_counts().to_dict()}")
        
        return cam_clusters, attention_types
    
    def _step5_classify_states(self, y_train):
        """步骤5：Sn状态分类"""
        from src.models.state_classifier import SnStateClassifier
        
        self.state_classifier = SnStateClassifier(
            n_states=self.config['n_states'],
            state_names=self.config['state_names']
        )
        
        edp_states = self.state_classifier.fit_predict(y_train)
        
        logger.info(f"  State distribution: {pd.Series(edp_states).value_counts().to_dict()}")
        
        return edp_states
    
    def _step6_discretize(self, train_data, edp_states, cam_clusters, attention_types):
        """步骤6：特征离散化"""
        from src.models.discretizer import QuantileDiscretizer
        
        # 原始特征离散化
        self.discretizer = QuantileDiscretizer(
            n_bins=self.config['n_bins'],
            strategy='quantile'  # QuantileDiscretizer只接受n_bins和strategy参数
        )
        
        features_to_discretize = self.config['feature_cols'].copy()
        # 确保只取与edp_states长度相同的样本
        n_samples = len(edp_states)
        discrete_data = self.discretizer.fit_transform(
            train_data[features_to_discretize].iloc[:n_samples]
        )
        
        # 转换为DataFrame（如果是numpy数组）
        if not isinstance(discrete_data, pd.DataFrame):
            discrete_data = pd.DataFrame(
                discrete_data,
                columns=features_to_discretize
            )
        
        # 添加EDP状态、CAM聚类、Attention类型
        discrete_data['EDP_State'] = edp_states
        discrete_data['CAM_Cluster'] = cam_clusters.astype(str)  # 转为字符串便于编码
        discrete_data['Attention_Type'] = attention_types
        
        logger.info(f"  Discretized features: {list(discrete_data.columns)}")
        
        return discrete_data
    
    def _step7_mine_rules(self, discrete_data):
        """步骤7：关联规则挖掘"""
        from src.models.association import AssociationRuleMiner
        
        self.association_miner = AssociationRuleMiner(
            min_support=self.config['min_support'],
            min_confidence=self.config['min_confidence'],
            min_lift=self.config['min_lift']
        )
        
        # 准备数据
        df_encoded = self.association_miner.prepare_data(discrete_data, edp_col='EDP_State')
        
        # 挖掘频繁项集
        self.association_miner.mine_frequent_itemsets(df_encoded)
        
        # 生成规则
        self.association_miner.generate_rules()
        
        # 筛选EDP规则
        self.association_miner.filter_edp_rules()
        
        # 提取候选边
        candidate_edges = self.association_miner.rules_to_constraints(top_k=50)
        
        logger.info(f"  Found {len(candidate_edges)} candidate edges")
        
        # 保存规则
        rules_path = os.path.join(self.output_dir, 'results', 'association_rules.csv')
        self.association_miner.save_rules(rules_path)
        
        return candidate_edges
    
    def _step8_learn_bayesian_network(self, discrete_data, candidate_edges):
        """步骤8：贝叶斯网络学习"""
        from src.models.bayesian_net import CausalBayesianNetwork
        
        # 领域知识约束（可根据实际情况调整）
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
        
        # 结构学习
        self.bayesian_network.learn_structure(
            discrete_data,
            candidate_edges=candidate_edges,
            max_iter=self.config['bn_max_iter']
        )
        
        # 参数学习
        self.bayesian_network.learn_parameters(
            discrete_data,
            estimator=self.config['bn_estimator']
        )
        
        bn_edges = list(self.bayesian_network.model.edges())
        logger.info(f"  Learned {len(bn_edges)} edges in Bayesian Network")
        
        # 保存网络结构图
        bn_viz_path = os.path.join(self.output_dir, 'results', 'bayesian_network.png')
        self.bayesian_network.visualize_structure(bn_viz_path)
        
        return bn_edges
    
    def _step9_save_models(self):
        """步骤9：保存所有模型"""
        models_dir = os.path.join(self.output_dir, 'models')
        
        # 保存预处理器
        joblib.dump(self.preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
        
        # 保存预测模型（使用新格式）
        self.predictor.model.save(os.path.join(models_dir, 'predictor.keras'))
        
        # 保存状态分类器
        joblib.dump(self.state_classifier, os.path.join(models_dir, 'state_classifier.pkl'))
        
        # 保存离散化器
        joblib.dump(self.discretizer, os.path.join(models_dir, 'discretizer.pkl'))
        
        # 保存聚类器
        joblib.dump(self.cam_clusterer, os.path.join(models_dir, 'cam_clusterer.pkl'))
        joblib.dump(self.attention_clusterer, os.path.join(models_dir, 'attention_clusterer.pkl'))
        
        # 保存贝叶斯网络
        self.bayesian_network.save_model(os.path.join(models_dir, 'bayesian_network.bif'))
        
        # 保存配置
        import json
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"  All models saved to {models_dir}")


if __name__ == "__main__":
    # 示例使用
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 300
    
    train_data = pd.DataFrame({
        'Temperature': np.random.randn(n_samples) * 10 + 20,
        'Humidity': np.random.randn(n_samples) * 15 + 60,
        'WindSpeed': np.random.randn(n_samples) * 5 + 10,
        'EDP': np.random.randn(n_samples) * 30 + 100
    })
    
    # 创建并运行流水线
    pipeline = TrainPipeline(output_dir='./outputs')
    results = pipeline.run(train_data)
    
    print("\n" + "="*60)
    print("Training Results Summary:")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")
