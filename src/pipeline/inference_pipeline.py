"""
推理流水线
单样本预测 + 因果解释 + 优化推荐
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
    推理流水线
    
    功能:
    1. 加载所有训练好的模型
    2. 对单个样本进行预测
    3. 提取DLP并进行聚类
    4. 因果推断分析
    5. 生成优化建议
    
    参数:
    - models_dir: 模型目录路径
    """
    
    def __init__(self, models_dir: str = './outputs/models'):
        self.models_dir = models_dir
        
        # 模型组件
        self.preprocessor = None
        self.predictor = None
        self.state_classifier = None
        self.discretizer = None
        self.cam_clusterer = None
        self.attention_clusterer = None
        self.bayesian_network = None
        self.causal_inference = None
        self.recommendation_engine = None
        
        # 加载模型
        self.load_models()
        
        logger.info("InferencePipeline initialized")
    
    def load_models(self):
        """加载所有模型"""
        logger.info(f"Loading models from {self.models_dir}...")
        
        # 加载预处理器
        self.preprocessor = joblib.load(os.path.join(self.models_dir, 'preprocessor.pkl'))
        
        # 加载预测模型
        from tensorflow import keras
        from src.models.predictor import AttentionLayer
        
        # 注册自定义层并加载模型
        model_path = os.path.join(self.models_dir, 'predictor.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.models_dir, 'predictor.h5')
        
        with keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
            self.predictor_model = keras.models.load_model(model_path)
        
        # 构建CAM和Attention提取模型
        self.cam_model = None
        self.attention_model = None
        
        # 加载状态分类器
        self.state_classifier = joblib.load(os.path.join(self.models_dir, 'state_classifier.pkl'))
        
        # 加载离散化器
        self.discretizer = joblib.load(os.path.join(self.models_dir, 'discretizer.pkl'))
        
        # 加载聚类器
        self.cam_clusterer = joblib.load(os.path.join(self.models_dir, 'cam_clusterer.pkl'))
        self.attention_clusterer = joblib.load(os.path.join(self.models_dir, 'attention_clusterer.pkl'))
        
        # 加载贝叶斯网络
        from src.models.bayesian_net import CausalBayesianNetwork
        self.bayesian_network = CausalBayesianNetwork()
        self.bayesian_network.load_model(os.path.join(self.models_dir, 'bayesian_network.bif'))
        
        # 创建因果推断和推荐引擎
        from src.inference.causal_inference import CausalInference
        from src.inference.recommendation import RecommendationEngine
        
        self.causal_inference = CausalInference(self.bayesian_network, target_var='EDP_State')
        self.recommendation_engine = RecommendationEngine(self.causal_inference)
        
        logger.info("All models loaded successfully")
    
    def _extract_cam(self, X):
        """提取CAM特征"""
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
        """提取Attention权重"""
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
        对单个或多个样本进行预测和解释
        
        参数:
        - data: 输入数据（包含特征列，至少sequence_length行）
        - generate_recommendations: 是否生成优化建议
        
        返回:
        - 预测结果字典
        """
        logger.info("Starting inference...")
        
        results = {}
        
        # Step 1: 预处理
        X, _ = self.preprocessor.transform(data)
        results['n_samples'] = len(X)
        
        # Step 2: 预测
        predictions = self.predictor_model.predict(X, verbose=0)
        results['predictions'] = predictions.flatten()
        
        # Step 3: 提取DLP
        cam_features = self._extract_cam(X)
        attention_features = self._extract_attention(X)
        
        # Step 4: DLP聚类
        cam_clusters = self.cam_clusterer.predict(cam_features)
        attention_clusters = self.attention_clusterer.predict(attention_features)
        attention_types = [
            self.attention_clusterer.cluster_names_[c] for c in attention_clusters
        ]
        
        results['cam_clusters'] = cam_clusters
        results['attention_types'] = attention_types
        
        # Step 5: 状态分类
        edp_states = self.state_classifier.predict(predictions)
        results['edp_states'] = edp_states
        
        # Step 6: 离散化特征
        features_to_discretize = self.preprocessor.feature_cols
        discrete_data = self.discretizer.transform(
            data[features_to_discretize].iloc[:len(predictions)]
        )
        
        # 转换为DataFrame
        discrete_features = pd.DataFrame(
            discrete_data,
            columns=features_to_discretize
        )
        
        # 构建当前状态字典（取最后一个样本）
        current_state = discrete_features.iloc[-1].to_dict()
        current_state['EDP_State'] = edp_states[-1]
        current_state['CAM_Cluster'] = str(cam_clusters[-1])
        current_state['Attention_Type'] = attention_types[-1]
        
        results['current_state'] = current_state
        
        # Step 7: 因果推断（如果需要生成建议）
        if generate_recommendations:
            # 敏感性分析
            features_to_analyze = list(discrete_features.columns)
            sensitivity = self.causal_inference.sensitivity_analysis(
                features_to_analyze,
                target_state='Peak'
            )
            results['sensitivity'] = sensitivity
            
            # 生成建议
            recommendations = self.recommendation_engine.generate_recommendations(
                current_state,
                target_state='Peak',
                top_k=5
            )
            results['recommendations'] = recommendations
            
            # 生成报告
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
        对单个样本进行预测并生成完整报告
        
        参数:
        - data: 输入数据（至少sequence_length行）
        - verbose: 是否打印详细信息
        
        返回:
        - 格式化的报告文本
        """
        results = self.predict(data, generate_recommendations=True)
        
        # 提取关键信息
        prediction = results['predictions'][-1]
        edp_state = results['edp_states'][-1]
        cam_cluster = results['cam_clusters'][-1]
        attention_type = results['attention_types'][-1]
        
        # 生成简要报告
        if verbose:
            print("\n" + "="*60)
            print("           预测结果")
            print("="*60)
            print(f"预测负荷: {prediction:.2f} kWh")
            print(f"负荷状态: {edp_state}")
            print(f"CAM聚类: {cam_cluster}")
            print(f"注意力类型: {attention_type}")
            print("\n")
        
        # 返回完整报告
        return results['report']
    
    def batch_predict(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量预测（不生成建议，仅预测）
        
        参数:
        - data: 输入数据
        - output_path: 输出CSV路径（可选）
        
        返回:
        - 包含预测结果的DataFrame
        """
        results = self.predict(data, generate_recommendations=False)
        
        # 构建结果DataFrame
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
        解释单个预测结果
        
        参数:
        - sample_idx: 样本索引
        - data: 完整数据
        
        返回:
        - 解释文本
        """
        results = self.predict(data, generate_recommendations=True)
        
        # 提取该样本的信息
        prediction = results['predictions'][sample_idx]
        state = results['edp_states'][sample_idx]
        current_state = results['current_state']
        
        # 生成解释
        explanation = self.causal_inference.generate_explanation(
            current_state,
            target_state=state,
            top_k=3
        )
        
        full_explanation = (
            f"样本 #{sample_idx} 预测解释\n"
            f"{'='*60}\n"
            f"预测负荷: {prediction:.2f} kWh\n"
            f"负荷状态: {state}\n\n"
            f"{explanation}"
        )
        
        return full_explanation


if __name__ == "__main__":
    # 示例使用
    import sys
    sys.path.append('/home/severin/Codelib/YS')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 注意：需要先运行训练流水线生成模型
    # 这里仅演示API用法
    
    try:
        # 创建推理流水线
        pipeline = InferencePipeline(models_dir='./outputs/models')
        
        # 模拟新数据
        np.random.seed(42)
        new_data = pd.DataFrame({
            'Temperature': np.random.randn(25) * 10 + 20,
            'Humidity': np.random.randn(25) * 15 + 60,
            'WindSpeed': np.random.randn(25) * 5 + 10
        })
        
        # 单样本预测
        report = pipeline.predict_single(new_data, verbose=True)
        print(report)
        
    except FileNotFoundError:
        print("模型文件未找到，请先运行训练流水线")
