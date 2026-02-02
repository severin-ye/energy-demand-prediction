"""
消融实验脚本 - 对比不同模型配置的性能

实验组:
1. Parallel CNN-LSTM-Attention (本文方法)
2. Serial CNN-LSTM-Attention (串联+注意力)
3. Serial CNN-LSTM (串联基线)
4. Parallel CNN-LSTM (无注意力)

目标: 验证论文声称的34.84%性能提升
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing.data_preprocessor import EnergyDataPreprocessor
from src.models.predictor import ParallelCNNLSTMAttention
from src.models.baseline_models import SerialCNNLSTM, SerialCNNLSTMAttention
from src.data_processing.uci_loader import load_uci_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationStudy:
    """消融实验管理器"""
    
    def __init__(self, config_path='configs/paper_config.json'):
        """
        参数:
            config_path: 配置文件路径
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.results = {}
        self.models = {}
    
    def prepare_data(self):
        """准备数据"""
        logger.info("=" * 80)
        logger.info("加载和预处理数据")
        logger.info("=" * 80)
        
        # 加载UCI数据集（使用已分割的训练/测试集）
        train_df, test_df = load_uci_dataset(use_splits=True)
        
        # 预处理
        preprocessor = EnergyDataPreprocessor(
            sequence_length=self.config['sequence_length'],
            feature_cols=self.config['feature_cols'],
            target_col=self.config['target_col']
        )
        
        logger.info("预处理训练数据...")
        X_train, y_train = preprocessor.fit_transform(train_df)
        
        logger.info("预处理测试数据...")
        X_test, y_test = preprocessor.transform(test_df)
        
        logger.info(f"训练集: X{X_train.shape}, y{y_train.shape}")
        logger.info(f"测试集: X{X_test.shape}, y{y_test.shape}")
        
        return X_train, y_train, X_test, y_test, preprocessor
    
    def build_model(self, model_type: str, input_shape: tuple):
        """
        构建模型
        
        参数:
            model_type: 'parallel-att', 'serial-att', 'serial', 'parallel'
            input_shape: 输入形状
        """
        if model_type == 'parallel-att':
            logger.info("构建并行CNN-LSTM-Attention（本文方法）")
            model = ParallelCNNLSTMAttention(
                input_shape=input_shape,
                cnn_filters=self.config['cnn_filters'][0],
                lstm_units=self.config['lstm_units'],
                attention_units=self.config['attention_units'],
                dense_units=self.config['dense_units']
            )
        
        elif model_type == 'serial-att':
            logger.info("构建串联CNN-LSTM-Attention")
            model = SerialCNNLSTMAttention(
                input_shape=input_shape,
                cnn_filters=self.config['cnn_filters'][0],
                lstm_units=self.config['lstm_units'],
                attention_units=self.config['attention_units'],
                dense_units=self.config['dense_units']
            )
        
        elif model_type == 'serial':
            logger.info("构建串联CNN-LSTM（基线）")
            model = SerialCNNLSTM(
                input_shape=input_shape,
                cnn_filters=self.config['cnn_filters'][0],
                lstm_units=self.config['lstm_units'],
                dense_units=self.config['dense_units']
            )
        
        elif model_type == 'parallel':
            logger.info("构建并行CNN-LSTM（无注意力）")
            # TODO: 需要实现无注意力的并行版本
            raise NotImplementedError("待实现")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def train_model(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        model_name: str
    ):
        """训练单个模型"""
        logger.info("=" * 80)
        logger.info(f"训练模型: {model_name}")
        logger.info("=" * 80)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 15),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
        model_name: str
    ) -> dict:
        """评估单个模型"""
        logger.info(f"评估模型: {model_name}")
        
        y_pred = model.predict(X_test).flatten()
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        
        # MAPE（避免除零）
        mask = y_test > 0.01
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        results = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'mape': mape
        }
        
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return results
    
    def compute_improvement(self, baseline_results: dict, improved_results: dict):
        """计算性能提升百分比"""
        improvements = {}
        
        for metric in ['mae', 'rmse', 'mse']:
            baseline_val = baseline_results[metric]
            improved_val = improved_results[metric]
            
            # 提升 = (baseline - improved) / baseline * 100
            improvement = (baseline_val - improved_val) / baseline_val * 100
            improvements[metric] = improvement
        
        return improvements
    
    def run_experiment(self):
        """运行完整消融实验"""
        logger.info("\n" + "=" * 80)
        logger.info("开始消融实验")
        logger.info("=" * 80 + "\n")
        
        # 1. 准备数据
        X_train, y_train, X_test, y_test, preprocessor = self.prepare_data()
        
        # 划分验证集
        val_split = int(len(X_train) * 0.2)
        X_val = X_train[-val_split:]
        y_val = y_train[-val_split:]
        X_train = X_train[:-val_split]
        y_train = y_train[:-val_split]
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 2. 实验配置
        experiments = [
            ('serial', '串联CNN-LSTM（基线）'),
            ('serial-att', '串联CNN-LSTM-Attention'),
            ('parallel-att', '并行CNN-LSTM-Attention（本文）')
        ]
        
        # 3. 运行所有实验
        all_results = []
        
        for model_type, model_name in experiments:
            try:
                # 构建模型
                model = self.build_model(model_type, input_shape)
                
                # 训练
                history = self.train_model(
                    model, X_train, y_train, X_val, y_val, model_name
                )
                
                # 评估
                results = self.evaluate_model(model, X_test, y_test, model_name)
                all_results.append(results)
                
                # 保存模型
                output_dir = f"outputs/ablation/{model_type}"
                os.makedirs(output_dir, exist_ok=True)
                model.save(f"{output_dir}/model.keras")
                
                self.models[model_type] = model
                self.results[model_type] = results
                
            except Exception as e:
                logger.error(f"实验 {model_name} 失败: {e}")
                continue
        
        # 4. 生成对比报告
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, results: list):
        """生成对比报告"""
        logger.info("\n" + "=" * 80)
        logger.info("消融实验结果")
        logger.info("=" * 80 + "\n")
        
        # 创建结果表
        df = pd.DataFrame(results)
        df = df.sort_values('mae')
        
        logger.info(df.to_string(index=False))
        
        # 计算相对于baseline的提升
        baseline_idx = df[df['model'].str.contains('基线')].index[0]
        baseline_results = df.iloc[baseline_idx]
        
        logger.info("\n" + "=" * 80)
        logger.info("相对于基线的性能提升")
        logger.info("=" * 80 + "\n")
        
        for idx, row in df.iterrows():
            if idx == baseline_idx:
                continue
            
            improvements = self.compute_improvement(
                baseline_results.to_dict(),
                row.to_dict()
            )
            
            logger.info(f"{row['model']}:")
            logger.info(f"  MAE提升: {improvements['mae']:+.2f}%")
            logger.info(f"  RMSE提升: {improvements['rmse']:+.2f}%")
            logger.info(f"  MSE提升: {improvements['mse']:+.2f}%")
            logger.info("")
        
        # 保存结果
        output_dir = "outputs/ablation"
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(f"{output_dir}/ablation_results.csv", index=False)
        
        # 生成报告文档
        report_path = f"{output_dir}/ABLATION_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 实验配置\n\n")
            f.write(f"- 序列长度: {self.config['sequence_length']}\n")
            f.write(f"- LSTM单元: {self.config['lstm_units']}\n")
            f.write(f"- 注意力单元: {self.config['attention_units']}\n")
            f.write(f"- 训练轮数: {self.config['epochs']}\n")
            f.write(f"- 批大小: {self.config['batch_size']}\n\n")
            f.write("## 结果对比\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## 性能提升\n\n")
            
            for idx, row in df.iterrows():
                if idx == baseline_idx:
                    continue
                improvements = self.compute_improvement(
                    baseline_results.to_dict(),
                    row.to_dict()
                )
                f.write(f"### {row['model']}\n\n")
                f.write(f"- MAE提升: **{improvements['mae']:+.2f}%**\n")
                f.write(f"- RMSE提升: **{improvements['rmse']:+.2f}%**\n")
                f.write(f"- MSE提升: **{improvements['mse']:+.2f}%**\n\n")
        
        logger.info(f"报告已保存到: {report_path}")


if __name__ == "__main__":
    # 运行消融实验
    study = AblationStudy('configs/paper_config.json')
    results = study.run_experiment()
    
    logger.info("\n实验完成！")
