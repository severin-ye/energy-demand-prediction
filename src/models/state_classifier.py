"""
Sn尺度状态分类器（基于论文公式5-7）
"""

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class SnStateClassifier:
    """
    基于Sn鲁棒尺度估计器的状态分类器
    
    论文公式:
    - Sn = c · median_m { median_n |x_m - x_n| }  (公式5)
    - Z_Sn(x_i) = |x_i - MED(X)| / (α × Sn)      (公式6)
    - 状态判定: Peak/Normal/Lower                (公式7)
    
    其中:
    - c = 1.1926 (正态分布修正因子)
    - α = 1.4285 (论文参数)
    """
    
    def __init__(self, n_states=3, state_names=None, threshold=2.0):
        """
        参数:
            n_states: 状态数量（默认3: Lower/Normal/Peak）
            state_names: 状态名称列表
            threshold: Z分数阈值（默认2.0）
        """
        self.n_states = n_states
        self.state_names = state_names or ['Lower', 'Normal', 'Peak']
        self.threshold = threshold
        
        # 论文常量
        self.c = 1.1926  # Sn修正因子
        self.alpha = 1.4285  # Z分数修正系数
        
        self.sn_scale_ = None
        self.median_ = None
    
    def compute_sn_scale(self, data):
        """
        计算Sn尺度估计器（论文公式5）
        
        Sn = c · median_m { median_n |x_m - x_n| }
        
        其中 c = 1.1926 是修正因子，使其在正态分布下无偏
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # 计算所有成对差异的中位数
        # 对于大数据集，采样以提高效率
        sample_size = min(n, 1000)
        if n > sample_size:
            indices = np.random.choice(n, sample_size, replace=False)
            data_sample = data[indices]
        else:
            data_sample = data
        
        diffs = []
        for i in range(len(data_sample)):
            diffs.append(np.median(np.abs(data_sample[i] - data_sample)))
        
        sn = np.median(diffs)
        
        # 应用修正因子
        return self.c * sn
    
    def compute_z_score(self, value, window_data):
        """
        计算鲁棒Z分数（论文公式6）
        
        Z_Sn(x_i) = |x_i - MED(X)| / (α × Sn)
        
        参数:
            value: 要评估的值
            window_data: 观测窗口数据
        
        返回:
            鲁棒Z分数
        """
        median = np.median(window_data)
        sn = self.compute_sn_scale(window_data)
        
        # 避免除零
        if sn < 1e-10:
            return 0.0
        
        z_score = abs(value - median) / (self.alpha * sn)
        return z_score
    
    def fit(self, data):
        """
        拟合分类器（计算全局统计量）
        
        参数:
            data: 训练数据
        """
        data = np.asarray(data).flatten()
        
        logger.info("计算Sn尺度估计器...")
        self.sn_scale_ = self.compute_sn_scale(data)
        self.median_ = np.median(data)
        
        logger.info(f"中位数: {self.median_:.4f}, Sn尺度: {self.sn_scale_:.4f}")
        logger.info(f"α系数: {self.alpha}, 阈值: {self.threshold}")
        
        return self
    
    def predict(self, data, window_data=None):
        """
        预测状态（基于论文公式7）
        
        论文判定逻辑:
        - 如果 Z_Sn(x_i) > threshold 且 x_i > median: Peak
        - 如果 Z_Sn(x_i) > threshold 且 x_i < median: Lower
        - 否则: Normal
        
        参数:
            data: 能源需求值（单个值或数组）
            window_data: 观测窗口数据（用于计算Z分数，如果为None则使用训练数据统计量）
        
        返回:
            状态标签（字符串或数组）
        """
        is_scalar = np.isscalar(data)
        data = np.asarray(data).flatten()
        
        if self.median_ is None or self.sn_scale_ is None:
            raise ValueError("Must call fit() before predict()")
        
        states = []
        for value in data:
            # 使用窗口数据或全局统计量计算Z分数
            if window_data is not None:
                z_score = self.compute_z_score(value, window_data)
                median = np.median(window_data)
            else:
                # 使用训练时的全局统计量
                z_score = abs(value - self.median_) / (self.alpha * self.sn_scale_)
                median = self.median_
            
            # 论文公式7的判定逻辑
            if z_score > self.threshold:
                if value > median:
                    state = self.state_names[2]  # Peak
                else:
                    state = self.state_names[0]  # Lower
            else:
                state = self.state_names[1]  # Normal
            
            states.append(state)
        
        states = np.array(states)
        return states[0] if is_scalar else states
    
    def predict_with_scores(self, data, window_data=None):
        """
        预测状态并返回Z分数
        
        返回:
            (states, z_scores) 元组
        """
        is_scalar = np.isscalar(data)
        data = np.asarray(data).flatten()
        
        states = []
        z_scores = []
        
        for value in data:
            if window_data is not None:
                z_score = self.compute_z_score(value, window_data)
                median = np.median(window_data)
            else:
                z_score = abs(value - self.median_) / (self.alpha * self.sn_scale_)
                median = self.median_
            
            if z_score > self.threshold:
                state = self.state_names[2] if value > median else self.state_names[0]
            else:
                state = self.state_names[1]
            
            states.append(state)
            z_scores.append(z_score)
        
        states = np.array(states)
        z_scores = np.array(z_scores)
        
        if is_scalar:
            return states[0], z_scores[0]
        return states, z_scores
    
    def fit_predict(self, data):
        """拟合并预测"""
        self.fit(data)
        return self.predict(data)


# 使用示例
if __name__ == "__main__":
    # 模拟数据（包含异常值）
    np.random.seed(42)
    
    data = np.concatenate([
        np.random.normal(1.0, 0.3, 300),   # Lower
        np.random.normal(3.0, 0.5, 500),   # Normal
        np.random.normal(6.0, 0.8, 200),   # Peak
        np.array([15.0, 20.0, -2.0])       # 异常值
    ])
    
    # 分类
    classifier = SnStateClassifier()
    states = classifier.fit_predict(data)
    
    # 统计
    unique, counts = np.unique(states, return_counts=True)
    for state, count in zip(unique, counts):
        print(f"{state}: {count} samples")
