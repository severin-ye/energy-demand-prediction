"""
数据处理模块

包含UCI数据集下载、预处理、划分等功能
"""

from .uci_loader import UCIDataLoader
from .data_splitter import DataSplitter

__all__ = [
    'UCIDataLoader',
    'DataSplitter',
]
