"""
Translation Checker - 翻译质量检查工具

一个基于机器学习的翻译质量检查工具，用于验证翻译的准确性和一致性。
"""

__version__ = "1.0.0"
__author__ = "Translation Checker Team"
__email__ = "support@translationchecker.com"

from .core.checker import TranslationChecker
from .processors.data_processor import DataProcessor
from .models.similarity_model import SimilarityModel

__all__ = [
    "TranslationChecker",
    "DataProcessor", 
    "SimilarityModel"
]
