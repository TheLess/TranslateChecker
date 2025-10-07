"""
数据处理模块
负责Excel文件的读取、预处理和数据清洗
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger


class DataProcessor:
    """数据处理器，负责Excel文件的读取和预处理"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # 从配置中获取列名映射
        self.column_mapping = self.config.get('data.column_mapping', {})
        self.supported_formats = self.config.get('data.supported_formats', ['.xlsx', '.xls', '.csv'])
        self.encoding = self.config.get('data.encoding', 'utf-8')
        
    def load_excel_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        加载Excel文件
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            pandas DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
        self.logger.info(f"正在加载文件: {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding=self.encoding)
            else:
                df = pd.read_excel(file_path)
                
            self.logger.info(f"成功加载文件，共 {len(df)} 行数据")
            return df
            
        except Exception as e:
            self.logger.error(f"加载文件失败: {e}")
            raise
            
    def identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        自动识别ID列、源语言和目标语言列
        
        Args:
            df: 数据框
            
        Returns:
            包含ID、源语言和目标语言列名的字典
            
        Raises:
            ValueError: 无法识别必要的列
        """
        columns = df.columns.tolist()
        result = {}
        
        # 识别ID列
        id_candidates = self.column_mapping.get('id', [])
        for col in columns:
            if any(candidate.lower() in col.lower() for candidate in id_candidates):
                result['id'] = col
                break
        
        # 识别源语言列（中文）
        source_candidates = self.column_mapping.get('source', [])
        for col in columns:
            if any(candidate.lower() in col.lower() for candidate in source_candidates):
                result['source'] = col
                break
                
        # 识别目标语言列（英文）
        target_candidates = self.column_mapping.get('target', [])
        for col in columns:
            if any(candidate.lower() in col.lower() for candidate in target_candidates):
                result['target'] = col
                break
        
        # 如果没有找到ID列，检查是否有数字类型的列作为ID
        if 'id' not in result:
            for col in columns:
                if df[col].dtype in ['int64', 'int32', 'float64'] and col not in [result.get('source'), result.get('target')]:
                    # 检查是否是连续的数字序列
                    if df[col].is_monotonic_increasing:
                        result['id'] = col
                        self.logger.info(f"自动识别数字序列列作为ID: {col}")
                        break
        
        # 如果自动识别失败，根据列数量进行推断
        available_columns = [col for col in columns if col not in result.values()]
        
        if 'source' not in result:
            if len(available_columns) >= 2:
                # 如果有ID列，源语言列通常是第二列，否则是第一列
                idx = 1 if 'id' in result else 0
                if idx < len(available_columns):
                    result['source'] = available_columns[idx]
                    self.logger.warning(f"无法自动识别源语言列，使用: {available_columns[idx]}")
            else:
                raise ValueError("无法识别源语言列")
                
        if 'target' not in result:
            available_columns = [col for col in columns if col not in result.values()]
            if len(available_columns) >= 1:
                result['target'] = available_columns[0]
                self.logger.warning(f"无法自动识别目标语言列，使用: {available_columns[0]}")
            else:
                raise ValueError("无法识别目标语言列")
        
        # 记录识别结果
        log_msg = f"识别到列映射: "
        if 'id' in result:
            log_msg += f"ID='{result['id']}', "
        log_msg += f"源语言='{result['source']}', 目标语言='{result['target']}'"
        self.logger.info(log_msg)
        
        return result
        
    def clean_text(self, text: str) -> str:
        """
        清洗文本数据
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if pd.isna(text) or text is None:
            return ""
            
        text = str(text).strip()
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊的Unicode字符
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        # 移除首尾的标点符号（可选）
        # text = text.strip('.,;:!?。，；：！？')
        
        return text
        
    def preprocess_data(self, df: pd.DataFrame, source_col: str, target_col: str, id_col: Optional[str] = None) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df: 原始数据框
            source_col: 源语言列名
            target_col: 目标语言列名
            id_col: ID列名（可选）
            
        Returns:
            预处理后的数据框
        """
        self.logger.info("开始数据预处理...")
        
        # 创建副本避免修改原数据
        processed_df = df.copy()
        
        # 如果有ID列，保留它并确保数据类型正确
        if id_col and id_col in processed_df.columns:
            # 保留原始ID
            processed_df['translation_id'] = processed_df[id_col]
            self.logger.info(f"保留ID列: {id_col}")
        else:
            # 如果没有ID列，创建一个基于索引的ID
            processed_df['translation_id'] = processed_df.index + 1
            self.logger.info("创建基于索引的ID列")
        
        # 添加行索引（用于追踪原始位置）
        processed_df['original_index'] = processed_df.index
        
        # 清洗文本数据
        processed_df[source_col] = processed_df[source_col].apply(self.clean_text)
        processed_df[target_col] = processed_df[target_col].apply(self.clean_text)
        
        # 添加基础统计信息
        processed_df['source_length'] = processed_df[source_col].str.len()
        processed_df['target_length'] = processed_df[target_col].str.len()
        processed_df['length_ratio'] = processed_df['target_length'] / processed_df['source_length'].replace(0, 1)
        
        # 标记空值
        processed_df['has_empty_source'] = processed_df[source_col].str.len() == 0
        processed_df['has_empty_target'] = processed_df[target_col].str.len() == 0
        
        # 移除完全空白的行
        initial_count = len(processed_df)
        processed_df = processed_df[
            ~(processed_df['has_empty_source'] & processed_df['has_empty_target'])
        ].reset_index(drop=True)
        
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            self.logger.info(f"移除了 {removed_count} 行空白数据")
            
        self.logger.info(f"预处理完成，剩余 {len(processed_df)} 行有效数据")
        
        return processed_df
        
    def get_data_statistics(self, df: pd.DataFrame, source_col: str, target_col: str) -> Dict:
        """
        获取数据统计信息
        
        Args:
            df: 数据框
            source_col: 源语言列名
            target_col: 目标语言列名
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_rows': len(df),
            'empty_source': df[source_col].str.len().eq(0).sum(),
            'empty_target': df[target_col].str.len().eq(0).sum(),
            'source_length_stats': {
                'mean': df[source_col].str.len().mean(),
                'median': df[source_col].str.len().median(),
                'max': df[source_col].str.len().max(),
                'min': df[source_col].str.len().min()
            },
            'target_length_stats': {
                'mean': df[target_col].str.len().mean(),
                'median': df[target_col].str.len().median(),
                'max': df[target_col].str.len().max(),
                'min': df[target_col].str.len().min()
            }
        }
        
        if 'length_ratio' in df.columns:
            stats['length_ratio_stats'] = {
                'mean': df['length_ratio'].mean(),
                'median': df['length_ratio'].median(),
                'max': df['length_ratio'].max(),
                'min': df['length_ratio'].min()
            }
            
        return stats
        
    def process_file(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
        """
        处理单个文件的完整流程
        
        Args:
            file_path: 文件路径
            
        Returns:
            (处理后的数据框, 列映射信息)
        """
        # 加载文件
        df = self.load_excel_file(file_path)
        
        # 识别列
        column_mapping = self.identify_columns(df)
        
        # 预处理数据
        processed_df = self.preprocess_data(
            df, 
            column_mapping['source'], 
            column_mapping['target'],
            column_mapping.get('id')  # 传递ID列信息
        )
        
        # 获取统计信息
        stats = self.get_data_statistics(
            processed_df,
            column_mapping['source'],
            column_mapping['target']
        )
        
        self.logger.info("数据处理完成")
        self.logger.debug(f"数据统计: {stats}")
        
        return processed_df, column_mapping
