"""
语义相似度模型
使用sentence-transformers计算中英文文本的语义相似度
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger


class SimilarityModel:
    """语义相似度模型"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化相似度模型
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # 获取配置
        self.similarity_config = self.config.get('detection.similarity', {})
        self.model_name = self.similarity_config.get(
            'model_name', 
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.threshold = self.similarity_config.get('threshold', 0.7)
        self.batch_size = self.similarity_config.get('batch_size', 32)
        
        # 初始化模型
        self.model = None
        self.device = None
        self._load_model()
        
    def _load_model(self):
        """加载sentence-transformers模型"""
        try:
            self.logger.info(f"正在加载模型: {self.model_name}")
            
            # 检测设备
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info("使用GPU加速")
            else:
                self.device = 'cpu'
                self.logger.info("使用CPU")
                
            # 加载模型
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.logger.warning("将使用备用模型")
            try:
                # 尝试使用更小的备用模型
                backup_model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
                self.model = SentenceTransformer(backup_model, device=self.device)
                self.logger.info(f"备用模型加载成功: {backup_model}")
            except Exception as e2:
                self.logger.error(f"备用模型也加载失败: {e2}")
                raise RuntimeError("无法加载任何相似度模型")
                
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            文本向量数组
        """
        if not texts:
            return np.array([])
            
        try:
            # 过滤空文本
            valid_texts = [text if text else "" for text in texts]
            
            # 批量编码
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"文本编码失败: {e}")
            raise
            
    def calculate_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        计算两组向量的余弦相似度
        
        Args:
            embeddings1: 第一组向量
            embeddings2: 第二组向量
            
        Returns:
            相似度数组
        """
        try:
            # 确保向量是二维的
            if embeddings1.ndim == 1:
                embeddings1 = embeddings1.reshape(1, -1)
            if embeddings2.ndim == 1:
                embeddings2 = embeddings2.reshape(1, -1)
                
            # 计算余弦相似度
            # 归一化向量
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # 避免除零
            norm1 = np.where(norm1 == 0, 1, norm1)
            norm2 = np.where(norm2 == 0, 1, norm2)
            
            embeddings1_normalized = embeddings1 / norm1
            embeddings2_normalized = embeddings2 / norm2
            
            # 计算点积
            similarities = np.sum(embeddings1_normalized * embeddings2_normalized, axis=1)
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            raise
            
    def check_similarity_pair(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查单个翻译对的相似度
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            相似度检查结果
        """
        if not source or not target:
            return {
                'passed': False,
                'similarity': 0.0,
                'issue': 'empty_text',
                'message': '源文本或目标文本为空'
            }
            
        try:
            # 编码文本
            source_embedding = self.encode_texts([source], show_progress=False)
            target_embedding = self.encode_texts([target], show_progress=False)
            
            # 计算相似度
            similarity = self.calculate_similarity(source_embedding, target_embedding)[0]
            
            # 判断是否通过
            passed = similarity >= self.threshold
            
            if passed:
                message = f'语义相似度正常 ({similarity:.3f})'
                issue = None
            else:
                message = f'语义相似度过低 ({similarity:.3f} < {self.threshold})'
                issue = 'low_similarity'
                
            return {
                'passed': passed,
                'similarity': float(similarity),
                'issue': issue,
                'message': message
            }
            
        except Exception as e:
            self.logger.error(f"相似度检查失败: {e}")
            return {
                'passed': False,
                'similarity': 0.0,
                'issue': 'calculation_error',
                'message': f'相似度计算错误: {str(e)}'
            }
            
    def check_dataframe(self, df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
        """
        批量检查数据框中的翻译对相似度
        
        Args:
            df: 数据框
            source_col: 源语言列名
            target_col: 目标语言列名
            
        Returns:
            包含相似度检查结果的数据框
        """
        self.logger.info(f"开始语义相似度检查，共 {len(df)} 条数据")
        
        # 准备文本数据
        source_texts = df[source_col].fillna('').astype(str).tolist()
        target_texts = df[target_col].fillna('').astype(str).tolist()
        
        try:
            # 批量编码
            self.logger.info("正在编码源文本...")
            source_embeddings = self.encode_texts(source_texts)
            
            self.logger.info("正在编码目标文本...")
            target_embeddings = self.encode_texts(target_texts)
            
            # 计算相似度
            self.logger.info("正在计算相似度...")
            similarities = self.calculate_similarity(source_embeddings, target_embeddings)
            
            # 生成结果
            results = []
            for i, similarity in enumerate(similarities):
                passed = similarity >= self.threshold
                
                if source_texts[i] == '' or target_texts[i] == '':
                    issue = 'empty_text'
                    message = '源文本或目标文本为空'
                    passed = False
                elif passed:
                    issue = None
                    message = f'语义相似度正常 ({similarity:.3f})'
                else:
                    issue = 'low_similarity'
                    message = f'语义相似度过低 ({similarity:.3f} < {self.threshold})'
                    
                results.append({
                    'passed': passed,
                    'similarity': float(similarity),
                    'issue': issue,
                    'message': message
                })
                
            # 添加结果到数据框
            result_df = df.copy()
            result_df['similarity_score'] = [r['similarity'] for r in results]
            result_df['similarity_passed'] = [r['passed'] for r in results]
            result_df['similarity_issue'] = [r['issue'] for r in results]
            result_df['similarity_message'] = [r['message'] for r in results]
            
            # 统计结果
            passed_count = sum(r['passed'] for r in results)
            avg_similarity = np.mean([r['similarity'] for r in results])
            
            self.logger.info(f"相似度检查完成")
            self.logger.info(f"通过率: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")
            self.logger.info(f"平均相似度: {avg_similarity:.3f}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"批量相似度检查失败: {e}")
            # 返回带有错误信息的数据框
            result_df = df.copy()
            result_df['similarity_score'] = 0.0
            result_df['similarity_passed'] = False
            result_df['similarity_issue'] = 'calculation_error'
            result_df['similarity_message'] = f'相似度计算错误: {str(e)}'
            return result_df
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {'status': 'not_loaded'}
            
        return {
            'status': 'loaded',
            'model_name': self.model_name,
            'device': self.device,
            'threshold': self.threshold,
            'batch_size': self.batch_size,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown')
        }
        
    def update_threshold(self, new_threshold: float):
        """
        更新相似度阈值
        
        Args:
            new_threshold: 新的阈值
        """
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
            self.logger.info(f"相似度阈值已更新为: {new_threshold}")
        else:
            raise ValueError("阈值必须在0-1之间")
            
    def benchmark_model(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        对模型进行基准测试
        
        Args:
            test_pairs: 测试文本对列表
            
        Returns:
            基准测试结果
        """
        if not test_pairs:
            return {'error': '无测试数据'}
            
        self.logger.info(f"开始基准测试，共 {len(test_pairs)} 个测试对")
        
        similarities = []
        processing_times = []
        
        import time
        
        for source, target in test_pairs:
            start_time = time.time()
            result = self.check_similarity_pair(source, target)
            end_time = time.time()
            
            similarities.append(result['similarity'])
            processing_times.append(end_time - start_time)
            
        return {
            'test_count': len(test_pairs),
            'avg_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'avg_processing_time': np.mean(processing_times),
            'total_time': sum(processing_times)
        }
