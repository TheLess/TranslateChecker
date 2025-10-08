"""
翻译检查器主类
整合所有检测模块，提供统一的检查接口
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime

from ..processors.data_processor import DataProcessor
from ..core.rule_checker import RuleChecker
from ..models.similarity_model import SimilarityModel
from ..models.llm_evaluator import LLMEvaluator
from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger, init_default_logger


class TranslationChecker:
    """翻译检查器主类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化翻译检查器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path)
        
        # 初始化日志
        log_config = self.config.get('logging', {})
        init_default_logger(log_config)
        self.logger = get_logger(__name__)
        
        self.logger.info("正在初始化翻译检查器...")
        
        # 初始化各个模块
        self.data_processor = DataProcessor(config_path)
        self.rule_checker = RuleChecker(config_path)
        self.similarity_model = None
        self.llm_evaluator = None
        
        # 检查结果存储
        self.last_results = None
        self.processing_stats = {}
        
        self.logger.info("翻译检查器初始化完成")
        
    def _init_similarity_model(self):
        """延迟初始化相似度模型"""
        if self.similarity_model is None:
            try:
                self.logger.info("正在初始化相似度模型...")
                self.similarity_model = SimilarityModel(self.config.config_path)
                self.logger.info("相似度模型初始化成功")
            except Exception as e:
                self.logger.error(f"相似度模型初始化失败: {e}")
                self.similarity_model = None
                
    def _init_llm_evaluator(self):
        """延迟初始化LLM评估器"""
        if self.llm_evaluator is None:
            try:
                self.logger.info("正在初始化LLM评估器...")
                self.llm_evaluator = LLMEvaluator(self.config.config_path)
                self.logger.info("LLM评估器初始化成功")
            except Exception as e:
                self.logger.error(f"LLM评估器初始化失败: {e}")
                self.llm_evaluator = None
                
    def check_file(self, 
                   file_path: Union[str, Path],
                   enable_similarity: bool = True,
                   enable_llm: bool = False,
                   llm_filter: Optional[str] = None) -> pd.DataFrame:
        """
        检查单个翻译文件
        
        Args:
            file_path: 文件路径
            enable_similarity: 是否启用相似度检测
            enable_llm: 是否启用LLM评估
            llm_filter: LLM评估的过滤条件
            
        Returns:
            检查结果数据框
        """
        start_time = time.time()
        self.logger.info(f"开始检查文件: {file_path}")
        
        try:
            # 1. 数据处理
            self.logger.info("步骤 1/4: 数据处理")
            df, column_mapping = self.data_processor.process_file(file_path)
            source_col = column_mapping['source']
            target_col = column_mapping['target']
            
            # 2. 基础规则检测
            self.logger.info("步骤 2/4: 基础规则检测")
            df = self.rule_checker.check_dataframe(df, source_col, target_col)
            
            # 3. 语义相似度检测
            if enable_similarity:
                self.logger.info("步骤 3/4: 语义相似度检测")
                self._init_similarity_model()
                if self.similarity_model:
                    df = self.similarity_model.check_dataframe(df, source_col, target_col)
                else:
                    self.logger.warning("相似度模型不可用，跳过相似度检测")
            else:
                self.logger.info("步骤 3/4: 跳过语义相似度检测")
                
            # 4. LLM质量评估
            if enable_llm:
                self.logger.info("步骤 4/4: LLM质量评估")
                self._init_llm_evaluator()
                if self.llm_evaluator:
                    df = self.llm_evaluator.evaluate_dataframe(df, source_col, target_col, llm_filter)
                else:
                    self.logger.warning("LLM评估器不可用，跳过LLM评估")
            else:
                self.logger.info("步骤 4/4: 跳过LLM质量评估")
                
            # 计算综合评分
            df = self._calculate_overall_score(df, enable_similarity, enable_llm)
            
            # 记录处理统计
            end_time = time.time()
            self.processing_stats = {
                'file_path': str(file_path),
                'total_rows': len(df),
                'processing_time': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'column_mapping': column_mapping,
                'enabled_features': {
                    'rules': True,
                    'similarity': enable_similarity and self.similarity_model is not None,
                    'llm': enable_llm and self.llm_evaluator is not None
                }
            }
            
            self.last_results = df
            self.logger.info(f"文件检查完成，耗时 {end_time - start_time:.2f} 秒")
            
            return df
            
        except Exception as e:
            self.logger.error(f"文件检查失败: {e}")
            raise
            
    def _calculate_overall_score(self, df: pd.DataFrame, 
                               has_similarity: bool, 
                               has_llm: bool) -> pd.DataFrame:
        """
        计算综合评分
        
        Args:
            df: 数据框
            has_similarity: 是否有相似度检测结果
            has_llm: 是否有LLM评估结果
            
        Returns:
            包含综合评分的数据框
        """
        # 初始化综合评分
        df['overall_score'] = 0.0
        df['overall_status'] = 'unknown'
        df['overall_issues'] = ''
        
        for idx, row in df.iterrows():
            score_components = []
            issues = []
            
            # 规则检测分数 (权重: 30%)
            if row.get('rule_check_passed', False):
                rule_score = 10
            else:
                rule_score = 0
                if row.get('rule_check_issues'):
                    issues.append(f"规则问题: {row['rule_check_issues']}")
            score_components.append(('rules', rule_score, 0.3))
            
            # 相似度分数 (权重: 40%)
            if has_similarity and 'similarity_score' in df.columns:
                similarity_score = row.get('similarity_score', 0) * 10  # 转换为10分制
                if not row.get('similarity_passed', False):
                    issues.append(f"相似度问题: {row.get('similarity_message', '')}")
            else:
                similarity_score = 5  # 默认中等分数
            score_components.append(('similarity', similarity_score, 0.4))
            
            # LLM评估分数 (权重: 30%)
            if has_llm and 'llm_score' in df.columns:
                llm_score = row.get('llm_score', 0)
                if llm_score < 7 and row.get('llm_issues'):
                    issues.append(f"LLM问题: {row['llm_issues']}")
            else:
                llm_score = 5  # 默认中等分数
            score_components.append(('llm', llm_score, 0.3))
            
            # 计算加权平均分
            total_score = sum(score * weight for _, score, weight in score_components)
            df.loc[idx, 'overall_score'] = round(total_score, 2)
            
            # 确定状态
            if total_score >= 8:
                status = 'excellent'
            elif total_score >= 6:
                status = 'good'
            elif total_score >= 4:
                status = 'fair'
            else:
                status = 'poor'
            df.loc[idx, 'overall_status'] = status
            
            # 合并问题描述
            df.loc[idx, 'overall_issues'] = '; '.join(issues)
            
        return df
        
    def get_summary_report(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        生成检查结果摘要报告
        
        Args:
            df: 数据框，如果为None则使用最后的检查结果
            
        Returns:
            摘要报告字典
        """
        if df is None:
            df = self.last_results
            
        if df is None:
            return {'error': '没有可用的检查结果'}
            
        total_count = len(df)
        
        # 基础统计
        summary = {
            'basic_stats': {
                'total_translations': total_count,
                'processing_time': self.processing_stats.get('processing_time', 0),
                'file_path': self.processing_stats.get('file_path', ''),
                'timestamp': self.processing_stats.get('timestamp', '')
            }
        }
        
        # 规则检测统计
        if 'rule_check_passed' in df.columns:
            rule_passed = df['rule_check_passed'].sum()
            summary['rule_check'] = {
                'passed': rule_passed,
                'failed': total_count - rule_passed,
                'pass_rate': f"{rule_passed/total_count*100:.1f}%"
            }
            
        # 相似度检测统计
        if 'similarity_score' in df.columns:
            similarity_scores = df['similarity_score'].dropna()
            summary['similarity_check'] = {
                'average_score': f"{similarity_scores.mean():.3f}",
                'min_score': f"{similarity_scores.min():.3f}",
                'max_score': f"{similarity_scores.max():.3f}",
                'passed': df['similarity_passed'].sum() if 'similarity_passed' in df.columns else 0
            }
            
        # LLM评估统计
        if 'llm_score' in df.columns:
            llm_scores = df[df['llm_score'] > 0]['llm_score']
            if len(llm_scores) > 0:
                summary['llm_evaluation'] = {
                    'evaluated_count': len(llm_scores),
                    'average_score': f"{llm_scores.mean():.2f}",
                    'score_distribution': {
                        'excellent (9-10)': len(llm_scores[llm_scores >= 9]),
                        'good (7-8)': len(llm_scores[(llm_scores >= 7) & (llm_scores < 9)]),
                        'fair (5-6)': len(llm_scores[(llm_scores >= 5) & (llm_scores < 7)]),
                        'poor (1-4)': len(llm_scores[llm_scores < 5])
                    }
                }
                
        # 综合评分统计
        if 'overall_score' in df.columns:
            status_counts = df['overall_status'].value_counts().to_dict()
            summary['overall_assessment'] = {
                'average_score': f"{df['overall_score'].mean():.2f}",
                'status_distribution': status_counts,
                'recommendations': self._generate_recommendations(df)
            }
            
        return summary
        
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """
        基于检查结果生成改进建议
        
        Args:
            df: 检查结果数据框
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 分析常见问题
        if 'overall_issues' in df.columns:
            all_issues = []
            for issues_str in df['overall_issues']:
                if pd.notna(issues_str) and issues_str:
                    all_issues.extend([issue.strip() for issue in str(issues_str).split(';')])
                    
            # 统计问题频率
            issue_counts = {}
            for issue in all_issues:
                if issue:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    
            # 生成针对性建议
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            
            for issue, count in sorted_issues[:5]:  # 前5个最常见问题
                if 'length' in issue.lower():
                    recommendations.append("建议检查翻译长度的合理性，避免过长或过短的翻译")
                elif 'similarity' in issue.lower():
                    recommendations.append("建议提高翻译的语义准确性，确保与原文意思一致")
                elif 'terminology' in issue.lower():
                    recommendations.append("建议统一专业术语的翻译，保持术语一致性")
                elif 'empty' in issue.lower():
                    recommendations.append("建议检查并补充缺失的翻译内容")
                    
        # 基于综合评分的建议
        if 'overall_score' in df.columns:
            avg_score = df['overall_score'].mean()
            if avg_score < 5:
                recommendations.append("整体翻译质量较低，建议进行全面的质量提升")
            elif avg_score < 7:
                recommendations.append("翻译质量中等，建议重点关注低分项目的改进")
            else:
                recommendations.append("翻译质量良好，建议保持现有标准并持续优化")
                
        return recommendations[:10]  # 最多返回10条建议
        
    def export_results(self, 
                      output_path: Union[str, Path],
                      df: Optional[pd.DataFrame] = None,
                      format_type: str = 'excel') -> str:
        """
        导出检查结果
        
        Args:
            output_path: 输出路径
            df: 数据框，如果为None则使用最后的检查结果
            format_type: 输出格式 ('excel', 'csv', 'json')
            
        Returns:
            实际输出文件路径
        """
        if df is None:
            df = self.last_results
            
        if df is None:
            raise ValueError("没有可导出的检查结果")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == 'excel':
            if not str(output_path).endswith('.xlsx'):
                output_path = output_path.with_suffix('.xlsx')
            df.to_excel(output_path, index=False)
        elif format_type.lower() == 'csv':
            if not str(output_path).endswith('.csv'):
                output_path = output_path.with_suffix('.csv')
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif format_type.lower() == 'json':
            if not str(output_path).endswith('.json'):
                output_path = output_path.with_suffix('.json')
            df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的输出格式: {format_type}")
            
        self.logger.info(f"结果已导出到: {output_path}")
        return str(output_path)
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        status = {
            'data_processor': 'ready',
            'rule_checker': 'ready',
            'similarity_model': 'lazy_load',  # 延迟加载
            'llm_evaluator': 'lazy_load'      # 延迟加载
        }
        
        # 检查相似度模型状态
        if self.similarity_model:
            model_info = self.similarity_model.get_model_info()
            status['similarity_model'] = model_info.get('status', 'loaded')
        else:
            # 尝试检测是否可以加载
            try:
                import sentence_transformers
                status['similarity_model'] = 'ready'
            except ImportError:
                status['similarity_model'] = 'missing_deps'
            
        # 检查LLM评估器状态  
        if self.llm_evaluator:
            status['llm_evaluator'] = 'ready' if self.llm_evaluator.client else 'no_api_key'
        else:
            # 根据配置检查LLM状态
            llm_config = self.config.get('llm', {})
            provider = llm_config.get('provider', 'openai')
            
            if provider == 'ollama':
                try:
                    import requests
                    base_url = llm_config.get('ollama', {}).get('base_url', 'http://localhost:11434')
                    response = requests.get(f"{base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        status['llm_evaluator'] = 'ollama_ready'
                    else:
                        status['llm_evaluator'] = 'ollama_offline'
                except:
                    status['llm_evaluator'] = 'ollama_offline'
            elif provider in ['qwen', 'glm', 'wenxin']:
                status['llm_evaluator'] = 'api_ready'
            elif provider == 'openai':
                import os
                if os.getenv('OPENAI_API_KEY') or llm_config.get('openai', {}).get('api_key'):
                    status['llm_evaluator'] = 'api_ready'
                else:
                    status['llm_evaluator'] = 'no_api_key'
            
        return status
