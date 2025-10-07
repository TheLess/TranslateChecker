"""
LLM翻译质量评估器
使用大语言模型进行深度的翻译质量分析
"""

import json
import re
from typing import Dict, List, Any, Optional
import pandas as pd
import openai
from tqdm import tqdm
import time

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger


class LLMEvaluator:
    """LLM翻译质量评估器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化LLM评估器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # 获取LLM配置
        self.llm_config = self.config.get('llm', {})
        self.provider = self.llm_config.get('provider', 'openai')
        
        # 初始化客户端
        self._init_client()
        
        # 获取提示词模板
        self.prompt_template = self.llm_config.get('prompt_template', self._get_default_prompt())
        
    def _init_client(self):
        """初始化LLM客户端"""
        if self.provider == 'openai':
            self._init_openai_client()
        else:
            self.logger.warning(f"不支持的LLM提供商: {self.provider}")
            self.client = None
            
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            # 从配置或环境变量获取API密钥
            api_key = self.llm_config.get('openai', {}).get('api_key')
            if not api_key:
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                
            if not api_key:
                self.logger.warning("未找到OpenAI API密钥，LLM评估功能将不可用")
                self.client = None
                return
                
            # 初始化客户端
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = self.llm_config.get('openai', {}).get('model', 'gpt-3.5-turbo')
            self.max_tokens = self.llm_config.get('openai', {}).get('max_tokens', 1000)
            self.temperature = self.llm_config.get('openai', {}).get('temperature', 0.1)
            
            self.logger.info(f"OpenAI客户端初始化成功，使用模型: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败: {e}")
            self.client = None
            
    def _get_default_prompt(self) -> str:
        """获取默认的评估提示词"""
        return """请评估以下中英文翻译的质量，从以下几个方面分析：
1. 准确性：翻译是否准确传达了原文意思
2. 流畅性：译文是否自然流畅
3. 一致性：专业术语和关键词是否一致
4. 完整性：是否有遗漏或添加内容

原文：{source}
译文：{target}

请按以下JSON格式返回评估结果：
{{
    "score": 评分(1-10),
    "accuracy": 准确性评分(1-10),
    "fluency": 流畅性评分(1-10),
    "consistency": 一致性评分(1-10),
    "completeness": 完整性评分(1-10),
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "explanation": "详细说明"
}}"""

    def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        调用LLM API
        
        Args:
            prompt: 提示词
            max_retries: 最大重试次数
            
        Returns:
            LLM响应文本
        """
        if self.client is None:
            return None
            
        for attempt in range(max_retries):
            try:
                if self.provider == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "你是一个专业的翻译质量评估专家。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                self.logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    
        return None
        
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的结果字典
        """
        if not response:
            return self._get_error_result("LLM无响应")
            
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ['score', 'accuracy', 'fluency', 'consistency', 'completeness']
                for field in required_fields:
                    if field not in result:
                        result[field] = 5  # 默认中等分数
                        
                # 确保分数在合理范围内
                for field in required_fields:
                    if not isinstance(result[field], (int, float)) or not (1 <= result[field] <= 10):
                        result[field] = 5
                        
                # 确保列表字段存在
                if 'issues' not in result or not isinstance(result['issues'], list):
                    result['issues'] = []
                if 'suggestions' not in result or not isinstance(result['suggestions'], list):
                    result['suggestions'] = []
                if 'explanation' not in result:
                    result['explanation'] = "无详细说明"
                    
                return result
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON解析失败: {e}")
            
        # 如果JSON解析失败，尝试从文本中提取信息
        return self._extract_from_text(response)
        
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        从纯文本中提取评估信息
        
        Args:
            text: 响应文本
            
        Returns:
            提取的结果字典
        """
        result = {
            'score': 5,
            'accuracy': 5,
            'fluency': 5,
            'consistency': 5,
            'completeness': 5,
            'issues': [],
            'suggestions': [],
            'explanation': text[:500]  # 截取前500字符作为说明
        }
        
        # 尝试提取分数
        score_patterns = [
            r'总分[：:]\s*(\d+)',
            r'评分[：:]\s*(\d+)',
            r'分数[：:]\s*(\d+)',
            r'(\d+)\s*分'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        result['score'] = score
                        break
                except ValueError:
                    continue
                    
        return result
        
    def _get_error_result(self, error_msg: str) -> Dict[str, Any]:
        """
        获取错误结果
        
        Args:
            error_msg: 错误信息
            
        Returns:
            错误结果字典
        """
        return {
            'score': 0,
            'accuracy': 0,
            'fluency': 0,
            'consistency': 0,
            'completeness': 0,
            'issues': [error_msg],
            'suggestions': [],
            'explanation': f"评估失败: {error_msg}"
        }
        
    def evaluate_single_pair(self, source: str, target: str) -> Dict[str, Any]:
        """
        评估单个翻译对
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            评估结果
        """
        if not source or not target:
            return self._get_error_result("源文本或目标文本为空")
            
        if self.client is None:
            return self._get_error_result("LLM客户端未初始化")
            
        # 构建提示词
        prompt = self.prompt_template.format(source=source, target=target)
        
        # 调用LLM
        response = self._call_llm(prompt)
        
        # 解析响应
        result = self._parse_llm_response(response)
        
        # 添加元数据
        result['llm_provider'] = self.provider
        result['llm_model'] = getattr(self, 'model_name', 'unknown')
        result['raw_response'] = response
        
        return result
        
    def evaluate_dataframe(self, df: pd.DataFrame, source_col: str, target_col: str, 
                          filter_condition: Optional[str] = None) -> pd.DataFrame:
        """
        批量评估数据框中的翻译对
        
        Args:
            df: 数据框
            source_col: 源语言列名
            target_col: 目标语言列名
            filter_condition: 过滤条件，只评估满足条件的行
            
        Returns:
            包含LLM评估结果的数据框
        """
        if self.client is None:
            self.logger.error("LLM客户端未初始化，跳过LLM评估")
            result_df = df.copy()
            result_df['llm_score'] = 0
            result_df['llm_evaluation'] = None
            return result_df
            
        # 应用过滤条件
        if filter_condition:
            eval_df = df.query(filter_condition).copy()
            self.logger.info(f"应用过滤条件 '{filter_condition}'，待评估数据: {len(eval_df)} 条")
        else:
            eval_df = df.copy()
            
        self.logger.info(f"开始LLM评估，共 {len(eval_df)} 条数据")
        
        results = []
        
        # 逐个评估（考虑到API限制）
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="LLM评估"):
            source = str(row[source_col]) if pd.notna(row[source_col]) else ""
            target = str(row[target_col]) if pd.notna(row[target_col]) else ""
            
            result = self.evaluate_single_pair(source, target)
            results.append(result)
            
            # 添加延迟以避免API限制
            time.sleep(0.5)
            
        # 将结果添加到数据框
        result_df = df.copy()
        
        # 初始化LLM评估列
        result_df['llm_score'] = 0
        result_df['llm_accuracy'] = 0
        result_df['llm_fluency'] = 0
        result_df['llm_consistency'] = 0
        result_df['llm_completeness'] = 0
        result_df['llm_issues'] = ""
        result_df['llm_suggestions'] = ""
        result_df['llm_explanation'] = ""
        result_df['llm_evaluation'] = None
        
        # 填入评估结果
        for i, (idx, result) in enumerate(zip(eval_df.index, results)):
            result_df.loc[idx, 'llm_score'] = result['score']
            result_df.loc[idx, 'llm_accuracy'] = result['accuracy']
            result_df.loc[idx, 'llm_fluency'] = result['fluency']
            result_df.loc[idx, 'llm_consistency'] = result['consistency']
            result_df.loc[idx, 'llm_completeness'] = result['completeness']
            result_df.loc[idx, 'llm_issues'] = '; '.join(result['issues'])
            result_df.loc[idx, 'llm_suggestions'] = '; '.join(result['suggestions'])
            result_df.loc[idx, 'llm_explanation'] = result['explanation']
            result_df.loc[idx, 'llm_evaluation'] = result
            
        # 统计结果
        evaluated_count = len(results)
        avg_score = sum(r['score'] for r in results) / evaluated_count if evaluated_count > 0 else 0
        
        self.logger.info(f"LLM评估完成，评估了 {evaluated_count} 条数据")
        self.logger.info(f"平均分数: {avg_score:.2f}")
        
        return result_df
        
    def get_evaluation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取评估结果摘要
        
        Args:
            df: 包含评估结果的数据框
            
        Returns:
            评估摘要
        """
        if 'llm_score' not in df.columns:
            return {'error': '数据框中没有LLM评估结果'}
            
        # 过滤出有评估结果的行
        evaluated_df = df[df['llm_score'] > 0]
        
        if len(evaluated_df) == 0:
            return {'error': '没有有效的评估结果'}
            
        summary = {
            'total_evaluated': len(evaluated_df),
            'average_scores': {
                'overall': evaluated_df['llm_score'].mean(),
                'accuracy': evaluated_df['llm_accuracy'].mean(),
                'fluency': evaluated_df['llm_fluency'].mean(),
                'consistency': evaluated_df['llm_consistency'].mean(),
                'completeness': evaluated_df['llm_completeness'].mean()
            },
            'score_distribution': {
                'excellent (9-10)': len(evaluated_df[evaluated_df['llm_score'] >= 9]),
                'good (7-8)': len(evaluated_df[(evaluated_df['llm_score'] >= 7) & (evaluated_df['llm_score'] < 9)]),
                'fair (5-6)': len(evaluated_df[(evaluated_df['llm_score'] >= 5) & (evaluated_df['llm_score'] < 7)]),
                'poor (1-4)': len(evaluated_df[evaluated_df['llm_score'] < 5])
            },
            'common_issues': self._analyze_common_issues(evaluated_df)
        }
        
        return summary
        
    def _analyze_common_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        分析常见问题
        
        Args:
            df: 评估结果数据框
            
        Returns:
            常见问题列表
        """
        issue_counts = {}
        
        for issues_str in df['llm_issues']:
            if pd.notna(issues_str) and issues_str:
                issues = [issue.strip() for issue in str(issues_str).split(';') if issue.strip()]
                for issue in issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    
        # 按频率排序
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'issue': issue, 'count': count, 'percentage': count/len(df)*100} 
                for issue, count in sorted_issues[:10]]  # 返回前10个常见问题
