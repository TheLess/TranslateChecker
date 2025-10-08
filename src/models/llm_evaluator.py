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
        elif self.provider == 'ollama':
            self._init_ollama_client()
        elif self.provider == 'qwen':
            self._init_qwen_client()
        elif self.provider == 'glm':
            self._init_glm_client()
        elif self.provider == 'wenxin':
            self._init_wenxin_client()
        else:
            supported_providers = ['openai', 'ollama', 'qwen', 'glm', 'wenxin']
            self.logger.warning(f"不支持的LLM提供商: {self.provider}")
            self.logger.info(f"支持的提供商: {', '.join(supported_providers)}")
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
    
    def _init_ollama_client(self):
        """初始化Ollama客户端"""
        try:
            import requests
            
            # 获取Ollama配置
            ollama_config = self.llm_config.get('ollama', {})
            self.base_url = ollama_config.get('base_url', 'http://localhost:11434')
            self.model_name = ollama_config.get('model', 'qwen2.5:7b')
            self.max_tokens = ollama_config.get('max_tokens', 1000)
            self.temperature = ollama_config.get('temperature', 0.1)
            
            # 测试连接
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names:
                    self.client = 'ollama'  # 标记客户端类型
                    self.logger.info(f"Ollama客户端初始化成功，使用模型: {self.model_name}")
                else:
                    self.logger.warning(f"模型 {self.model_name} 未找到，可用模型: {model_names}")
                    self.logger.info(f"请运行: ollama pull {self.model_name}")
                    self.client = None
            else:
                self.logger.warning("无法连接到Ollama服务，请确保Ollama已启动")
                self.client = None
                
        except ImportError:
            self.logger.error("缺少requests库，请安装: pip install requests")
            self.client = None
        except Exception as e:
            self.logger.error(f"Ollama客户端初始化失败: {e}")
            self.logger.info("请确保Ollama服务正在运行: ollama serve")
            self.client = None
    
    def _init_qwen_client(self):
        """初始化通义千问客户端"""
        try:
            # 获取通义千问配置
            qwen_config = self.llm_config.get('qwen', {})
            api_key = qwen_config.get('api_key')
            
            if not api_key:
                import os
                api_key = os.getenv('DASHSCOPE_API_KEY')
                
            if not api_key:
                self.logger.warning("未找到通义千问API密钥，请设置DASHSCOPE_API_KEY环境变量")
                self.client = None
                return
            
            # 设置客户端参数
            self.api_key = api_key
            self.model_name = qwen_config.get('model', 'qwen-turbo')
            self.max_tokens = qwen_config.get('max_tokens', 1000)
            self.temperature = qwen_config.get('temperature', 0.1)
            self.client = 'qwen'  # 标记客户端类型
            
            self.logger.info(f"通义千问客户端初始化成功，使用模型: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"通义千问客户端初始化失败: {e}")
            self.client = None
    
    def _init_glm_client(self):
        """初始化智谱AI客户端"""
        try:
            # 获取智谱AI配置
            glm_config = self.llm_config.get('glm', {})
            api_key = glm_config.get('api_key')
            
            if not api_key:
                import os
                api_key = os.getenv('GLM_API_KEY')
                
            if not api_key:
                self.logger.warning("未找到智谱AI API密钥，请设置GLM_API_KEY环境变量")
                self.client = None
                return
            
            # 设置客户端参数
            self.api_key = api_key
            self.model_name = glm_config.get('model', 'glm-4-flash')
            self.max_tokens = glm_config.get('max_tokens', 1000)
            self.temperature = glm_config.get('temperature', 0.1)
            self.client = 'glm'  # 标记客户端类型
            
            self.logger.info(f"智谱AI客户端初始化成功，使用模型: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"智谱AI客户端初始化失败: {e}")
            self.client = None
    
    def _init_wenxin_client(self):
        """初始化百度文心一言客户端"""
        try:
            # 获取文心一言配置
            wenxin_config = self.llm_config.get('wenxin', {})
            api_key = wenxin_config.get('api_key')
            secret_key = wenxin_config.get('secret_key')
            
            if not api_key or not secret_key:
                import os
                api_key = api_key or os.getenv('WENXIN_API_KEY')
                secret_key = secret_key or os.getenv('WENXIN_SECRET_KEY')
                
            if not api_key or not secret_key:
                self.logger.warning("未找到文心一言API密钥，请设置WENXIN_API_KEY和WENXIN_SECRET_KEY环境变量")
                self.client = None
                return
            
            # 设置客户端参数
            self.api_key = api_key
            self.secret_key = secret_key
            self.model_name = wenxin_config.get('model', 'ernie-3.5-turbo')
            self.max_tokens = wenxin_config.get('max_tokens', 1000)
            self.temperature = wenxin_config.get('temperature', 0.1)
            self.client = 'wenxin'  # 标记客户端类型
            
            self.logger.info(f"文心一言客户端初始化成功，使用模型: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"文心一言客户端初始化失败: {e}")
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
                
                elif self.provider == 'ollama':
                    return self._call_ollama(prompt)
                
                elif self.provider == 'qwen':
                    return self._call_qwen(prompt)
                
                elif self.provider == 'glm':
                    return self._call_glm(prompt)
                
                elif self.provider == 'wenxin':
                    return self._call_wenxin(prompt)
                    
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
            # 尝试提取JSON部分 - 改进的正则表达式
            # 优先匹配完整的JSON对象
            json_patterns = [
                r'\{[^{}]*"score"[^{}]*\}',  # 简单JSON
                r'\{(?:[^{}]|{[^{}]*})*\}',  # 嵌套JSON
                r'\{.*?\}(?=\s*$|\s*\n|\s*[。！？.])',  # 以标点结尾的JSON
                r'\{.*\}'  # 兜底匹配
            ]
            
            json_str = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        # 尝试解析，如果成功就使用这个
                        test_result = json.loads(json_str)
                        if 'score' in test_result:  # 确保包含关键字段
                            break
                    except:
                        continue
            
            if json_str:
                result = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ['score', 'accuracy', 'fluency', 'consistency', 'completeness']
                missing_fields = []
                invalid_fields = []
                
                for field in required_fields:
                    if field not in result:
                        missing_fields.append(field)
                        result[field] = 5  # 默认中等分数
                        
                # 确保分数在合理范围内
                for field in required_fields:
                    if not isinstance(result[field], (int, float)) or not (1 <= result[field] <= 10):
                        invalid_fields.append(f"{field}={result.get(field)}")
                        result[field] = 5
                
                # 记录异常情况
                if missing_fields:
                    self.logger.warning(f"LLM响应缺少字段: {missing_fields}，已填充默认值5分")
                if invalid_fields:
                    self.logger.warning(f"LLM响应字段值无效: {invalid_fields}，已修正为5分")
                    
                # 添加质量标记
                result['_parse_quality'] = {
                    'missing_fields': missing_fields,
                    'invalid_fields': invalid_fields,
                    'has_issues': len(missing_fields) > 0 or len(invalid_fields) > 0
                }
                        
                # 确保列表字段存在
                if 'issues' not in result or not isinstance(result['issues'], list):
                    result['issues'] = []
                if 'suggestions' not in result or not isinstance(result['suggestions'], list):
                    result['suggestions'] = []
                if 'explanation' not in result:
                    result['explanation'] = "无详细说明"
                    
                return result
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            self.logger.error(f"错误位置: 第{e.lineno}行, 第{e.colno}列")
            self.logger.error(f"原始响应全文:\n{'-'*50}\n{response}\n{'-'*50}")
            if json_str:
                self.logger.error(f"提取的JSON字符串:\n{json_str}")
                # 尝试找出具体的问题字符
                try:
                    problem_char = json_str[e.pos] if e.pos < len(json_str) else '(超出范围)'
                    self.logger.error(f"问题字符位置 {e.pos}: '{problem_char}' (ASCII: {ord(problem_char) if len(problem_char)==1 else 'N/A'})")
                except:
                    pass
            else:
                self.logger.error("未能从响应中提取到JSON格式")
                # 显示响应中是否包含大括号
                if '{' in response and '}' in response:
                    self.logger.error("响应中包含大括号，但正则表达式未能匹配")
                else:
                    self.logger.error("响应中不包含JSON大括号")
            
        # 如果JSON解析失败，尝试从文本中提取信息
        self.logger.warning("JSON解析完全失败，降级到文本解析模式")
        fallback_result = self._extract_from_text(response)
        
        # 标记这是降级结果
        fallback_result['_parse_quality'] = {
            'missing_fields': ['所有字段'],
            'invalid_fields': [],
            'has_issues': True,
            'fallback_mode': True,
            'original_response': response[:500]  # 保存原始响应用于调试
        }
        
        return fallback_result
        
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
            
        self.logger.info(f"🔄 开始LLM评估，共 {len(eval_df)} 条数据")
        
        results = []
        
        # 逐个评估（考虑到API限制）
        for i, (idx, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="LLM评估", leave=False)):
            source = str(row[source_col]) if pd.notna(row[source_col]) else ""
            target = str(row[target_col]) if pd.notna(row[target_col]) else ""
            
            self.logger.debug(f"正在评估第 {i+1}/{len(eval_df)} 条数据")
            result = self.evaluate_single_pair(source, target)
            results.append(result)
            
            # 添加延迟以避免API限制（减少延迟提升性能）
            time.sleep(0.1)
        
        self.logger.info(f"✓ LLM评估完成，共处理 {len(results)} 条数据")
            
        # 将结果添加到数据框
        result_df = df.copy()
        
        # 初始化LLM评估列 - 使用正确的数据类型
        result_df['llm_score'] = 0.0  # float类型
        result_df['llm_accuracy'] = 0.0
        result_df['llm_fluency'] = 0.0
        result_df['llm_consistency'] = 0.0
        result_df['llm_completeness'] = 0.0
        result_df['llm_issues'] = ""  # string类型
        result_df['llm_suggestions'] = ""
        result_df['llm_explanation'] = ""
        result_df['llm_evaluation'] = pd.Series([None] * len(result_df), dtype=object)  # 明确指定object类型
        
        # 填入评估结果 - 使用更安全的方法
        for i, (idx, result) in enumerate(zip(eval_df.index, results)):
            try:
                # 确保索引存在于result_df中
                if idx not in result_df.index:
                    self.logger.warning(f"索引{idx}不存在于result_df中，跳过")
                    continue
                    
                # 安全地填充数值字段
                result_df.at[idx, 'llm_score'] = float(result.get('score', 0))
                result_df.at[idx, 'llm_accuracy'] = float(result.get('accuracy', 0))
                result_df.at[idx, 'llm_fluency'] = float(result.get('fluency', 0))
                result_df.at[idx, 'llm_consistency'] = float(result.get('consistency', 0))
                result_df.at[idx, 'llm_completeness'] = float(result.get('completeness', 0))
                
                # 安全地填充字符串字段
                issues = result.get('issues', [])
                result_df.at[idx, 'llm_issues'] = '; '.join(issues) if isinstance(issues, list) else str(issues)
                
                suggestions = result.get('suggestions', [])
                result_df.at[idx, 'llm_suggestions'] = '; '.join(suggestions) if isinstance(suggestions, list) else str(suggestions)
                
                result_df.at[idx, 'llm_explanation'] = str(result.get('explanation', ''))
                
                # 填充复杂对象字段
                result_df.at[idx, 'llm_evaluation'] = result
                
            except Exception as e:
                self.logger.error(f"填充LLM结果时出错，索引{idx}: {e}")
                self.logger.error(f"DataFrame索引范围: {result_df.index.min()} - {result_df.index.max()}")
                self.logger.error(f"当前索引: {idx}, 类型: {type(idx)}")
                # 使用默认值
                if idx in result_df.index:
                    result_df.at[idx, 'llm_score'] = 0.0
                    result_df.at[idx, 'llm_accuracy'] = 0.0
                    result_df.at[idx, 'llm_fluency'] = 0.0
                    result_df.at[idx, 'llm_consistency'] = 0.0
                    result_df.at[idx, 'llm_completeness'] = 0.0
                    result_df.at[idx, 'llm_issues'] = "数据处理错误"
                    result_df.at[idx, 'llm_suggestions'] = ""
                    result_df.at[idx, 'llm_explanation'] = f"处理错误: {str(e)}"
                    result_df.at[idx, 'llm_evaluation'] = None
            
        # 统计结果和质量分析
        evaluated_count = len(results)
        avg_score = sum(r['score'] for r in results) / evaluated_count if evaluated_count > 0 else 0
        
        # 统计解析质量
        parse_issues = 0
        fallback_count = 0
        missing_fields_count = 0
        invalid_fields_count = 0
        
        for result in results:
            if '_parse_quality' in result and result['_parse_quality']['has_issues']:
                parse_issues += 1
                if result['_parse_quality'].get('fallback_mode'):
                    fallback_count += 1
                if result['_parse_quality']['missing_fields']:
                    missing_fields_count += 1
                if result['_parse_quality']['invalid_fields']:
                    invalid_fields_count += 1
        
        self.logger.info(f"LLM评估完成，评估了 {evaluated_count} 条数据")
        self.logger.info(f"平均分数: {avg_score:.2f}")
        
        # 质量报告
        if parse_issues > 0:
            self.logger.warning(f"解析质量报告:")
            self.logger.warning(f"  - 有解析问题的条目: {parse_issues}/{evaluated_count} ({parse_issues/evaluated_count*100:.1f}%)")
            if fallback_count > 0:
                self.logger.warning(f"  - 降级到文本解析: {fallback_count} 条")
            if missing_fields_count > 0:
                self.logger.warning(f"  - 缺少字段: {missing_fields_count} 条")
            if invalid_fields_count > 0:
                self.logger.warning(f"  - 字段值无效: {invalid_fields_count} 条")
            self.logger.warning(f"  建议检查提示词模板或模型配置")
        else:
            self.logger.info("所有LLM响应解析正常 ✓")
        
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
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """调用Ollama API"""
        try:
            import requests
            
            data = {
                "model": self.model_name,
                "prompt": f"你是一个专业的翻译质量评估专家。\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=120  # 增加到120秒
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                self.logger.error(f"Ollama API调用失败: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Ollama调用异常: {e}")
            return None
    
    def _call_qwen(self, prompt: str) -> Optional[str]:
        """调用通义千问API"""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {"role": "system", "content": "你是一个专业的翻译质量评估专家。"},
                        {"role": "user", "content": prompt}
                    ]
                },
                "parameters": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            }
            
            response = requests.post(
                'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('output') and result['output'].get('choices'):
                    return result['output']['choices'][0]['message']['content']
            else:
                self.logger.error(f"通义千问API调用失败: {response.status_code}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"通义千问调用异常: {e}")
            return None
    
    def _call_glm(self, prompt: str) -> Optional[str]:
        """调用智谱AI API"""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "你是一个专业的翻译质量评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                'https://open.bigmodel.cn/api/paas/v4/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('choices'):
                    return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"智谱AI API调用失败: {response.status_code}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"智谱AI调用异常: {e}")
            return None
    
    def _call_wenxin(self, prompt: str) -> Optional[str]:
        """调用百度文心一言API"""
        try:
            import requests
            
            # 首先获取access_token
            token_url = "https://aip.baidubce.com/oauth/2.0/token"
            token_params = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key
            }
            
            token_response = requests.post(token_url, params=token_params, timeout=30)
            if token_response.status_code != 200:
                self.logger.error("获取文心一言access_token失败")
                return None
                
            access_token = token_response.json().get('access_token')
            if not access_token:
                self.logger.error("文心一言access_token为空")
                return None
            
            # 调用文心一言API
            api_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-turbo"
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "messages": [
                    {"role": "user", "content": f"你是一个专业的翻译质量评估专家。\n\n{prompt}"}
                ],
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                f"{api_url}?access_token={access_token}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('result', '')
            else:
                self.logger.error(f"文心一言API调用失败: {response.status_code}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"文心一言调用异常: {e}")
            return None
