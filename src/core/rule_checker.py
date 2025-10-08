"""
基础规则检测器
负责基本的翻译质量规则检查
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger


class RuleChecker:
    """基础规则检测器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化规则检测器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # 加载规则配置
        self.rules_config = self.config.get('detection.rules', {})
        self.keywords_config = self.config.get('detection.keywords', {})
        
        # 加载关键词字典
        self.keywords_dict = self._load_keywords_dict()
        
    def _load_keywords_dict(self) -> Dict:
        """加载关键词字典"""
        keywords_path = self.keywords_config.get('dictionary_path', 'config/keywords.yaml')
        keywords_path = Path(keywords_path)
        
        if not keywords_path.exists():
            self.logger.warning(f"关键词字典文件不存在: {keywords_path}")
            return {}
            
        try:
            with open(keywords_path, 'r', encoding='utf-8') as f:
                keywords_dict = yaml.safe_load(f) or {}
                
            # 加载术语映射
            terminology_config = keywords_dict.get('terminology', {})
            if terminology_config.get('source_type') == 'file':
                keywords_dict['terminology'] = self._load_terminology_from_excel(terminology_config)
            else:
                # 使用内联术语定义
                keywords_dict['terminology'] = terminology_config.get('inline_terms', {})
                
            return keywords_dict
        except Exception as e:
            self.logger.error(f"加载关键词字典失败: {e}")
            return {}
    
    def _load_terminology_from_excel(self, terminology_config: Dict) -> Dict:
        """从Excel文件加载术语映射"""
        try:
            excel_file = terminology_config.get('excel_file', 'config/terminology.xlsx')
            sheet_name = terminology_config.get('sheet_name', 'terminology')
            columns = terminology_config.get('columns', {})
            
            # 获取列名配置
            chinese_col = columns.get('chinese', '中文术语')
            english_col = columns.get('english', '英文术语')
            alternatives_col = columns.get('alternatives', '备选翻译')
            
            excel_path = Path(excel_file)
            if not excel_path.exists():
                self.logger.warning(f"术语Excel文件不存在: {excel_path}")
                return {}
            
            # 读取Excel文件
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # 检查必需的列是否存在
            if chinese_col not in df.columns or english_col not in df.columns:
                self.logger.error(f"Excel文件缺少必需的列: {chinese_col} 或 {english_col}")
                return {}
            
            terminology = {}
            
            for _, row in df.iterrows():
                chinese_term = str(row[chinese_col]).strip()
                english_term = str(row[english_col]).strip()
                
                if not chinese_term or not english_term or chinese_term == 'nan' or english_term == 'nan':
                    continue
                
                # 基础英文翻译
                english_list = [english_term]
                
                # 添加备选翻译
                if alternatives_col in df.columns:
                    alternatives = str(row[alternatives_col]).strip()
                    if alternatives and alternatives != 'nan':
                        # 支持多种分隔符
                        alt_list = [alt.strip() for alt in alternatives.replace('；', ';').replace('，', ',').split(',') if alt.strip()]
                        english_list.extend(alt_list)
                
                terminology[chinese_term] = english_list
            
            self.logger.info(f"从Excel加载了 {len(terminology)} 个术语映射")
            return terminology
            
        except Exception as e:
            self.logger.error(f"从Excel加载术语失败: {e}")
            return {}
            
    def check_length_ratio(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查长度比例
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            检查结果
        """
        if not source or not target:
            return {
                'passed': False,
                'issue': 'empty_text',
                'message': '源文本或目标文本为空',
                'ratio': 0
            }
            
        ratio = len(target) / len(source)
        min_ratio = self.rules_config.get('length_ratio_min', 0.3)
        max_ratio = self.rules_config.get('length_ratio_max', 3.0)
        
        if ratio < min_ratio:
            return {
                'passed': False,
                'issue': 'too_short',
                'message': f'翻译过短，长度比例 {ratio:.2f} < {min_ratio}',
                'ratio': ratio
            }
        elif ratio > max_ratio:
            return {
                'passed': False,
                'issue': 'too_long',
                'message': f'翻译过长，长度比例 {ratio:.2f} > {max_ratio}',
                'ratio': ratio
            }
        else:
            return {
                'passed': True,
                'issue': None,
                'message': '长度比例正常',
                'ratio': ratio
            }
            
    def check_empty_values(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查空值
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            检查结果
        """
        if not self.rules_config.get('check_empty', True):
            return {'passed': True, 'issue': None, 'message': '跳过空值检查'}
            
        issues = []
        
        if not source or source.strip() == '':
            issues.append('源文本为空')
            
        if not target or target.strip() == '':
            issues.append('目标文本为空')
            
        if issues:
            return {
                'passed': False,
                'issue': 'empty_values',
                'message': '; '.join(issues)
            }
        else:
            return {
                'passed': True,
                'issue': None,
                'message': '无空值问题'
            }
            
    def check_special_characters(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查特殊字符
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            检查结果
        """
        if not self.rules_config.get('check_special_chars', True):
            return {'passed': True, 'issue': None, 'message': '跳过特殊字符检查'}
            
        issues = []
        
        # 检查是否包含明显的机器翻译标记
        machine_translation_patterns = [
            r'\[翻译\]', r'\[译\]', r'机翻', r'谷歌翻译', r'百度翻译'
        ]
        
        for pattern in machine_translation_patterns:
            if re.search(pattern, target, re.IGNORECASE):
                issues.append(f'包含机器翻译标记: {pattern}')
                
        # 检查异常的字符重复
        if re.search(r'(.)\1{4,}', target):
            issues.append('存在异常的字符重复')
            
        # 检查是否包含过多的特殊符号
        special_char_threshold = self.rules_config.get('special_char_threshold', 0.1)
        special_char_count = len(re.findall(r'[^\w\s\u4e00-\u9fff.,;:!?。，；：！？]', target))
        if special_char_count > len(target) * special_char_threshold:
            issues.append(f'包含过多特殊字符 ({special_char_count}/{len(target)})')
            
        if issues:
            return {
                'passed': False,
                'issue': 'special_characters',
                'message': '; '.join(issues)
            }
        else:
            return {
                'passed': True,
                'issue': None,
                'message': '特殊字符检查通过'
            }
            
    def check_terminology_consistency(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查术语一致性
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            检查结果
        """
        if not self.keywords_config.get('check_terminology', True):
            return {'passed': True, 'issue': None, 'message': '跳过术语检查'}
            
        terminology = self.keywords_dict.get('terminology', {})
        if not terminology:
            return {'passed': True, 'issue': None, 'message': '无术语字典'}
            
        issues = []
        case_sensitive = self.keywords_config.get('case_sensitive', False)
        
        for chinese_term, english_terms in terminology.items():
            if chinese_term in source:
                # 检查目标文本是否包含对应的英文术语
                found_match = False
                for english_term in english_terms:
                    if case_sensitive:
                        if english_term in target:
                            found_match = True
                            break
                    else:
                        if english_term.lower() in target.lower():
                            found_match = True
                            break
                            
                if not found_match:
                    issues.append(f'术语 "{chinese_term}" 未找到对应的英文翻译')
                    
        if issues:
            return {
                'passed': False,
                'issue': 'terminology_inconsistency',
                'message': '; '.join(issues)
            }
        else:
            return {
                'passed': True,
                'issue': None,
                'message': '术语一致性检查通过'
            }
            
    def check_brand_names(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查品牌名称一致性
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            检查结果
        """
        brands = self.keywords_dict.get('brands', [])
        if not brands:
            return {'passed': True, 'issue': None, 'message': '无品牌名称字典'}
            
        issues = []
        
        for brand in brands:
            # 在源文本中查找品牌名称（可能是英文）
            if brand.lower() in source.lower():
                # 检查目标文本中是否保持一致
                if brand.lower() not in target.lower():
                    issues.append(f'品牌名称 "{brand}" 在翻译中丢失')
                    
        if issues:
            return {
                'passed': False,
                'issue': 'brand_inconsistency',
                'message': '; '.join(issues)
            }
        else:
            return {
                'passed': True,
                'issue': None,
                'message': '品牌名称检查通过'
            }
            
    def check_forbidden_words(self, target: str) -> Dict[str, Any]:
        """
        检查禁用词汇
        
        Args:
            target: 目标文本
            
        Returns:
            检查结果
        """
        forbidden_words = self.keywords_dict.get('forbidden_words', [])
        if not forbidden_words:
            return {'passed': True, 'issue': None, 'message': '无禁用词汇字典'}
            
        found_forbidden = []
        
        for word in forbidden_words:
            if word.lower() in target.lower():
                found_forbidden.append(word)
                
        if found_forbidden:
            return {
                'passed': False,
                'issue': 'forbidden_words',
                'message': f'包含禁用词汇: {", ".join(found_forbidden)}'
            }
        else:
            return {
                'passed': True,
                'issue': None,
                'message': '禁用词汇检查通过'
            }
            
    def check_single_pair(self, source: str, target: str) -> Dict[str, Any]:
        """
        检查单个翻译对
        
        Args:
            source: 源文本
            target: 目标文本
            
        Returns:
            完整的检查结果
        """
        results = {
            'length_ratio': self.check_length_ratio(source, target),
            'empty_values': self.check_empty_values(source, target),
            'special_characters': self.check_special_characters(source, target),
            'terminology': self.check_terminology_consistency(source, target),
            'brand_names': self.check_brand_names(source, target),
            'forbidden_words': self.check_forbidden_words(target)
        }
        
        # 计算总体通过状态
        all_passed = all(result['passed'] for result in results.values())
        
        # 收集所有问题
        issues = []
        for check_name, result in results.items():
            if not result['passed']:
                issues.append(f"{check_name}: {result['message']}")
                
        return {
            'overall_passed': all_passed,
            'issues': issues,
            'detailed_results': results
        }
        
    def check_dataframe(self, df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
        """
        批量检查数据框中的翻译对
        
        Args:
            df: 数据框
            source_col: 源语言列名
            target_col: 目标语言列名
            
        Returns:
            包含检查结果的数据框
        """
        self.logger.info(f"开始规则检查，共 {len(df)} 条数据")
        
        results = []
        for idx, row in df.iterrows():
            source = str(row[source_col]) if pd.notna(row[source_col]) else ""
            target = str(row[target_col]) if pd.notna(row[target_col]) else ""
            
            result = self.check_single_pair(source, target)
            results.append(result)
            
        # 添加结果到数据框
        result_df = df.copy()
        result_df['rule_check_passed'] = [r['overall_passed'] for r in results]
        result_df['rule_check_issues'] = ['; '.join(r['issues']) for r in results]
        result_df['rule_check_details'] = results
        
        passed_count = sum(r['overall_passed'] for r in results)
        self.logger.info(f"规则检查完成，通过率: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")
        
        return result_df
