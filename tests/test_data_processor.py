"""
数据处理器测试
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.processors.data_processor import DataProcessor


class TestDataProcessor:
    """数据处理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = DataProcessor()
        
    def test_clean_text(self):
        """测试文本清洗功能"""
        # 测试正常文本
        assert self.processor.clean_text("  hello world  ") == "hello world"
        
        # 测试空值
        assert self.processor.clean_text(None) == ""
        assert self.processor.clean_text("") == ""
        
        # 测试多余空白
        assert self.processor.clean_text("hello   world\n\t") == "hello world"
        
    def test_identify_columns(self):
        """测试列识别功能"""
        # 创建测试数据框
        df = pd.DataFrame({
            '中文': ['你好', '世界'],
            '英文': ['hello', 'world'],
            '其他': ['other', 'data']
        })
        
        columns = self.processor.identify_columns(df)
        
        assert columns['source'] == '中文'
        assert columns['target'] == '英文'
        
    def test_preprocess_data(self):
        """测试数据预处理"""
        # 创建测试数据
        df = pd.DataFrame({
            'source': ['  测试文本  ', '', '另一个测试'],
            'target': ['test text', None, 'another test']
        })
        
        processed_df = self.processor.preprocess_data(df, 'source', 'target')
        
        # 检查清洗结果
        assert processed_df.loc[0, 'source'] == '测试文本'
        assert processed_df.loc[1, 'target'] == ''
        
        # 检查统计信息
        assert 'source_length' in processed_df.columns
        assert 'target_length' in processed_df.columns
        assert 'length_ratio' in processed_df.columns
        
    def create_test_excel_file(self):
        """创建测试Excel文件"""
        data = {
            '中文': ['人工智能', '机器学习', '深度学习'],
            '英文': ['Artificial Intelligence', 'Machine Learning', 'Deep Learning']
        }
        df = pd.DataFrame(data)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df.to_excel(tmp.name, index=False)
            return tmp.name
            
    def test_load_excel_file(self):
        """测试Excel文件加载"""
        # 创建测试文件
        test_file = self.create_test_excel_file()
        
        try:
            df = self.processor.load_excel_file(test_file)
            
            assert len(df) == 3
            assert '中文' in df.columns
            assert '英文' in df.columns
            
        finally:
            # 清理临时文件
            Path(test_file).unlink()
            
    def test_process_file(self):
        """测试完整文件处理流程"""
        # 创建测试文件
        test_file = self.create_test_excel_file()
        
        try:
            df, column_mapping = self.processor.process_file(test_file)
            
            # 检查结果
            assert len(df) == 3
            assert column_mapping['source'] == '中文'
            assert column_mapping['target'] == '英文'
            assert 'original_index' in df.columns
            
        finally:
            # 清理临时文件
            Path(test_file).unlink()


if __name__ == '__main__':
    pytest.main([__file__])
