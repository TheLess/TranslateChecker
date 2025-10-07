"""
创建示例翻译文件
"""

import pandas as pd
from pathlib import Path

def create_sample_file():
    """创建示例翻译文件"""
    
    # 示例数据 - 包含各种质量的翻译
    data = {
        'ID': list(range(1, 16)),  # 添加ID列
        '中文': [
            '人工智能是计算机科学的一个分支',
            '机器学习可以让计算机自动学习',
            '深度学习是机器学习的一个子集',
            '自然语言处理帮助计算机理解人类语言',
            '数据科学结合了统计学和计算机科学',
            '云计算提供了可扩展的计算资源',
            '区块链是一种分布式账本技术',
            '物联网连接了各种智能设备',
            '大数据分析帮助企业做出更好的决策',
            '网络安全保护数字资产免受威胁',
            # 一些有问题的例子
            '这是一个测试句子',
            '苹果公司是一家科技企业',
            '',  # 空的中文
            '这个翻译应该很短',
            '这是一个非常非常非常非常非常长的句子，用来测试长度比例检查功能是否正常工作'
        ],
        '英文': [
            'Artificial intelligence is a branch of computer science',
            'Machine learning enables computers to learn automatically',
            'Deep learning is a subset of machine learning',
            'Natural language processing helps computers understand human language',
            'Data science combines statistics and computer science',
            'Cloud computing provides scalable computing resources',
            'Blockchain is a distributed ledger technology',
            'Internet of Things connects various smart devices',
            'Big data analytics helps businesses make better decisions',
            'Cybersecurity protects digital assets from threats',
            # 一些有问题的翻译
            'This is wrong translation',  # 语义不匹配
            'Apple Inc. is a technology company',  # 正确翻译
            'Empty Chinese text above',  # 对应空中文
            'Very long translation for a short sentence that should be much shorter',  # 长度不匹配
            'Short'  # 太短的翻译
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 确保目录存在
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    # 保存文件
    output_file = examples_dir / 'sample_translation.xlsx'
    df.to_excel(output_file, index=False)
    
    print(f"示例文件已创建: {output_file}")
    print(f"包含 {len(df)} 条翻译数据")
    
    return output_file

if __name__ == '__main__':
    create_sample_file()
