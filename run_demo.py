"""
Translation Checker 演示脚本
快速演示翻译检查功能
"""

import sys
from pathlib import Path
import pandas as pd

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.checker import TranslationChecker
from create_sample import create_sample_file


def main():
    """运行演示"""
    print("=" * 60)
    print("Translation Checker 演示")
    print("=" * 60)
    
    try:
        # 1. 创建示例文件
        print("\n步骤 1: 创建示例翻译文件...")
        sample_file = create_sample_file()
        
        # 2. 初始化检查器
        print("\n步骤 2: 初始化翻译检查器...")
        checker = TranslationChecker()
        
        # 3. 显示系统状态
        print("\n步骤 3: 检查系统状态...")
        status = checker.get_system_status()
        for module, state in status.items():
            print(f"  {module}: {state}")
        
        # 4. 执行检查
        print("\n步骤 4: 执行翻译质量检查...")
        print("  - 启用规则检测: ✓")
        print("  - 启用相似度检测: ✓")
        print("  - 启用LLM评估: ✗ (需要API密钥)")
        
        df = checker.check_file(
            file_path=sample_file,
            enable_similarity=True,
            enable_llm=False  # 演示中不使用LLM以避免API密钥问题
        )
        
        # 5. 显示结果摘要
        print("\n步骤 5: 生成检查报告...")
        summary = checker.get_summary_report(df)
        
        print("\n" + "=" * 40)
        print("检查结果摘要")
        print("=" * 40)
        
        # 基础统计
        if 'basic_stats' in summary:
            basic = summary['basic_stats']
            print(f"总翻译条目: {basic.get('total_translations', 0)}")
            print(f"处理时间: {basic.get('processing_time', 0):.2f} 秒")
        
        # 规则检测结果
        if 'rule_check' in summary:
            rule = summary['rule_check']
            print(f"\n规则检测:")
            print(f"  通过: {rule.get('passed', 0)}")
            print(f"  失败: {rule.get('failed', 0)}")
            print(f"  通过率: {rule.get('pass_rate', '0%')}")
        
        # 相似度检测结果
        if 'similarity_check' in summary:
            sim = summary['similarity_check']
            print(f"\n相似度检测:")
            print(f"  平均分数: {sim.get('average_score', '0')}")
            print(f"  通过数量: {sim.get('passed', 0)}")
        
        # 综合评估
        if 'overall_assessment' in summary:
            overall = summary['overall_assessment']
            print(f"\n综合评估:")
            print(f"  平均分数: {overall.get('average_score', '0')}")
            
            if 'status_distribution' in overall:
                print("  质量分布:")
                for status, count in overall['status_distribution'].items():
                    print(f"    {status}: {count}")
        
        # 6. 导出结果
        print("\n步骤 6: 导出检查结果...")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # 导出Excel
        excel_file = checker.export_results("output/demo_results.xlsx", df, 'excel')
        print(f"Excel报告: {excel_file}")
        
        # 导出CSV
        csv_file = checker.export_results("output/demo_results.csv", df, 'csv')
        print(f"CSV报告: {csv_file}")
        
        # 7. 显示问题示例
        print("\n步骤 7: 显示发现的问题...")
        problem_rows = df[~df.get('rule_check_passed', True) | 
                         ~df.get('similarity_passed', True)]
        
        if len(problem_rows) > 0:
            print(f"\n发现 {len(problem_rows)} 个潜在问题:")
            for idx, row in problem_rows.head(3).iterrows():  # 只显示前3个
                source_col = None
                target_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['中文', 'chinese', 'source']):
                        source_col = col
                    elif any(keyword in col.lower() for keyword in ['英文', 'english', 'target']):
                        target_col = col
                
                if source_col and target_col:
                    print(f"\n  问题 {idx + 1}:")
                    print(f"    原文: {row[source_col]}")
                    print(f"    译文: {row[target_col]}")
                    
                    issues = []
                    if not row.get('rule_check_passed', True):
                        issues.append(f"规则问题: {row.get('rule_check_issues', '')}")
                    if not row.get('similarity_passed', True):
                        issues.append(f"相似度问题: {row.get('similarity_message', '')}")
                    
                    for issue in issues:
                        print(f"    {issue}")
        
        # 8. 改进建议
        if 'overall_assessment' in summary and 'recommendations' in summary['overall_assessment']:
            recommendations = summary['overall_assessment']['recommendations']
            if recommendations:
                print(f"\n步骤 8: 改进建议")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        print("\n下一步:")
        print("1. 查看生成的报告文件")
        print("2. 配置OpenAI API密钥以启用LLM评估")
        print("3. 使用自己的翻译文件进行检查")
        print("4. 根据需要调整配置参数")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("\n可能的解决方案:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 检查Python版本是否为3.8+")
        print("3. 确保有足够的磁盘空间和内存")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
