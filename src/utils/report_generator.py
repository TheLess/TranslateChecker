"""
报告生成器
生成各种格式的检查报告
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .config_loader import ConfigLoader
from .logger import get_logger


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化报告生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path)
        self.logger = get_logger(__name__)
        
        # 设置中文字体（用于图表）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_excel_report(self, 
                            df: pd.DataFrame, 
                            summary: Dict[str, Any],
                            output_path: str) -> str:
        """
        生成Excel格式的详细报告
        
        Args:
            df: 检查结果数据框
            summary: 摘要信息
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # 主要结果
            df.to_excel(writer, sheet_name='检查结果', index=False)
            
            # 摘要信息
            summary_df = self._create_summary_dataframe(summary)
            summary_df.to_excel(writer, sheet_name='摘要报告', index=False)
            
            # 问题统计
            if 'overall_issues' in df.columns:
                issues_df = self._analyze_issues(df)
                issues_df.to_excel(writer, sheet_name='问题统计', index=False)
            
            # 分数分布
            if 'overall_score' in df.columns:
                score_dist_df = self._create_score_distribution(df)
                score_dist_df.to_excel(writer, sheet_name='分数分布', index=False)
                
        self.logger.info(f"Excel报告已生成: {output_path}")
        return str(output_path)
        
    def generate_html_report(self, 
                           df: pd.DataFrame, 
                           summary: Dict[str, Any],
                           output_path: str) -> str:
        """
        生成HTML格式的可视化报告
        
        Args:
            df: 检查结果数据框
            summary: 摘要信息
            output_path: 输出路径
            
        Returns:
            生成的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成图表
        charts_dir = output_path.parent / 'charts'
        charts_dir.mkdir(exist_ok=True)
        
        chart_files = []
        
        # 分数分布图
        if 'overall_score' in df.columns:
            score_chart = self._create_score_chart(df, charts_dir / 'score_distribution.png')
            chart_files.append(('分数分布', score_chart))
            
        # 问题类型图
        if 'overall_issues' in df.columns:
            issues_chart = self._create_issues_chart(df, charts_dir / 'issues_analysis.png')
            chart_files.append(('问题分析', issues_chart))
            
        # 生成HTML
        html_content = self._generate_html_content(df, summary, chart_files)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"HTML报告已生成: {output_path}")
        return str(output_path)
        
    def _create_summary_dataframe(self, summary: Dict[str, Any]) -> pd.DataFrame:
        """创建摘要数据框"""
        data = []
        
        # 基础统计
        if 'basic_stats' in summary:
            basic = summary['basic_stats']
            data.extend([
                ['基础信息', '总翻译数', basic.get('total_translations', 0)],
                ['基础信息', '处理时间(秒)', f"{basic.get('processing_time', 0):.2f}"],
                ['基础信息', '处理时间', basic.get('timestamp', '')],
            ])
            
        # 规则检测
        if 'rule_check' in summary:
            rule = summary['rule_check']
            data.extend([
                ['规则检测', '通过数量', rule.get('passed', 0)],
                ['规则检测', '失败数量', rule.get('failed', 0)],
                ['规则检测', '通过率', rule.get('pass_rate', '0%')],
            ])
            
        # 相似度检测
        if 'similarity_check' in summary:
            sim = summary['similarity_check']
            data.extend([
                ['相似度检测', '平均分数', sim.get('average_score', '0')],
                ['相似度检测', '最低分数', sim.get('min_score', '0')],
                ['相似度检测', '最高分数', sim.get('max_score', '0')],
                ['相似度检测', '通过数量', sim.get('passed', 0)],
            ])
            
        # LLM评估
        if 'llm_evaluation' in summary:
            llm = summary['llm_evaluation']
            data.extend([
                ['LLM评估', '评估数量', llm.get('evaluated_count', 0)],
                ['LLM评估', '平均分数', llm.get('average_score', '0')],
            ])
            
        return pd.DataFrame(data, columns=['类别', '指标', '值'])
        
    def _analyze_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析问题统计"""
        all_issues = []
        
        for issues_str in df['overall_issues']:
            if pd.notna(issues_str) and issues_str:
                issues = [issue.strip() for issue in str(issues_str).split(';') if issue.strip()]
                all_issues.extend(issues)
                
        # 统计问题频率
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
        # 创建数据框
        data = []
        total_issues = len(all_issues)
        
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_issues * 100) if total_issues > 0 else 0
            data.append([issue, count, f"{percentage:.1f}%"])
            
        return pd.DataFrame(data, columns=['问题类型', '出现次数', '占比'])
        
    def _create_score_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建分数分布统计"""
        if 'overall_score' not in df.columns:
            return pd.DataFrame()
            
        scores = df['overall_score']
        
        # 分数区间统计
        bins = [0, 4, 6, 8, 10]
        labels = ['差 (0-4)', '中 (4-6)', '良 (6-8)', '优 (8-10)']
        
        score_ranges = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
        distribution = score_ranges.value_counts().sort_index()
        
        data = []
        total = len(scores)
        
        for range_label, count in distribution.items():
            percentage = (count / total * 100) if total > 0 else 0
            data.append([range_label, count, f"{percentage:.1f}%"])
            
        return pd.DataFrame(data, columns=['分数区间', '数量', '占比'])
        
    def _create_score_chart(self, df: pd.DataFrame, output_path: Path) -> str:
        """创建分数分布图表"""
        if 'overall_score' not in df.columns:
            return ""
            
        plt.figure(figsize=(10, 6))
        
        # 分数分布直方图
        plt.subplot(1, 2, 1)
        plt.hist(df['overall_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('分数分布直方图')
        plt.xlabel('分数')
        plt.ylabel('频次')
        
        # 分数区间饼图
        plt.subplot(1, 2, 2)
        bins = [0, 4, 6, 8, 10]
        labels = ['差 (0-4)', '中 (4-6)', '良 (6-8)', '优 (8-10)']
        score_ranges = pd.cut(df['overall_score'], bins=bins, labels=labels, include_lowest=True)
        distribution = score_ranges.value_counts()
        
        plt.pie(distribution.values, labels=distribution.index, autopct='%1.1f%%', startangle=90)
        plt.title('分数区间分布')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_issues_chart(self, df: pd.DataFrame, output_path: Path) -> str:
        """创建问题分析图表"""
        if 'overall_issues' not in df.columns:
            return ""
            
        # 统计问题
        all_issues = []
        for issues_str in df['overall_issues']:
            if pd.notna(issues_str) and issues_str:
                issues = [issue.strip() for issue in str(issues_str).split(';') if issue.strip()]
                all_issues.extend(issues)
                
        if not all_issues:
            return ""
            
        issue_counts = {}
        for issue in all_issues:
            # 简化问题描述
            if '规则问题' in issue:
                category = '规则问题'
            elif '相似度问题' in issue:
                category = '相似度问题'
            elif 'LLM问题' in issue:
                category = 'LLM问题'
            else:
                category = '其他问题'
            issue_counts[category] = issue_counts.get(category, 0) + 1
            
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        categories = list(issue_counts.keys())
        counts = list(issue_counts.values())
        
        plt.bar(categories, counts, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        plt.title('问题类型分布')
        plt.xlabel('问题类型')
        plt.ylabel('出现次数')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _generate_html_content(self, 
                             df: pd.DataFrame, 
                             summary: Dict[str, Any], 
                             chart_files: List[tuple]) -> str:
        """生成HTML内容"""
        
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>翻译质量检查报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background-color: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h3 {{ margin-top: 0; color: #333; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .chart img {{ max-width: 100%; height: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-danger {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>翻译质量检查报告</h1>
        <p>生成时间: {timestamp}</p>
        <p>文件: {file_path}</p>
    </div>
    
    <div class="summary">
        {summary_cards}
    </div>
    
    <div class="charts">
        {charts_section}
    </div>
    
    <div class="details">
        <h2>详细结果</h2>
        {results_table}
    </div>
</body>
</html>
        """
        
        # 生成摘要卡片
        summary_cards = self._generate_summary_cards(summary)
        
        # 生成图表部分
        charts_section = ""
        for chart_title, chart_path in chart_files:
            if chart_path:
                relative_path = Path(chart_path).name
                charts_section += f"""
                <div class="chart">
                    <h3>{chart_title}</h3>
                    <img src="charts/{relative_path}" alt="{chart_title}">
                </div>
                """
        
        # 生成结果表格
        results_table = self._generate_results_table(df)
        
        # 填充模板
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            file_path=summary.get('basic_stats', {}).get('file_path', ''),
            summary_cards=summary_cards,
            charts_section=charts_section,
            results_table=results_table
        )
        
        return html_content
        
    def _generate_summary_cards(self, summary: Dict[str, Any]) -> str:
        """生成摘要卡片HTML"""
        cards = []
        
        # 基础统计卡片
        if 'basic_stats' in summary:
            basic = summary['basic_stats']
            cards.append(f"""
            <div class="card">
                <h3>基础统计</h3>
                <p>翻译条目: <strong>{basic.get('total_translations', 0)}</strong></p>
                <p>处理时间: <strong>{basic.get('processing_time', 0):.2f}秒</strong></p>
            </div>
            """)
            
        # 规则检测卡片
        if 'rule_check' in summary:
            rule = summary['rule_check']
            cards.append(f"""
            <div class="card">
                <h3>规则检测</h3>
                <p>通过率: <strong class="status-good">{rule.get('pass_rate', '0%')}</strong></p>
                <p>通过: {rule.get('passed', 0)} | 失败: {rule.get('failed', 0)}</p>
            </div>
            """)
            
        # 相似度检测卡片
        if 'similarity_check' in summary:
            sim = summary['similarity_check']
            cards.append(f"""
            <div class="card">
                <h3>相似度检测</h3>
                <p>平均分数: <strong>{sim.get('average_score', '0')}</strong></p>
                <p>通过数量: {sim.get('passed', 0)}</p>
            </div>
            """)
            
        return "".join(cards)
        
    def _generate_results_table(self, df: pd.DataFrame) -> str:
        """生成结果表格HTML"""
        # 选择关键列
        display_columns = []
        
        # 获取源列和目标列名
        source_col = None
        target_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['中文', 'chinese', 'source']):
                source_col = col
            elif any(keyword in col.lower() for keyword in ['英文', 'english', 'target']):
                target_col = col
                
        if source_col:
            display_columns.append(source_col)
        if target_col:
            display_columns.append(target_col)
            
        # 添加检查结果列
        result_columns = ['overall_score', 'overall_status', 'rule_check_passed', 
                         'similarity_score', 'llm_score']
        
        for col in result_columns:
            if col in df.columns:
                display_columns.append(col)
                
        # 限制显示行数
        display_df = df[display_columns].head(50)  # 只显示前50行
        
        # 生成HTML表格
        html_table = display_df.to_html(classes='table', escape=False, index=False)
        
        if len(df) > 50:
            html_table += f"<p><em>注: 仅显示前50行，总共{len(df)}行数据</em></p>"
            
        return html_table
