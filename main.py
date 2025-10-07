"""
Translation Checker 主程序
提供命令行接口进行翻译质量检查
"""

import click
import sys
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.checker import TranslationChecker
from src.utils.config_loader import ConfigLoader


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="Translation Checker")
def cli():
    """Translation Checker - 翻译质量检查工具"""
    pass


@cli.command()
@click.option('--file', '-f', 'file_path', required=True, type=click.Path(exists=True),
              help='要检查的Excel文件路径')
@click.option('--output', '-o', 'output_path', type=click.Path(),
              help='输出文件路径（可选）')
@click.option('--similarity/--no-similarity', default=True,
              help='是否启用语义相似度检测（默认启用）')
@click.option('--llm/--no-llm', default=False,
              help='是否启用LLM质量评估（默认禁用）')
@click.option('--llm-filter', type=str,
              help='LLM评估过滤条件，例如: "similarity_score < 0.7"')
@click.option('--format', 'output_format', type=click.Choice(['excel', 'csv', 'json']), 
              default='excel', help='输出格式（默认Excel）')
@click.option('--detailed-report/--no-detailed-report', default=False,
              help='是否生成详细报告（默认否）')
@click.option('--config', 'config_path', type=click.Path(),
              help='配置文件路径（可选）')
def check(file_path, output_path, similarity, llm, llm_filter, output_format, detailed_report, config_path):
    """检查单个翻译文件的质量"""
    
    try:
        # 显示开始信息
        console.print(Panel.fit(
            f"[bold blue]Translation Checker[/bold blue]\n"
            f"正在检查文件: [green]{file_path}[/green]",
            title="开始检查"
        ))
        
        # 初始化检查器
        config_file = config_path or "config/config.yaml"
        checker = TranslationChecker(config_file)
        
        # 显示系统状态
        status = checker.get_system_status()
        _display_system_status(status)
        
        # 执行检查
        with Progress() as progress:
            task = progress.add_task("[cyan]检查中...", total=100)
            
            progress.update(task, advance=25, description="[cyan]数据处理中...")
            df = checker.check_file(
                file_path=file_path,
                enable_similarity=similarity,
                enable_llm=llm,
                llm_filter=llm_filter
            )
            progress.update(task, advance=75, description="[green]检查完成!")
        
        # 显示结果摘要
        summary = checker.get_summary_report(df)
        _display_summary(summary)
        
        # 导出结果
        if output_path:
            exported_path = checker.export_results(output_path, df, output_format)
            console.print(f"[green]✓[/green] 结果已导出到: {exported_path}")
        else:
            # 自动生成输出文件名
            input_file = Path(file_path)
            timestamp = checker.processing_stats['timestamp'][:19].replace(':', '-')
            auto_output = f"output/{input_file.stem}_checked_{timestamp}.{output_format}"
            exported_path = checker.export_results(auto_output, df, output_format)
            console.print(f"[green]✓[/green] 结果已自动导出到: {exported_path}")
            
        # 生成详细报告
        if detailed_report:
            _generate_detailed_report(checker, df, summary)
            
    except Exception as e:
        console.print(f"[red]✗[/red] 检查失败: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--input-dir', '-i', 'input_dir', required=True, type=click.Path(exists=True),
              help='包含翻译文件的目录')
@click.option('--output-dir', '-o', 'output_dir', type=click.Path(),
              help='输出目录（可选）')
@click.option('--pattern', '-p', default='*.xlsx', 
              help='文件匹配模式（默认 *.xlsx）')
@click.option('--similarity/--no-similarity', default=True,
              help='是否启用语义相似度检测')
@click.option('--llm/--no-llm', default=False,
              help='是否启用LLM质量评估')
@click.option('--config', 'config_path', type=click.Path(),
              help='配置文件路径（可选）')
def batch(input_dir, output_dir, pattern, similarity, llm, config_path):
    """批量检查多个翻译文件"""
    
    try:
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        if not files:
            console.print(f"[yellow]警告[/yellow]: 在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return
            
        console.print(Panel.fit(
            f"[bold blue]批量检查模式[/bold blue]\n"
            f"目录: [green]{input_dir}[/green]\n"
            f"找到 [yellow]{len(files)}[/yellow] 个文件",
            title="批量检查"
        ))
        
        # 初始化检查器
        config_file = config_path or "config/config.yaml"
        checker = TranslationChecker(config_file)
        
        # 创建输出目录
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("output/batch_results")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 批量处理
        results_summary = []
        
        with Progress() as progress:
            main_task = progress.add_task("[cyan]批量处理中...", total=len(files))
            
            for file_path in files:
                progress.update(main_task, description=f"[cyan]处理: {file_path.name}")
                
                try:
                    # 检查文件
                    df = checker.check_file(
                        file_path=file_path,
                        enable_similarity=similarity,
                        enable_llm=llm
                    )
                    
                    # 导出结果
                    output_file = output_path / f"{file_path.stem}_checked.xlsx"
                    checker.export_results(output_file, df, 'excel')
                    
                    # 收集摘要
                    summary = checker.get_summary_report(df)
                    summary['file_name'] = file_path.name
                    summary['output_file'] = str(output_file)
                    results_summary.append(summary)
                    
                    console.print(f"[green]✓[/green] 完成: {file_path.name}")
                    
                except Exception as e:
                    console.print(f"[red]✗[/red] 失败: {file_path.name} - {str(e)}")
                    
                progress.update(main_task, advance=1)
        
        # 生成批量报告
        _generate_batch_report(results_summary, output_path)
        
        console.print(f"[green]✓[/green] 批量处理完成，结果保存在: {output_path}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] 批量处理失败: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--config', 'config_path', type=click.Path(),
              help='配置文件路径（可选）')
def status(config_path):
    """显示系统状态和配置信息"""
    
    try:
        config_file = config_path or "config/config.yaml"
        checker = TranslationChecker(config_file)
        
        # 系统状态
        system_status = checker.get_system_status()
        _display_system_status(system_status, detailed=True)
        
        # 配置信息
        config = ConfigLoader(config_file)
        _display_config_info(config)
        
    except Exception as e:
        console.print(f"[red]✗[/red] 获取状态失败: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--example-file', '-e', 'example_file', 
              type=click.Path(), default='examples/sample_translation.xlsx',
              help='示例文件路径')
def demo(example_file):
    """运行演示，创建示例文件并进行检查"""
    
    try:
        # 创建示例文件
        _create_example_file(example_file)
        console.print(f"[green]✓[/green] 示例文件已创建: {example_file}")
        
        # 运行检查
        console.print("\n[bold blue]运行演示检查...[/bold blue]")
        
        checker = TranslationChecker()
        df = checker.check_file(
            file_path=example_file,
            enable_similarity=True,
            enable_llm=False
        )
        
        # 显示结果
        summary = checker.get_summary_report(df)
        _display_summary(summary)
        
        # 导出结果
        output_file = "examples/demo_results.xlsx"
        checker.export_results(output_file, df)
        console.print(f"[green]✓[/green] 演示结果已保存: {output_file}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] 演示失败: {str(e)}")
        sys.exit(1)


def _display_system_status(status: dict, detailed: bool = False):
    """显示系统状态"""
    table = Table(title="系统状态")
    table.add_column("模块", style="cyan")
    table.add_column("状态", style="green")
    
    status_icons = {
        'ready': '[green]✓ 就绪[/green]',
        'loaded': '[green]✓ 已加载[/green]',
        'not_loaded': '[yellow]○ 未加载[/yellow]',
        'no_api_key': '[red]✗ 缺少API密钥[/red]',
        'unknown': '[red]? 未知[/red]'
    }
    
    for module, state in status.items():
        table.add_row(
            module.replace('_', ' ').title(),
            status_icons.get(state, f'[red]{state}[/red]')
        )
    
    console.print(table)


def _display_summary(summary: dict):
    """显示检查结果摘要"""
    if 'error' in summary:
        console.print(f"[red]错误[/red]: {summary['error']}")
        return
        
    # 基础统计
    basic = summary.get('basic_stats', {})
    console.print(Panel.fit(
        f"文件: [green]{basic.get('file_path', 'N/A')}[/green]\n"
        f"翻译条目: [yellow]{basic.get('total_translations', 0)}[/yellow]\n"
        f"处理时间: [blue]{basic.get('processing_time', 0):.2f}[/blue] 秒",
        title="基础统计"
    ))
    
    # 检查结果表格
    table = Table(title="检查结果摘要")
    table.add_column("检查项目", style="cyan")
    table.add_column("结果", style="green")
    
    # 规则检测
    if 'rule_check' in summary:
        rule = summary['rule_check']
        table.add_row(
            "规则检测",
            f"通过: {rule['passed']}, 失败: {rule['failed']} (通过率: {rule['pass_rate']})"
        )
    
    # 相似度检测
    if 'similarity_check' in summary:
        sim = summary['similarity_check']
        table.add_row(
            "相似度检测",
            f"平均分: {sim['average_score']}, 通过: {sim['passed']}"
        )
    
    # LLM评估
    if 'llm_evaluation' in summary:
        llm = summary['llm_evaluation']
        table.add_row(
            "LLM评估",
            f"评估数: {llm['evaluated_count']}, 平均分: {llm['average_score']}"
        )
    
    # 综合评估
    if 'overall_assessment' in summary:
        overall = summary['overall_assessment']
        table.add_row(
            "综合评分",
            f"平均分: {overall['average_score']}"
        )
    
    console.print(table)
    
    # 建议
    if 'overall_assessment' in summary and 'recommendations' in summary['overall_assessment']:
        recommendations = summary['overall_assessment']['recommendations']
        if recommendations:
            console.print("\n[bold yellow]改进建议:[/bold yellow]")
            for i, rec in enumerate(recommendations[:5], 1):
                console.print(f"{i}. {rec}")


def _display_config_info(config: ConfigLoader):
    """显示配置信息"""
    table = Table(title="配置信息")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="green")
    
    # 关键配置项
    key_configs = [
        ("相似度模型", config.get('detection.similarity.model_name', 'N/A')),
        ("相似度阈值", config.get('detection.similarity.threshold', 'N/A')),
        ("LLM提供商", config.get('llm.provider', 'N/A')),
        ("输出格式", config.get('output.format', 'N/A')),
        ("日志级别", config.get('logging.level', 'N/A'))
    ]
    
    for key, value in key_configs:
        table.add_row(key, str(value))
    
    console.print(table)


def _create_example_file(file_path: str):
    """创建示例翻译文件"""
    import pandas as pd
    
    # 示例数据
    data = {
        '中文': [
            '人工智能是计算机科学的一个分支',
            '机器学习可以让计算机自动学习',
            '深度学习是机器学习的一个子集',
            '自然语言处理帮助计算机理解人类语言',
            '数据科学结合了统计学和计算机科学',
            '云计算提供了可扩展的计算资源',
            '区块链是一种分布式账本技术',
            '物联网连接了各种智能设备'
        ],
        '英文': [
            'Artificial intelligence is a branch of computer science',
            'Machine learning enables computers to learn automatically',
            'Deep learning is a subset of machine learning',
            'Natural language processing helps computers understand human language',
            'Data science combines statistics and computer science',
            'Cloud computing provides scalable computing resources',
            'Blockchain is a distributed ledger technology',
            'Internet of Things connects various smart devices'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 确保目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    df.to_excel(file_path, index=False)


def _generate_detailed_report(checker, df, summary):
    """生成详细的HTML报告"""
    # 这里可以实现HTML报告生成
    console.print("[yellow]详细报告功能开发中...[/yellow]")


def _generate_batch_report(results_summary, output_path):
    """生成批量处理报告"""
    report_file = output_path / "batch_report.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]✓[/green] 批量报告已生成: {report_file}")


if __name__ == '__main__':
    cli()
