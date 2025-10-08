"""
Translation Checker 主程序
提供命令行接口进行翻译质量检查
"""

import sys
import time
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

# 简单配置，避免复杂的全局修改

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
        console.print("\n[cyan]开始执行检查...[/cyan]")
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # 根据启用的功能计算总权重
            total_weight = 100
            step_weights = {
                'data_processing': 5,    # 数据处理很快
                'rule_checking': 10,     # 规则检测较快
                'similarity': 25 if similarity else 0,  # 相似度检测中等
                'llm': 60 if llm else 0  # LLM评估最耗时
            }
            
            # 如果某些步骤被跳过，重新分配权重
            active_weight = sum(step_weights.values())
            if active_weight < total_weight:
                # 将剩余权重分配给数据处理和规则检测
                remaining = total_weight - active_weight
                step_weights['data_processing'] += remaining // 2
                step_weights['rule_checking'] += remaining - (remaining // 2)
            
            # 创建主任务
            main_task = progress.add_task("检查中...", total=total_weight)
            current_progress = 0
            
            # 步骤1: 数据处理
            console.print("[cyan]🔄 开始步骤 1/4: 数据处理[/cyan]")
            progress.update(main_task, description="[cyan]步骤 1/4: 数据处理")
            df, column_mapping = checker.data_processor.process_file(file_path)
            current_progress += step_weights['data_processing']
            progress.update(main_task, completed=current_progress)
            console.print(f"[green]✓ 完成步骤 1/4: 数据处理 - 处理了 {len(df)} 条数据[/green]")
            
            # 步骤2: 规则检测
            console.print("[cyan]🔄 开始步骤 2/4: 基础规则检测[/cyan]")
            progress.update(main_task, description="[cyan]步骤 2/4: 基础规则检测")
            source_col = column_mapping['source']
            target_col = column_mapping['target']
            df = checker.rule_checker.check_dataframe(df, source_col, target_col)
            current_progress += step_weights['rule_checking']
            progress.update(main_task, completed=current_progress)
            console.print("[green]✓ 完成步骤 2/4: 基础规则检测[/green]")
            
            # 步骤3: 相似度检测
            if similarity:
                console.print("[cyan]🔄 开始步骤 3/4: 语义相似度检测[/cyan]")
                progress.update(main_task, description="[cyan]步骤 3/4: 语义相似度检测")
                checker._init_similarity_model()
                if checker.similarity_model:
                    df = checker.similarity_model.check_dataframe(df, source_col, target_col)
                current_progress += step_weights['similarity']
                progress.update(main_task, completed=current_progress)
                console.print("[green]✓ 完成步骤 3/4: 语义相似度检测[/green]")
            else:
                console.print("[yellow]⏭️ 跳过步骤 3/4: 语义相似度检测[/yellow]")
                progress.update(main_task, description="[yellow]步骤 3/4: 跳过相似度检测")
            
            # 步骤4: LLM评估
            if llm:
                console.print("[cyan]🔄 开始步骤 4/4: LLM质量评估[/cyan]")
                progress.update(main_task, description="[cyan]步骤 4/4: LLM质量评估")
                checker._init_llm_evaluator()
                if checker.llm_evaluator:
                    df = checker.llm_evaluator.evaluate_dataframe(df, source_col, target_col, llm_filter)
                current_progress += step_weights['llm']
                progress.update(main_task, completed=current_progress)
                console.print("[green]✓ 完成步骤 4/4: LLM质量评估[/green]")
            else:
                console.print("[yellow]⏭️ 跳过步骤 4/4: LLM质量评估[/yellow]")
                progress.update(main_task, description="[yellow]步骤 4/4: 跳过LLM评估")
            
            progress.update(main_task, description="[green]检查完成!", completed=total_weight)
        
        # 记录处理统计
        end_time = time.time()
        processing_time = end_time - start_time
        checker.processing_stats = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'processing_time': processing_time,
            'total_items': len(df)
        }
        
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
        'lazy_load': '[blue]⏳ 延迟加载[/blue]',
        'not_loaded': '[yellow]○ 未加载[/yellow]',
        'no_api_key': '[red]✗ 缺少API密钥[/red]',
        'missing_deps': '[red]✗ 缺少依赖[/red]',
        'ollama_ready': '[green]✓ Ollama就绪[/green]',
        'ollama_offline': '[red]✗ Ollama离线[/red]',
        'api_ready': '[green]✓ API就绪[/green]',
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
