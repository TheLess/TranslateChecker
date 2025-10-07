"""
日志工具
统一的日志配置和管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "translation_checker",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return logging.getLogger(name)


class LoggerMixin:
    """日志器混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取当前类的日志器"""
        return get_logger(self.__class__.__name__)


# 默认日志器配置
_default_logger = None


def init_default_logger(config_dict: Optional[dict] = None):
    """
    初始化默认日志器
    
    Args:
        config_dict: 日志配置字典
    """
    global _default_logger
    
    if config_dict is None:
        config_dict = {
            'level': 'INFO',
            'file': f'logs/translation_checker_{datetime.now().strftime("%Y%m%d")}.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    _default_logger = setup_logger(
        name="translation_checker",
        level=config_dict.get('level', 'INFO'),
        log_file=config_dict.get('file'),
        format_string=config_dict.get('format')
    )
    
    return _default_logger
