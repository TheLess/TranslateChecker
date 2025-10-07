"""
配置加载器
负责加载和管理配置文件
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
        
    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                logging.warning(f"配置文件不存在: {self.config_path}")
                self._config = {}
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            self._config = {}
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的嵌套键
        
        Args:
            key: 配置键，支持 'section.subsection.key' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        if self._config is None:
            return default
            
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        if self._config is None:
            self._config = {}
            
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save(self, output_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，默认为原配置文件路径
        """
        if output_path is None:
            output_path = self.config_path
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
            
    def reload(self):
        """重新加载配置文件"""
        self._load_config()
        
    @property
    def config(self) -> Dict:
        """获取完整配置字典"""
        return self._config or {}
        
    def update_from_env(self):
        """从环境变量更新配置"""
        # OpenAI API Key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.set('llm.openai.api_key', openai_key)
            
        # 其他环境变量可以在这里添加
        azure_key = os.getenv('AZURE_OPENAI_KEY')
        if azure_key:
            self.set('llm.azure.api_key', azure_key)
