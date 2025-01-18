from typing import Dict, Any
from dataclasses import dataclass
import json
import os

@dataclass
class LLMConfig:
    """LLM 配置"""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 1.0

@dataclass
class PreviewConfig:
    """预览配置"""
    max_preview_rows: int = 5
    max_preview_size: int = 2000
    preview_length: int = 200

@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = "outputs"
    file_prefix: str = "generated_"
    include_timestamp: bool = True

class Config:
    """配置管理器"""
    def __init__(self, config_path: str = None):
        self.llm = LLMConfig()
        self.preview = PreviewConfig()
        self.output = OutputConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
            # 更新 LLM 配置
            if 'llm' in config_data:
                for key, value in config_data['llm'].items():
                    if hasattr(self.llm, key):
                        setattr(self.llm, key, value)
            
            # 更新预览配置
            if 'preview' in config_data:
                for key, value in config_data['preview'].items():
                    if hasattr(self.preview, key):
                        setattr(self.preview, key, value)
            
            # 更新输出配置
            if 'output' in config_data:
                for key, value in config_data['output'].items():
                    if hasattr(self.output, key):
                        setattr(self.output, key, value)
    
    def save_config(self, config_path: str) -> None:
        """保存配置到文件"""
        config_data = {
            'llm': {
                'model': self.llm.model,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'top_p': self.llm.top_p
            },
            'preview': {
                'max_preview_rows': self.preview.max_preview_rows,
                'max_preview_size': self.preview.max_preview_size,
                'preview_length': self.preview.preview_length
            },
            'output': {
                'output_dir': self.output.output_dir,
                'file_prefix': self.output.file_prefix,
                'include_timestamp': self.output.include_timestamp
            }
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def get_llm_params(self) -> Dict[str, Any]:
        """获取 LLM 参数"""
        return {
            'model': self.llm.model,
            'temperature': self.llm.temperature,
            'max_tokens': self.llm.max_tokens,
            'top_p': self.llm.top_p
        } 