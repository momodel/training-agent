from typing import Callable, Any, Dict, List
import time
from functools import wraps
import logging

class RetryHandler:
    """重试处理器"""
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
        self.logger = logging.getLogger(__name__)
    
    def retry_on_exception(self, func: Callable) -> Callable:
        """装饰器：在发生异常时进行重试"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    self.logger.warning(
                        f"第 {attempt + 1} 次尝试失败: {str(e)}, "
                        f"{'正在重试...' if attempt < self.max_retries - 1 else '已达到最大重试次数'}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.delay * (attempt + 1))  # 指数退避
            
            raise last_exception
        
        return wrapper

class LLMError(Exception):
    """LLM 相关错误"""
    pass

class ValidationError(Exception):
    """数据验证错误"""
    pass

def validate_llm_response(response: Dict[str, Any], required_fields: List[str]) -> None:
    """验证 LLM 响应"""
    if not response or not isinstance(response, dict):
        raise ValidationError("LLM 响应格式无效")
    
    missing_fields = [field for field in required_fields if field not in response]
    if missing_fields:
        raise ValidationError(f"LLM 响应缺少必要字段: {', '.join(missing_fields)}") 

class ProgressManager:
    """进度管理器"""
    def __init__(self):
        self.steps = {
            'dataset_analysis': '数据集分析',
            'theory_generation': '理论内容生成',
            'code_generation': '代码示例生成',
            'exercises_generation': '练习题生成',
            'notebook_assembly': 'Notebook 组装'
        }
        self.current_step = None
        self.total_steps = len(self.steps)
        self.current_step_number = 0
    
    def start_step(self, step_key: str) -> None:
        """开始一个步骤"""
        if step_key not in self.steps:
            raise ValueError(f"未知的步骤: {step_key}")
        
        self.current_step = step_key
        self.current_step_number += 1
        self._print_progress()
    
    def complete_step(self, step_key: str) -> None:
        """完成一个步骤"""
        if step_key != self.current_step:
            raise ValueError(f"步骤不匹配: 期望 {self.current_step}, 实际 {step_key}")
        
        print(f"✓ {self.steps[step_key]}完成")
    
    def _print_progress(self) -> None:
        """打印当前进度"""
        print(f"\n[{self.current_step_number}/{self.total_steps}] 正在执行: {self.steps[self.current_step]}...") 