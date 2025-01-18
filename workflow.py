from typing import Dict, Any, List
import os
from openai import OpenAI
from agent_components import DatasetAnalyzer, ContentGenerator, NotebookAssembler
from interface import UserInput
from utils import ProgressManager
from config import Config
from datetime import datetime
import json

class AITrainingGenerator:
    """AI 实训生成器"""
    def __init__(self, api_key: str, config_path: str = None):
        self.config = Config(config_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            # base_url="https://open.momodel.cn/v1"
        )
        self.dataset_analyzer = DatasetAnalyzer(self.llm_client, config=self.config)
        self.content_generator = ContentGenerator(self.llm_client, config=self.config)
        self.notebook_assembler = NotebookAssembler(self.llm_client, config=self.config)
        self.user_input = UserInput()
        self.progress = ProgressManager()
        # 添加缓存目录
        self.cache_dir = os.path.join(self.config.output.output_dir, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, step: str, dataset_path: str) -> str:
        """获取缓存文件路径"""
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        return os.path.join(self.cache_dir, f"{dataset_name}_{step}.json")
    
    def _load_from_cache(self, step: str, dataset_path: str) -> Dict[str, Any]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(step, dataset_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"读取缓存失败: {str(e)}")
        return None
    
    def _save_to_cache(self, step: str, dataset_path: str, data: Dict[str, Any]) -> None:
        """保存数据到缓存"""
        try:
            cache_path = self._get_cache_path(step, dataset_path)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {str(e)}")
    
    def generate_training(self, input_data: Dict[str, Any]) -> str:
        """生成实训内容的主流程"""
        try:
            # 1. 验证用户输入
            self.user_input.validate_input(input_data)
            
            # 2. 分析数据集
            self.progress.start_step('dataset_analysis')
            dataset_info = self._load_from_cache('dataset_analysis', input_data['dataset_path'])
            if dataset_info is None:
                dataset_info = self.dataset_analyzer.analyze_dataset(input_data['dataset_path'])
                self._save_to_cache('dataset_analysis', input_data['dataset_path'], dataset_info)
            self.progress.complete_step('dataset_analysis')
            
            # 3. 生成内容
            content = self._generate_content(input_data, dataset_info)
            
            # 显示内容预览
            self._show_content_preview(content)
            
            # 4. 组装 notebook
            self.progress.start_step('notebook_assembly')
            notebook = self._load_from_cache('notebook', input_data['dataset_path'])
            if notebook is None:
                notebook = self.notebook_assembler.create_notebook(content)
                self._save_to_cache('notebook', input_data['dataset_path'], notebook)
            self.progress.complete_step('notebook_assembly')
            
            # 5. 保存结果
            output_path = self._save_notebook(notebook, input_data['dataset_path'])
            
            print(f"\n✨ 实训内容生成完成！文件保存在: {output_path}")
            return output_path
            
        except Exception as e:
            self._handle_error(e)
    
    def _generate_content(self, input_data: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成所有需要的内容"""
        content = {}
        
        # 1. 生成理论内容
        self.progress.start_step('theory_generation')
        theory_cache = self._load_from_cache('theory', input_data['dataset_path'])
        if theory_cache is None:
            content['theory'] = self.content_generator.generate_theory(
                knowledge_points=input_data['knowledge_points'],
                dataset_info=dataset_info
            )
            self._save_to_cache('theory', input_data['dataset_path'], {'theory': content['theory']})
        else:
            content['theory'] = theory_cache['theory']
        self.progress.complete_step('theory_generation')
        
        # 2. 生成代码示例
        self.progress.start_step('code_generation')
        code_cache = self._load_from_cache('code', input_data['dataset_path'])
        if code_cache is None:
            content['code_examples'] = self.content_generator.generate_code_examples(
                dataset_info=dataset_info
            )
            self._save_to_cache('code', input_data['dataset_path'], {'code_examples': content['code_examples']})
        else:
            content['code_examples'] = code_cache['code_examples']
        self.progress.complete_step('code_generation')
        
        # 3. 生成练习题
        self.progress.start_step('exercises_generation')
        exercises_cache = self._load_from_cache('exercises', input_data['dataset_path'])
        if exercises_cache is None:
            content['exercises'] = self.content_generator.generate_exercises(
                difficulty_level=input_data['difficulty_level'],
                dataset_info=dataset_info
            )
            self._save_to_cache('exercises', input_data['dataset_path'], {'exercises': content['exercises']})
        else:
            content['exercises'] = exercises_cache['exercises']
        self.progress.complete_step('exercises_generation')
        
        return content
    
    def _save_notebook(self, notebook: str, dataset_path: str) -> str:
        """保存生成的 notebook"""
        # 创建输出目录
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        
        # 生成文件名
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.config.output.include_timestamp else ""
        file_name = f"{self.config.output.file_prefix}{base_name}{timestamp}.ipynb"
        output_path = os.path.join(self.config.output.output_dir, file_name)
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notebook)
        
        return output_path
    
    def _handle_error(self, error: Exception) -> None:
        """处理错误"""
        error_map = {
            ValueError: "输入参数错误",
            FileNotFoundError: "找不到数据集文件",
            Exception: "生成过程中发生错误"
        }
        
        error_type = type(error)
        error_message = error_map.get(error_type, error_map[Exception])
        
        print(f"错误: {error_message}")
        print(f"详细信息: {str(error)}")
        raise error
    
    def _show_content_preview(self, content: Dict[str, Any]) -> None:
        """显示生成内容的预览"""
        print("\n📝 生成内容预览:")
        
        # 理论内容预览
        if 'theory' in content:
            theory_preview = content['theory'][:self.config.preview.preview_length]
            print("\n理论部分:")
            print("-" * 40)
            print(f"{theory_preview}...")
        
        # 代码示例预览
        if 'code_examples' in content:
            code_preview = self._extract_code_preview(content['code_examples'])
            print("\n代码示例:")
            print("-" * 40)
            print(code_preview)
        
        # 练习题预览
        if 'exercises' in content:
            print("\n练习题:")
            print("-" * 40)
            exercises = content['exercises'].split("## 练习")
            exercise_count = len(exercises) - 1  # 减去第一个空分片
            print(f"共生成 {exercise_count} 个练习题")
            
            # 显示第一个练习题的预览
            if exercise_count > 0:
                first_exercise = exercises[1]  # 第一个实际的练习题
                print("\n第一个练习题预览:")
                # 提取问题部分
                if "### 问题" in first_exercise:
                    question = first_exercise.split("### 问题")[1].split("###")[0]
                    print(f"问题: {question[:self.config.preview.preview_length].strip()}...")
        
        print("\n" + "-" * 40)
    
    def _extract_code_preview(self, code_content: str) -> str:
        """提取代码示例的预览"""
        # 查找第一个代码块
        code_blocks = code_content.split("```python")
        if len(code_blocks) > 1:
            first_block = code_blocks[1].split("```")[0]
            # 获取前几行代码
            code_lines = first_block.strip().split("\n")[:5]
            return "\n".join(code_lines) + "\n..."
        return "未找到代码块"

def create_training(api_key: str, input_data: Dict[str, Any], config_path: str = None) -> str:
    """便捷函数，用于创建实训内容"""
    generator = AITrainingGenerator(api_key, config_path)
    return generator.generate_training(input_data) 