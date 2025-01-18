from typing import Dict, Any, List
import os
import json
import pandas as pd
from openai import OpenAI
from utils import RetryHandler, LLMError, ValidationError, validate_llm_response
import logging
from prompts import (
    get_analysis_prompt,
    get_theory_prompt,
    get_code_prompt,
    get_exercises_prompt,
    get_content_organization_prompt
)

class DatasetAnalyzer:
    """数据集分析组件"""
    def __init__(self, llm_client: OpenAI, *, config=None):
        self.llm = llm_client
        self.retry_handler = RetryHandler()
        self.logger = logging.getLogger(__name__)
        self.max_preview_rows = config.preview.max_preview_rows if config else 5
        self.max_preview_size = config.preview.max_preview_size if config else 2000
        self.model = config.llm.model if config else "gpt-4o"
    
    def _get_dataset_preview(self, dataset_path: str) -> str:
        """获取数据集预览信息"""
        file_type = self._get_file_type(dataset_path)
        preview_info = []
        
        try:
            if file_type == 'csv':
                df = pd.read_csv(dataset_path, nrows=self.max_preview_rows)
                preview_info.extend([
                    "数据集基本信息：",
                    f"总行数：{len(pd.read_csv(dataset_path, usecols=[0]))}", # 只读取第一列来获取总行数
                    f"列数：{len(df.columns)}",
                    f"列名：{', '.join(df.columns)}",
                    "\n数据类型：",
                    df.dtypes.to_string(),
                    "\n前几行数据：",
                    df.head().to_string()
                ])
            
            elif file_type == 'excel':
                df = pd.read_excel(dataset_path, nrows=self.max_preview_rows)
                preview_info.extend([
                    "数据集基本信息：",
                    f"总行数：{len(pd.read_excel(dataset_path, usecols=[0]))}",
                    f"列数：{len(df.columns)}",
                    f"列名：{', '.join(df.columns)}",
                    "\n数据类型：",
                    df.dtypes.to_string(),
                    "\n前几行数据：",
                    df.head().to_string()
                ])
            
            else:
                raise ValueError(f"不支持的文件类型：{file_type}")
            
            # 添加缺失值信息
            missing_info = df.isnull().sum()
            if missing_info.any():
                preview_info.extend([
                    "\n缺失值信息：",
                    missing_info.to_string()
                ])
            
            # 添加数值列的基本统计信息
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                preview_info.extend([
                    "\n数值列统计信息：",
                    df[numeric_cols].describe().to_string()
                ])
            
            preview_text = "\n".join(preview_info)
            
            # 如果预览文本太长，进行截断
            if len(preview_text) > self.max_preview_size:
                preview_text = preview_text[:self.max_preview_size] + "..."
            
            return preview_text
            
        except Exception as e:
            raise ValueError(f"读取数据集时出错：{str(e)}")

    def _get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        ext = os.path.splitext(file_path)[1].lower()
        supported_types = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel'
        }
        
        if ext not in supported_types:
            raise ValueError(f"不支持的文件类型：{ext}")
        
        return supported_types[ext]

    def _get_llm_analysis(self, dataset_preview: str) -> str:
        """获取 LLM 分析结果"""
        try:
            print("正在发送请求到 LLM...")
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": get_analysis_prompt(dataset_preview)
                }]
            )
            print("LLM 响应接收完成")
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用出错: {str(e)}")
            raise LLMError(f"LLM 调用失败: {str(e)}")

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """解析 LLM 的分析结果"""
        print("开始解析 LLM 响应...")
        print(f"响应内容: {response_text[:200]}...")  # 只打印前200个字符
        
        # 清理响应文本，移除 Markdown 代码块标记
        def clean_json_response(text: str) -> str:
            # 移除 ```json 和 ``` 标记
            text = text.replace("```json", "").replace("```", "")
            # 移除开头和结尾的空白字符
            text = text.strip()
            return text
        
        try:
            # 清理并尝试解析 JSON
            print("尝试解析 JSON...")
            cleaned_response = clean_json_response(response_text)
            result = json.loads(cleaned_response)
            print("JSON 解析成功")
            
            # 验证必要的字段
            required_fields = ['data_type', 'task_type', 'features', 'target', 'preprocessing_steps']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                print(f"缺少必要字段: {missing_fields}")
                raise ValueError(f"分析结果缺少必要字段：{', '.join(missing_fields)}")
            
            print("所有必要字段都存在")
            
            # 标准化结果格式
            standardized_result = {
                'data_type': result['data_type'],
                'task_type': result['task_type'],
                'features': result['features'],
                'target': result['target'],
                'preprocessing_steps': result['preprocessing_steps']
            }
            
            print("结果标准化完成")
            return standardized_result
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {str(e)}")
            # 如果不是有效的 JSON，尝试提取关键信息
            try:
                print("尝试重新格式化响应...")
                # 重新请求 LLM 格式化结果
                format_prompt = f"""
                请将以下分析结果转换为标准的 JSON 格式（不要添加任何 Markdown 标记）：
                {response_text}
                
                需要包含以下字段：
                - data_type: 数据集类型
                - task_type: 任务类型
                - features: 特征说明
                - target: 目标变量
                - preprocessing_steps: 预处理步骤
                
                直接返回 JSON 对象，不要添加任何其他标记。
                """
                
                format_response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": format_prompt}]
                )
                
                print("重新格式化完成，尝试解析新的响应...")
                return self._parse_analysis_response(format_response.choices[0].message.content)
                
            except Exception as e:
                print(f"重新格式化失败: {str(e)}")
                raise ValueError(f"无法解析分析结果：{str(e)}")

    @RetryHandler().retry_on_exception
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """使用 LLM 分析数据集"""
        try:
            print("开始读取数据集...")
            dataset_preview = self._get_dataset_preview(dataset_path)
            print("数据集预览获取完成，正在调用 LLM 分析...")
            response = self._get_llm_analysis(dataset_preview)
            print("LLM 分析完成，正在解析结果...")
            return self._parse_analysis_response(response)
        except Exception as e:
            self.logger.error(f"数据集分析失败: {str(e)}")
            raise LLMError(f"数据集分析失败: {str(e)}")

class ContentGenerator:
    """内容生成组件"""
    def __init__(self, llm_client: OpenAI, *, config=None):
        self.llm = llm_client
        self.retry_handler = RetryHandler()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = config.llm.model if config else "gpt-4"
    
    def _get_llm_response(self, prompt: str) -> str:
        """获取 LLM 响应"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {str(e)}")

    def _validate_theory_content(self, content: str) -> None:
        """验证理论内容"""
        if not content or len(content.strip()) < 100:
            raise ValidationError("理论内容过短或为空")
        
        # 检查是否包含必要的部分（支持多种标题格式）
        title_patterns = [
            ["# 背景介绍", "## 背景介绍", "背景介绍", "一、背景介绍", "1. 背景介绍", "1、背景介绍"],
            ["# 理论讲解", "## 理论讲解", "理论讲解", "二、理论讲解", "2. 理论讲解", "2、理论讲解"],
            ["# 实践应用", "## 实践应用", "实践应用", "三、实践应用", "3. 实践应用", "3、实践应用", "代码实现", "# 代码实现", "## 代码实现"]
        ]
        
        missing_sections = []
        for patterns in title_patterns:
            if not any(pattern in content for pattern in patterns):
                # 取第一个模式作为代表性名称（去掉 # 号）
                section_name = patterns[0].replace("#", "").strip()
                missing_sections.append(section_name)
        
        # 如果内容长度足够且包含代码示例，我们可以适当放宽要求
        if len(content) >= 2000 and "```python" in content:
            # 只要有背景介绍和理论讲解就可以了
            missing_sections = [s for s in missing_sections if s not in ["实践应用"]]
        
        if missing_sections:
            raise ValidationError(f"理论内容缺少以下部分: {', '.join(missing_sections)}")
        
        # 检查内容长度是否合适（假设每个知识点至少需要500字）
        if len(content) < 1500:  # 降低最小长度要求
            raise ValidationError("理论内容长度不足，请确保内容充分详细")
    
    def _validate_code_content(self, content: str) -> None:
        """验证代码内容"""
        if not content or "```python" not in content:
            raise ValidationError("代码内容无效")
        
        # 检查是否包含必要的导入语句
        required_imports = ["pandas", "numpy", "sklearn"]
        for imp in required_imports:
            if imp not in content.lower():
                raise ValidationError(f"代码缺少 {imp} 库的导入")
    
    def _validate_exercises_content(self, content: str) -> None:
        """验证练习题内容"""
        if not content or len(content.strip()) < 100:
            raise ValidationError("练习题内容过短或为空")

        # 检查练习题数量
        exercise_count = content.count("## 练习")
        if exercise_count < 3:
            raise ValidationError("练习题数量不足，至少需要3个练习题")

        # 检查每个练习题的结构
        exercises = content.split("## 练习")[1:]  # 跳过第一个空字符串
        for i, exercise in enumerate(exercises, 1):
            if "### 问题" not in exercise:
                raise ValidationError(f"练习 {i} 缺少问题描述")
            if "### 参考答案" not in exercise:
                raise ValidationError(f"练习 {i} 缺少参考答案")
            if len(exercise.strip()) < 50:
                raise ValidationError(f"练习 {i} 内容过短")

    @RetryHandler().retry_on_exception
    def generate_theory(self, knowledge_points: List[str], dataset_info: Dict[str, Any]) -> str:
        """生成理论解释内容"""
        try:
            response = self._get_llm_response(
                get_theory_prompt(knowledge_points, dataset_info)
            )
            self._validate_theory_content(response)
            return response
        except Exception as e:
            self.logger.error(f"理论内容生成失败: {str(e)}")
            raise LLMError(f"理论内容生成失败: {str(e)}")
    
    @RetryHandler().retry_on_exception
    def generate_code_examples(self, dataset_info: Dict[str, Any]) -> str:
        """生成代码示例"""
        try:
            response = self._get_llm_response(
                get_code_prompt(dataset_info)
            )
            self._validate_code_content(response)
            return response
        except Exception as e:
            self.logger.error(f"代码示例生成失败: {str(e)}")
            raise LLMError(f"代码示例生成失败: {str(e)}")
    
    @RetryHandler().retry_on_exception
    def generate_exercises(self, difficulty_level: str, dataset_info: Dict[str, Any]) -> str:
        """生成练习题"""
        try:
            response = self._get_llm_response(
                get_exercises_prompt(difficulty_level, dataset_info)
            )
            self._validate_exercises_content(response)
            return response
        except Exception as e:
            self.logger.error(f"练习题生成失败: {str(e)}")
            raise LLMError(f"练习题生成失败: {str(e)}")

class NotebookAssembler:
    """Notebook 组装组件"""
    def __init__(self, llm_client: OpenAI, *, config=None):
        self.llm = llm_client
        self.retry_handler = RetryHandler()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = config.llm.model if config else "gpt-4"
        self.notebook_template = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def _organize_content(self, content: Dict[str, Any]) -> str:
        """组织内容的顺序和结构"""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user", 
                "content": get_content_organization_prompt(content)
            }]
        )
        return response.choices[0].message.content

    def _parse_content(self, content: str) -> List[Dict[str, str]]:
        """解析 LLM 返回的内容，将其分割成不同的部分"""
        sections = []
        current_section = {"cell_type": "markdown", "source": []}
        lines = content.split("\n")
        i = 0
        
        def save_current_markdown():
            nonlocal current_section
            if current_section["source"]:
                # 过滤掉 markdown 中的代码块
                filtered_lines = []
                skip_code = False
                for line in current_section["source"]:
                    if "```" in line:
                        skip_code = not skip_code
                        continue
                    if not skip_code:
                        filtered_lines.append(line)
                if filtered_lines:
                    current_section["source"] = filtered_lines
                    sections.append(current_section)
                current_section = {"cell_type": "markdown", "source": []}
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # 检查是否是代码块的开始
            if "```python" in line or (line.strip() == "```" and i > 0 and "python" in lines[i-1]):
                # 保存当前的 markdown 内容
                save_current_markdown()
                
                # 收集代码内容
                code_lines = []
                i += 1  # 跳过开始标记
                while i < len(lines):
                    line = lines[i].rstrip()
                    if "```" in line:  # 代码块结束
                        break
                    # 检查行是否包含实际代码（不是空行或注释）
                    if line.strip() and not line.strip().startswith("#"):
                        code_lines.append(line)
                    i += 1
                
                # 只有当代码行不为空时才创建代码单元格
                if code_lines:
                    sections.append({
                        "cell_type": "code",
                        "source": code_lines,
                        "execution_count": None,
                        "outputs": []
                    })
            
            # 处理普通文本
            elif not (line.strip().startswith("```") and "python" in line):
                current_section["source"].append(line)
            
            i += 1
        
        # 保存最后的 markdown 内容
        save_current_markdown()
        
        # 后处理：合并相邻的 markdown 单元格
        merged_sections = []
        for section in sections:
            if (merged_sections and 
                section["cell_type"] == "markdown" and 
                merged_sections[-1]["cell_type"] == "markdown"):
                # 合并相邻的 markdown 单元格
                merged_sections[-1]["source"].extend(section["source"])
            else:
                merged_sections.append(section)
        
        return merged_sections

    def _convert_to_ipynb(self, content: str) -> str:
        """将内容转换为 ipynb 格式"""
        try:
            # 解析内容
            sections = self._parse_content(content)
            
            # 创建新的 notebook
            notebook = self.notebook_template.copy()
            
            # 添加标题
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# 机器学习实训课程\n\n"]
            })
            
            # 添加各个部分的内容
            for section in sections:
                cell = {
                    "cell_type": section["cell_type"],
                    "metadata": {},
                }
                
                if section["cell_type"] == "markdown":
                    cell["source"] = ["\n".join(section["source"])]
                else:
                    cell["source"] = section["source"]
                    cell["execution_count"] = None
                    cell["outputs"] = []
                
                notebook["cells"].append(cell)
            
            # 转换为 JSON 字符串
            return json.dumps(notebook, ensure_ascii=False, indent=2)
            
        except Exception as e:
            raise ValueError(f"转换 notebook 格式时出错：{str(e)}")

    def _validate_notebook(self, notebook: str) -> None:
        """验证生成的 notebook 是否有效"""
        try:
            notebook_dict = json.loads(notebook)
            required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
            validate_llm_response(notebook_dict, required_fields)
            
            # 验证单元格
            if not notebook_dict['cells']:
                raise ValidationError("Notebook 不能没有内容")
            
            # 验证每个单元格的格式
            for cell in notebook_dict['cells']:
                if 'cell_type' not in cell or 'source' not in cell:
                    raise ValidationError("单元格格式无效")
                
        except json.JSONDecodeError:
            raise ValidationError("生成的 notebook 不是有效的 JSON 格式")

    def _show_structure_preview(self, content: str) -> None:
        """显示 notebook 结构预览"""
        print("\n📔 Notebook 结构预览:")
        print("-" * 40)
        
        # 提取所有标题
        sections = []
        current_level = 0
        
        for line in content.split("\n"):
            if line.startswith("#"):
                # 计算标题级别
                level = len(line.split()[0])
                title = line.strip("#").strip()
                sections.append("  " * (level - 1) + "- " + title)
        
        # 显示目录结构
        print("目录结构:")
        for section in sections:
            print(section)
        
        # 统计信息
        code_blocks = content.count("```python")
        print(f"\n包含 {code_blocks} 个代码块")
        
        print("-" * 40)

    @RetryHandler().retry_on_exception
    def create_notebook(self, content: Dict[str, Any]) -> str:
        """将所有内容组装成 Jupyter notebook"""
        try:
            organized_content = self._organize_content(content)
            
            # 在转换之前显示结构预览
            self._show_structure_preview(organized_content)
            
            notebook = self._convert_to_ipynb(organized_content)
            self._validate_notebook(notebook)
            return notebook
            
        except Exception as e:
            self.logger.error(f"Notebook 创建失败: {str(e)}")
            raise ValueError(f"Notebook 创建失败: {str(e)}") 