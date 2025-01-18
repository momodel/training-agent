# AI 训练内容生成器

这是一个基于 LLM 的智能实训内容生成器，可以根据数据集自动生成结构化的 Jupyter notebook 教程。

## 功能特点

- 自动分析数据集特征和任务类型
- 生成理论讲解、代码示例和练习题
- 支持自定义知识点和难度级别
- 生成结构化的 Jupyter notebook
- 缓存机制避免重复生成
- 支持多种数据集格式（CSV、Excel）

## 安装要求

- Python 3.8+
- OpenAI API 密钥
- 依赖包（见 requirements.txt）

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository-url]
cd training-agent
```

2. 创建并激活虚拟环境：
```bash
pyenv virtualenv 3.10.x training-agent
pyenv local training-agent
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 设置环境变量：
```bash
export OPENAI_API_KEY='your-api-key'
```

## 使用方法

1. 准备数据集文件（支持 CSV 或 Excel 格式）

2. 运行程序：
```python
python main.py
```

3. 根据提示输入：
   - 数据集路径
   - 知识点列表
   - 难度级别

4. 生成的 notebook 将保存在 `outputs` 目录下

## 配置说明

可以通过 `config.yaml` 文件配置以下参数：
- LLM 模型选择
- 输出目录设置
- 预览行数限制
- 缓存设置

## 目录结构

```
training-agent/
├── agent_components.py   # 核心组件实现
├── config.py            # 配置管理
├── interface.py         # 用户交互接口
├── main.py             # 主程序入口
├── prompts.py          # 提示词管理
├── utils.py            # 工具函数
├── requirements.txt    # 依赖包列表
└── outputs/            # 生成的内容
    └── .cache/        # 缓存目录
```

## 注意事项

- 确保 OpenAI API 密钥已正确设置
- 数据集大小建议不超过 10MB
- 首次运行可能需要较长时间
- 建议保留缓存以提高后续生成速度

## 许可证

MIT License 