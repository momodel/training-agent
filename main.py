from workflow import create_training
import os

def main():
    # OpenAI API 密钥
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    # 测试数据
    input_data = {
        'dataset_path': input("请输入数据集路径: "),
        'knowledge_points': [
            '决策树'
        ],
        'difficulty_level': 'intermediate',
        'expected_duration': 120  # 分钟
    }

    print("\n开始生成实训内容...")
    output_path = create_training(api_key, input_data, config_path=None)
    print(f"\n✨ 生成成功！文件保存在: {output_path}")

if __name__ == "__main__":
    main()