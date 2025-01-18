import os

class UserInput:
    def __init__(self):
        self.required_fields = {
            "dataset_path": str,  # 数据集路径
            "knowledge_points": list,  # 需要覆盖的知识点
            "difficulty_level": str,  # 难度级别：'beginner', 'intermediate', 'advanced'
            "expected_duration": int,  # 预期完成时间（分钟）
        }
        
        self.difficulty_levels = ['beginner', 'intermediate', 'advanced']
    
    def validate_input(self, user_input):
        """验证用户输入"""
        if not all(field in user_input for field in self.required_fields):
            missing_fields = [f for f in self.required_fields if f not in user_input]
            raise ValueError(f"缺少必要的输入字段: {', '.join(missing_fields)}")
        
        # 验证数据集路径
        if not os.path.exists(user_input['dataset_path']):
            raise FileNotFoundError(f"找不到数据集文件: {user_input['dataset_path']}")
        
        # 验证知识点列表
        if not user_input['knowledge_points'] or not isinstance(user_input['knowledge_points'], list):
            raise ValueError("知识点必须是非空列表")
        
        # 验证难度级别
        if user_input['difficulty_level'] not in self.difficulty_levels:
            raise ValueError(f"难度级别必须是以下之一: {', '.join(self.difficulty_levels)}")
        
        # 验证预期时长
        if not isinstance(user_input['expected_duration'], int) or user_input['expected_duration'] <= 0:
            raise ValueError("预期完成时间必须是正整数（分钟）")
        
        return True 