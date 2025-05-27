import os  
import sys  
import json  
# 添加当前目录到路径，确保能找到scripts模块  
current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)  
sys.path.insert(0, project_root)  

from scripts.blender_runner import run_blender_script, get_default_blender_path  
from .image_evaluation import ImageQualityEvaluator  

class BlenderImageEvaluator:  
    def __init__(self, blender_executable=None, config_path=None):  
        """  
        初始化Blender图像评估器  
        """  
        # 设置默认配置路径  
        if config_path is None:  
            current_dir = os.path.dirname(os.path.abspath(__file__))  
            config_path = os.path.join(os.path.dirname(current_dir), "config", "default.json")  
        
        # 读取配置  
        self.config = self._load_config(config_path)  
        
        # 设置Blender路径  
        if blender_executable:  
            self.blender_executable = blender_executable  
        else:  
            self.blender_executable = self.config.get('blender_config', {}).get('executable_path', get_default_blender_path())  
        
        # 设置输出目录  
        self.output_folder = self.config.get('blender_config', {}).get('render_output', "./output/evaluation_renders")  
        os.makedirs(self.output_folder, exist_ok=True)  
        
        # 获取API密钥 - 多种方式尝试  
        api_key = self._get_api_key()  
        
        # 初始化图像评估器  
        self.image_evaluator = ImageQualityEvaluator(api_key)  
        
        print(f"🔧 Blender路径: {self.blender_executable}")  
        print(f"📁 输出目录: {self.output_folder}")  
        if api_key:  
            print(f"🔑 API密钥: {api_key[:10]}...")  
        else:  
            print("⚠️ 未配置API密钥，将使用简化评估")  
    
    def _load_config(self, config_path):  
        """加载配置文件"""  
        try:  
            if os.path.exists(config_path):  
                with open(config_path, 'r', encoding='utf-8') as f:  
                    config = json.load(f)  
                print(f"✅ 配置文件加载成功: {config_path}")  
                return config  
            else:  
                print(f"⚠️ 配置文件不存在: {config_path}")  
                return {}  
        except Exception as e:  
            print(f"❌ 配置文件加载失败: {e}")  
            return {}  
    
    def _get_api_key(self):  
        """获取OpenAI API密钥 - 多种方式尝试"""  
        
        # 方式1: 从配置文件读取  
        api_key = self.config.get('evaluation_config', {}).get('openai_api_key')  
        if api_key and api_key != "your_openai_api_key_here":  
            print("🔑 从配置文件获取API密钥")  
            return api_key  
        
        # 方式2: 从环境变量读取  
        api_key = os.getenv('OPENAI_API_KEY')  
        if api_key:  
            print("🔑 从环境变量获取API密钥")  
            return api_key  
        
        # 方式3: 从其他环境变量读取  
        for env_name in ['OPENAI_API_KEY', 'OPENAI_KEY', 'GPT_API_KEY']:  
            api_key = os.getenv(env_name)  
            if api_key:  
                print(f"🔑 从环境变量 {env_name} 获取API密钥")  
                return api_key  
        
        return None  
    
    # 其他方法保持不变...  
    def run_blender_script_and_render(self, script: str, obj_name: str) -> str:  
        """运行Blender脚本并渲染图像"""  
        try:  
            resolution = self.config.get('evaluation_config', {}).get('render_resolution', [512, 512])  
            
            success = run_blender_script(  
                script=script,  
                obj_name=obj_name,  
                output_folder=self.output_folder,  
                blender_executable=self.blender_executable,  
                save_obj=False,  
                save_image=True,  
                resolution=tuple(resolution)  
            )  
            
            if success:  
                image_path = os.path.join(self.output_folder, f"{obj_name}.png")  
                if os.path.exists(image_path):  
                    return image_path  
                else:  
                    print(f"❌ 图像文件未找到: {image_path}")  
                    return None  
            else:  
                print(f"❌ Blender脚本执行失败: {obj_name}")  
                return None  
                
        except Exception as e:  
            print(f"❌ 渲染过程发生错误: {e}")  
            return None  
    
    def evaluate_image_with_openai(self, image_path: str, prompt: str) -> dict:  
        """使用OpenAI评估图像质量，如果没有API密钥则使用简化评估"""  
        if self.image_evaluator.api_key:  
            return self.image_evaluator.evaluate_chair_design(image_path, prompt)  
        else:  
            # 简化评估：基于文件是否存在和大小  
            return self._simple_image_evaluation(image_path, prompt)  
    
    def _simple_image_evaluation(self, image_path: str, prompt: str) -> dict:  
        """简化的图像评估（当没有API密钥时）"""  
        if not os.path.exists(image_path):  
            return {  
                "total_score": 0,  
                "structure_score": 0,  
                "style_score": 0,  
                "function_score": 0,  
                "aesthetic_score": 0,  
                "comments": "图像文件不存在"  
            }  
        
        # 基于文件大小的简单评估  
        file_size = os.path.getsize(image_path)  
        
        if file_size < 1000:  # 小于1KB，可能是空文件  
            score = 10  
        elif file_size < 10000:  # 小于10KB，可能渲染不完整  
            score = 30  
        elif file_size < 50000:  # 50KB以下，基本渲染  
            score = 50  
        else:  # 50KB以上，较好的渲染  
            score = 70  
        
        return {  
            "total_score": score,  
            "structure_score": score // 4,  
            "style_score": score // 4,  
            "function_score": score // 4,  
            "aesthetic_score": score // 4,  
            "comments": f"简化评估（基于文件大小: {file_size} bytes）- 建议配置OpenAI API密钥获得详细评估"  
        }