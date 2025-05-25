"""  
Blender图像渲染和评估工具  
整合渲染和质量评估功能  
"""  

import os  
import sys  
sys.path.append('../scripts')  
from scripts.blender_runner import run_blender_script, get_default_blender_path  
from .image_evaluation import ImageQualityEvaluator  

class BlenderImageEvaluator:  
    def __init__(self, blender_executable=None, config_path="../config/default.json"):  
        """  
        初始化Blender图像评估器  
        
        Args:  
            blender_executable: Blender可执行文件路径  
            config_path: 配置文件路径  
        """  
        # 读取配置  
        try:  
            import json  
            with open(config_path, 'r') as f:  
                config = json.load(f)  
            self.config = config  
        except:  
            self.config = {}  
        
        # 设置Blender路径  
        if blender_executable:  
            self.blender_executable = blender_executable  
        else:  
            self.blender_executable = self.config.get('blender_config', {}).get('executable_path', get_default_blender_path())  
        
        # 设置输出目录  
        self.output_folder = self.config.get('blender_config', {}).get('render_output', "./output/evaluation_renders")  
        os.makedirs(self.output_folder, exist_ok=True)  
        
        # 初始化图像评估器  
        api_key = self.config.get('evaluation_config', {}).get('openai_api_key')  
        self.image_evaluator = ImageQualityEvaluator(api_key)  
        
        print(f"🔧 Blender路径: {self.blender_executable}")  
        print(f"📁 输出目录: {self.output_folder}")  
    
    def run_blender_script_and_render(self, script: str, obj_name: str) -> str:  
        """  
        运行Blender脚本并渲染图像  
        
        Args:  
            script: Blender Python脚本  
            obj_name: 对象名称  
            
        Returns:  
            str: 渲染图像路径，失败时返回None  
        """  
        try:  
            # 获取渲染分辨率  
            resolution = self.config.get('evaluation_config', {}).get('render_resolution', [512, 512])  
            
            # 运行Blender脚本  
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
        """  
        使用OpenAI评估图像质量  
        
        Args:  
            image_path: 图像文件路径  
            prompt: 原始设计提示  
            
        Returns:  
            dict: 评估结果  
        """  
        return self.image_evaluator.evaluate_chair_design(image_path, prompt)  
    
    def batch_evaluate_scripts(self, script_prompt_pairs: list, name_prefix: str = "test") -> dict:  
        """  
        批量评估脚本生成的图像  
        
        Args:  
            script_prompt_pairs: [(script, prompt), ...] 列表  
            name_prefix: 文件名前缀  
            
        Returns:  
            dict: 批量评估结果  
        """  
        results = []  
        total_score = 0  
        success_count = 0  
        
        for i, (script, prompt) in enumerate(script_prompt_pairs):  
            obj_name = f"{name_prefix}_{i+1:02d}"  
            
            # 渲染图像  
            image_path = self.run_blender_script_and_render(script, obj_name)  
            
            if image_path:  
                # 评估图像质量  
                evaluation = self.evaluate_image_with_openai(image_path, prompt)  
                results.append({  
                    'obj_name': obj_name,  
                    'prompt': prompt,  
                    'image_path': image_path,  
                    'evaluation': evaluation  
                })  
                
                score = evaluation.get('total_score', 0)  
                total_score += score  
                success_count += 1  
                
                print(f"✅ {obj_name}: {score}/100")  
            else:  
                print(f"❌ {obj_name}: 渲染失败")  
                results.append({  
                    'obj_name': obj_name,  
                    'prompt': prompt,  
                    'image_path': None,  
                    'evaluation': {'total_score': 0, 'comments': '渲染失败'}  
                })  
        
        return {  
            'individual_results': results,  
            'total_score': total_score,  
            'average_score': total_score / success_count if success_count > 0 else 0,  
            'success_count': success_count,  
            'total_count': len(script_prompt_pairs)  
        }