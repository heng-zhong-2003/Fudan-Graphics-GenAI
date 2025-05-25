"""  
图像质量评估工具  
使用OpenAI GPT-4V对渲染图像进行质量评估  
"""  

import base64  
import requests  
import os  
import json  
from typing import Dict, Any  

class ImageQualityEvaluator:  
    def __init__(self, api_key: str = None):  
        """  
        初始化图像质量评估器  
        
        Args:  
            api_key: OpenAI API密钥，如果不提供则从环境变量读取  
        """  
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')  
        if not self.api_key:  
            print("⚠️ 未找到OpenAI API密钥，图像评估功能将被禁用")  
    
    def encode_image_to_base64(self, image_path: str) -> str:  
        """将图像文件编码为base64字符串"""  
        try:  
            with open(image_path, "rb") as image_file:  
                return base64.b64encode(image_file.read()).decode('utf-8')  
        except Exception as e:  
            print(f"❌ 图像编码失败: {e}")  
            return None  
    
    def evaluate_chair_design(self, image_path: str, prompt: str) -> Dict[str, Any]:  
        """  
        评估椅子设计图像质量  
        
        Args:  
            image_path: 图像文件路径  
            prompt: 原始设计提示  
            
        Returns:  
            包含评分和评价的字典  
        """  
        if not self.api_key:  
            return {"total_score": 0, "comments": "API密钥未配置"}  
        
        base64_image = self.encode_image_to_base64(image_path)  
        if not base64_image:  
            return {"total_score": 0, "comments": "图像编码失败"}  
        
        evaluation_prompt = f"""  
请评估这个3D椅子模型的质量。原始设计要求："{prompt}"  

评分标准（总分100分）：  
1. 结构完整性 (0-25分)：椅子是否有完整的座椅、靠背、支撑结构  
2. 风格匹配度 (0-25分)：是否符合描述的风格特征（现代、复古、工业等）  
3. 功能实现度 (0-25分)：是否包含描述的功能元素（扶手、轮子、可调节等）  
4. 美观协调性 (0-25分)：整体造型、比例、细节是否协调美观  

请严格按照以下JSON格式返回评估结果：  
{{  
    "total_score": 总分数字(0-100),  
    "structure_score": 结构分数字(0-25),  
    "style_score": 风格分数字(0-25),   
    "function_score": 功能分数字(0-25),  
    "aesthetic_score": 美观分数字(0-25),  
    "comments": "详细评价文字"  
}}  
"""  
        
        try:  
            response = requests.post(  
                "https://api.openai.com/v1/chat/completions",  
                headers={  
                    "Authorization": f"Bearer {self.api_key}",  
                    "Content-Type": "application/json"  
                },  
                json={  
                    "model": "gpt-4-vision-preview",  
                    "messages": [  
                        {  
                            "role": "user",  
                            "content": [  
                                {"type": "text", "text": evaluation_prompt},  
                                {  
                                    "type": "image_url",  
                                    "image_url": {  
                                        "url": f"data:image/png;base64,{base64_image}"  
                                    }  
                                }  
                            ]  
                        }  
                    ],  
                    "max_tokens": 500  
                }  
            )  
            
            if response.status_code == 200:  
                result = response.json()  
                content = result['choices'][0]['message']['content']  
                
                # 尝试解析JSON  
                try:  
                    scores = json.loads(content)  
                    return scores  
                except json.JSONDecodeError:  
                    # 如果JSON解析失败，尝试提取数字  
                    import re  
                    total_match = re.search(r'"total_score":\s*(\d+)', content)  
                    total_score = int(total_match.group(1)) if total_match else 50  
                    
                    return {  
                        "total_score": total_score,  
                        "structure_score": total_score // 4,  
                        "style_score": total_score // 4,  
                        "function_score": total_score // 4,  
                        "aesthetic_score": total_score // 4,  
                        "comments": content  
                    }  
            else:  
                print(f"❌ OpenAI API错误: {response.status_code}")  
                return {"total_score": 0, "comments": f"API调用失败: {response.status_code}"}  
                
        except Exception as e:  
            print(f"❌ 图像评估错误: {e}")  
            return {"total_score": 0, "comments": f"评估失败: {e}"}  
    
    def batch_evaluate(self, image_prompt_pairs: list) -> Dict[str, Any]:  
        """  
        批量评估多个图像  
        
        Args:  
            image_prompt_pairs: [(image_path, prompt), ...] 列表  
            
        Returns:  
            包含所有评估结果的字典  
        """  
        results = []  
        total_score = 0  
        
        for image_path, prompt in image_prompt_pairs:  
            if os.path.exists(image_path):  
                result = self.evaluate_chair_design(image_path, prompt)  
                results.append({  
                    'image_path': image_path,  
                    'prompt': prompt,  
                    'evaluation': result  
                })  
                total_score += result.get('total_score', 0)  
                print(f"✅ 评估完成: {os.path.basename(image_path)} - {result.get('total_score', 0)}/100")  
            else:  
                print(f"❌ 图像不存在: {image_path}")  
        
        return {  
            'individual_results': results,  
            'total_score': total_score,  
            'average_score': total_score / len(results) if results else 0,  
            'count': len(results)  
        }