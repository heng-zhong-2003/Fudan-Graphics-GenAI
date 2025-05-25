#!/usr/bin/env python3  
"""  
专业的椅子设计生成器 - 基于原始BlenderLLM  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  
import json  
import random  

class ChairDesignGenerator:  
    def __init__(self, model_path="../models/BlenderLLM"):  
        print("🪑 初始化椅子设计生成器...")  
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
        torch.cuda.empty_cache()  
        
        # 加载模型  
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
        self.model = AutoModelForCausalLM.from_pretrained(  
            model_path,  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map="auto"  
        )  
        
        print("✅ 模型加载完成")  
        
        # 椅子设计模板  
        self.chair_styles = [  
            "modern minimalist", "vintage wooden", "ergonomic office",   
            "comfortable armchair", "industrial bar stool", "scandinavian dining",  
            "luxury executive", "retro mid-century", "contemporary lounge",  
            "rustic farmhouse", "sleek racing", "elegant accent"  
        ]  
        
        self.chair_materials = [  
            "wood", "metal", "plastic", "leather", "fabric", "glass",  
            "bamboo", "steel", "aluminum", "oak", "pine", "mahogany"  
        ]  
        
        self.chair_features = [  
            "with armrests", "with cushions", "with wheels", "with swivel base",  
            "with high back", "with lumbar support", "with adjustable height",  
            "with reclining function", "with footrest", "stackable design"  
        ]  
    
    def generate_chair_design(self, style=None, material=None, feature=None, custom_prompt=None):  
        """生成椅子设计"""  
        
        if custom_prompt:  
            prompt = custom_prompt  
        else:  
            # 随机选择或使用指定参数  
            style = style or random.choice(self.chair_styles)  
            material = material or random.choice(self.chair_materials)  
            feature = feature or random.choice(self.chair_features)  
            
            prompt = f"Generate chair design: {style} {material} chair {feature}"  
        
        print(f"🎨 生成提示: {prompt}")  
        
        try:  
            inputs = self.tokenizer(prompt, return_tensors="pt")  
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = self.model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 600,  
                    temperature=0.7,  
                    do_sample=True,  
                    top_p=0.9,  
                    pad_token_id=self.tokenizer.eos_token_id,  
                    repetition_penalty=1.1,  
                    no_repeat_ngram_size=3  
                )  
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(prompt):].strip()  
            
            return {  
                'prompt': prompt,  
                'code': generated,  
                'success': True  
            }  
            
        except Exception as e:  
            print(f"❌ 生成失败: {e}")  
            return {  
                'prompt': prompt,  
                'code': '',  
                'success': False,  
                'error': str(e)  
            }  
    
    def batch_generate(self, count=10, save_to_file=True):  
        """批量生成椅子设计"""  
        print(f"🏭 批量生成 {count} 个椅子设计...")  
        
        results = []  
        
        for i in range(count):  
            print(f"\n--- 生成 {i+1}/{count} ---")  
            
            result = self.generate_chair_design()  
            results.append(result)  
            
            if result['success']:  
                print(f"✅ 成功生成 ({len(result['code'])} 字符)")  
                print(f"📝 代码预览: {result['code'][:150]}...")  
            else:  
                print(f"❌ 生成失败: {result.get('error', 'Unknown error')}")  
        
        # 统计  
        successful = len([r for r in results if r['success']])  
        print(f"\n📊 生成统计: {successful}/{count} 成功")  
        
        # 保存结果  
        if save_to_file:  
            output_dir = "./output/chair_designs"  
            os.makedirs(output_dir, exist_ok=True)  
            
            output_file = os.path.join(output_dir, "generated_chairs.json")  
            with open(output_file, 'w', encoding='utf-8') as f:  
                json.dump(results, f, indent=2, ensure_ascii=False)  
            
            print(f"💾 结果保存到: {output_file}")  
            
            # 同时保存为Python文件  
            py_file = os.path.join(output_dir, "chair_designs.py")  
            with open(py_file, 'w', encoding='utf-8') as f:  
                f.write("# Generated Chair Designs\n\n")  
                for i, result in enumerate(results):  
                    if result['success']:  
                        f.write(f"# Design {i+1}: {result['prompt']}\n")  
                        f.write(result['code'])  
                        f.write("\n\n" + "="*80 + "\n\n")  
            
            print(f"🐍 Python代码保存到: {py_file}")  
        
        return results  
    
    def interactive_mode(self):  
        """交互式椅子设计生成"""  
        print("\n🎮 进入交互式椅子设计模式")  
        print("输入 'quit' 退出，'random' 随机生成，或直接输入设计要求")  
        
        while True:  
            try:  
                user_input = input("\n🎨 请输入椅子设计要求: ").strip()  
                
                if user_input.lower() in ['quit', 'exit', 'q']:  
                    print("👋 再见!")  
                    break  
                
                if user_input.lower() == 'random':  
                    result = self.generate_chair_design()  
                else:  
                    # 如果不是完整的提示，添加前缀  
                    if not user_input.lower().startswith('generate'):  
                        user_input = f"Generate chair design: {user_input}"  
                    result = self.generate_chair_design(custom_prompt=user_input)  
                
                if result['success']:  
                    print(f"\n✅ 生成成功!")  
                    print(f"📝 提示: {result['prompt']}")  
                    print(f"🤖 生成的Blender代码:")  
                    print("-" * 60)  
                    print(result['code'])  
                    print("-" * 60)  
                else:  
                    print(f"❌ 生成失败: {result.get('error', 'Unknown error')}")  
                
            except KeyboardInterrupt:  
                print("\n👋 再见!")  
                break  
            except Exception as e:  
                print(f"❌ 错误: {e}")  

def main():  
    """主函数"""  
    print("🪑 椅子设计生成器")  
    print("1. 批量生成")  
    print("2. 交互式生成")  
    print("3. 测试特定设计")  
    
    try:  
        choice = input("\n请选择模式 (1/2/3): ").strip()  
        
        generator = ChairDesignGenerator()  
        
        if choice == "1":  
            count = int(input("生成数量 (默认10): ") or "10")  
            generator.batch_generate(count)  
            
        elif choice == "2":  
            generator.interactive_mode()  
            
        elif choice == "3":  
            test_designs = [  
                "modern ergonomic office chair with lumbar support",  
                "vintage wooden dining chair with carved details",  
                "minimalist steel bar stool with footrest",  
                "luxury leather executive chair with massage function",  
                "scandinavian oak armchair with wool cushions"  
            ]  
            
            print("🧪 测试特定设计...")  
            for design in test_designs:  
                print(f"\n{'='*60}")  
                result = generator.generate_chair_design(custom_prompt=f"Generate chair design: {design}")  
                if result['success']:  
                    print(f"✅ {design}")  
                    print(f"📝 代码长度: {len(result['code'])} 字符")  
                else:  
                    print(f"❌ {design} - 失败")  
        
        else:  
            print("❌ 无效选择")  
            
    except Exception as e:  
        print(f"❌ 程序错误: {e}")  

if __name__ == "__main__":  
    main()  
