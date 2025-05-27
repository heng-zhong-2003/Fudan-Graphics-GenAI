#!/usr/bin/env python3  
"""  
分离式模型评估 - 避免同时加载两个模型  
先运行原始模型，保存结果，清理内存，再运行LoRA模型  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from peft import PeftModel  
import json  
import os  
import gc  
import time  
from datetime import datetime  
from utils.blender_evaluator import BlenderImageEvaluator

class SeparateModelEvaluator:  
    """分离式模型评估器"""  
    
    def __init__(self):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        self.test_prompts = [  
            # 基础风格类型  
            # "Design a chair: modern minimalist office chair with wheels",  
            # "Design a chair: comfortable recliner with armrests and cushions",   
            # "Design a chair: dining chair with tall backrest and no armrests",  
            # "Design a chair: ergonomic gaming chair with adjustable height",  
            # "Design a chair: vintage wooden chair with carved details",  
            
            # # 工业和现代风格  
            # "Design a chair: industrial metal chair with reinforced frame",  
            # "Design a chair: contemporary sleek chair with leather upholstery",  
            # "Design a chair: scandinavian wooden chair with minimalist design",  
            # "Design a chair: modern acrylic transparent chair with simple lines",  
            
            # # 功能特化椅子  
            # "Design a chair: high bar stool with footrest and swivel base",  
            # "Design a chair: folding portable chair with compact design",  
            # "Design a chair: office chair with wheels, armrests and lumbar support",  
            # "Design a chair: gaming racing chair with headrest and massage nodes",  
            # "Design a chair: lounge recliner with adjustable backrest and footrest",  
            
            # # 材质和舒适性  
            # "Design a chair: fabric cushioned chair with padded seat and backrest",  
            # "Design a chair: wide spacious chair with soft upholstery and arms",  
            # "Design a chair: small compact chair for space-saving dining room",  
            # "Design a chair: traditional chair with carved wooden details and high back",  
            
            # # 特殊功能  
            # "Design a chair: height adjustable desk chair with wheels and cup holder",  
            # "Design a chair: ergonomic chair with lumbar support, armrests and mesh backrest"  
            "Design a chair: modern minimalist office chair with wheels",  
            "Design a chair: ergonomic gaming chair with adjustable height",   
            "Design a chair: vintage wooden chair with carved details",  
            "Design a chair: industrial metal chair with reinforced frame",  
            "Design a chair: scandinavian wooden chair with minimalist design"  
        ]  
        
    def clear_gpu_memory(self):  
        """清理GPU内存"""  
        if torch.cuda.is_available():  
            torch.cuda.empty_cache()  
            torch.cuda.synchronize()  
            gc.collect()  
            print(f"🧹 GPU内存已清理")  
            
    def load_tokenizer(self):  
        """加载tokenizer"""  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        return tokenizer  
    
    def generate_responses(self, model, tokenizer, prompts, model_name):  
        """生成模型响应"""  
        print(f"\n📝 使用{model_name}生成响应...")  
        responses = {}  
        
        for i, prompt in enumerate(prompts, 1):  
            print(f"  处理 {i}/{len(prompts)}: {prompt[:40]}...")  
            
            input_text = f"User: {prompt}\n\nAssistant:"  
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_new_tokens=200,  
                    temperature=0.7,  
                    do_sample=True,  
                    pad_token_id=tokenizer.pad_token_id,  
                    eos_token_id=tokenizer.eos_token_id  
                )  
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            assistant_response = response.split("Assistant:")[-1].strip()  
            responses[prompt] = assistant_response  
            
            print(f"    生成长度: {len(assistant_response)} 字符")  
        
        return responses  
    
    def evaluate_original_model(self):  
        """评估原始模型"""  
        print("🔵 评估原始BlenderLLM模型")  
        print("=" * 40)  
        
        # 清理内存  
        self.clear_gpu_memory()  
        
        # 加载tokenizer  
        tokenizer = self.load_tokenizer()  
        
        # 加载原始模型  
        print("📦 加载原始BlenderLLM...")  
        original_model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        
        # 生成响应  
        original_responses = self.generate_responses(  
            original_model, tokenizer, self.test_prompts, "原始BlenderLLM"  
        )  
        
        # 保存结果  
        os.makedirs('./output/evaluation_results', exist_ok=True)  
        with open('./output/evaluation_results/original_responses.json', 'w') as f:  
            json.dump({  
                'model': 'original_blenderllm',  
                'timestamp': datetime.now().isoformat(),  
                'responses': original_responses  
            }, f, indent=2, ensure_ascii=False)  
        
        # 清理模型  
        del original_model  
        del tokenizer  
        self.clear_gpu_memory()  
        
        print("✅ 原始模型评估完成，结果已保存")  
        return original_responses  
    
    def evaluate_lora_model(self):  
        """评估LoRA模型"""  
        print("\n🟢 评估LoRA增强模型")  
        print("=" * 40)  
        
        # 清理内存  
        self.clear_gpu_memory()  
        time.sleep(2)  # 等待内存完全释放  
        
        # 加载tokenizer  
        tokenizer = self.load_tokenizer()  
        
        # 加载LoRA模型  
        print("🔧 加载LoRA增强模型...")  
        base_model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        lora_model = PeftModel.from_pretrained(base_model, "./output/lora_blender_enhanced")  
        
        # 生成响应  
        lora_responses = self.generate_responses(  
            lora_model, tokenizer, self.test_prompts, "LoRA增强模型"  
        )  
        
        # 保存结果  
        with open('./output/evaluation_results/lora_responses.json', 'w') as f:  
            json.dump({  
                'model': 'lora_enhanced_blenderllm',  
                'timestamp': datetime.now().isoformat(),  
                'responses': lora_responses  
            }, f, indent=2, ensure_ascii=False)  
        
        # 清理模型  
        del lora_model  
        del base_model  
        del tokenizer  
        self.clear_gpu_memory()  
        
        print("✅ LoRA模型评估完成，结果已保存")  
        return lora_responses  
    
    def simple_compare_responses(self, original_responses, lora_responses):  
        """简单对比响应质量"""  
        print("\n📊 响应质量对比")  
        print("=" * 50)  
        
        def simple_score(response):  
            """简单评分系统"""  
            score = 0  
            if 'import bpy' in response: score += 2  
            if 'primitive' in response: score += 2  
            if 'chair' in response.lower(): score += 1  
            if any(word in response.lower() for word in ['seat', 'back', 'leg', 'arm']): score += 2  
            if '.scale' in response or '.location' in response: score += 2  
            if len(response.split('\n')) > 5: score += 1  
            return score  
        
        total_original = 0  
        total_lora = 0  
        
        for i, prompt in enumerate(self.test_prompts, 1):  
            orig_resp = original_responses.get(prompt, "")  
            lora_resp = lora_responses.get(prompt, "")  
            
            orig_score = simple_score(orig_resp)  
            lora_score = simple_score(lora_resp)  
            
            total_original += orig_score  
            total_lora += lora_score  
            
            print(f"\n📝 测试 {i}: {prompt[:50]}...")  
            print(f"  🔵 原始模型: {orig_score}/10")  
            print(f"  🟢 LoRA模型: {lora_score}/10")  
            print(f"  📈 改进: {lora_score - orig_score:+d}")  
            
            # 显示部分响应  
            print(f"  原始响应: {orig_resp[:100]}...")  
            print(f"  LoRA响应: {lora_resp[:100]}...")  
        
        print(f"\n🎯 总体结果:")  
        print(f"  原始模型总分: {total_original}/{len(self.test_prompts)*10}")  
        print(f"  LoRA模型总分: {total_lora}/{len(self.test_prompts)*10}")  
        print(f"  整体改进: {total_lora - total_original:+d} ({(total_lora-total_original)/total_original*100:+.1f}%)")  
        
        return {  
            'original_total': total_original,  
            'lora_total': total_lora,  
            'improvement': total_lora - total_original  
        }  
    

    def enhanced_compare_responses(self, original_responses, lora_responses):  
        """增强的响应质量对比 - 包含图像渲染评估"""  
        print("\n📊 增强版响应质量对比")  
        print("=" * 60)  
        
        evaluator = BlenderImageEvaluator()  
        
        def enhanced_code_score(response):  
            """增强的代码评分系统"""  
            score = 0  
            
            # 基础语法检查 (0-20分)  
            if 'import bpy' in response: score += 5  
            if 'primitive' in response: score += 5  
            if any(op in response for op in ['add', 'create', 'mesh']): score += 5  
            if 'location' in response and 'scale' in response: score += 5  
            
            # 椅子特征检查 (0-30分)  
            chair_features = ['seat', 'backrest', 'leg', 'arm', 'chair']  
            feature_count = sum(1 for f in chair_features if f in response.lower())  
            score += min(feature_count * 6, 30)  
            
            # 代码复杂度 (0-20分)  
            lines = len(response.split('\n'))  
            if lines > 10: score += 5  
            if lines > 20: score += 5  
            if lines > 30: score += 5  
            if 'for' in response or 'while' in response: score += 5  
            
            # 材质和细节 (0-15分)  
            if 'material' in response: score += 5  
            if 'modifier' in response: score += 5  
            if any(mat in response.lower() for mat in ['wood', 'metal', 'fabric', 'leather']): score += 5  
            
            # 高级功能 (0-15分)  
            advanced_features = ['wheel', 'adjust', 'cushion', 'armrest', 'headrest']  
            advanced_count = sum(1 for f in advanced_features if f in response.lower())  
            score += min(advanced_count * 3, 15)  
            
            return min(score, 100)  
        
        # 选择关键测试用例进行图像评估  
        key_prompts = [  
            "Design a chair: modern minimalist office chair with wheels",  
            "Design a chair: ergonomic gaming chair with adjustable height",   
            "Design a chair: vintage wooden chair with carved details",  
            "Design a chair: industrial metal chair with reinforced frame",  
            "Design a chair: scandinavian wooden chair with minimalist design"  
        ]  
        
        total_original_code = 0  
        total_lora_code = 0  
        total_original_image = 0  
        total_lora_image = 0  
        
        for i, prompt in enumerate(self.test_prompts, 1):  
            orig_resp = original_responses.get(prompt, "")  
            lora_resp = lora_responses.get(prompt, "")  
            
            # 代码质量评分  
            orig_code_score = enhanced_code_score(orig_resp)  
            lora_code_score = enhanced_code_score(lora_resp)  
            
            total_original_code += orig_code_score  
            total_lora_code += lora_code_score  
            
            print(f"\n📝 测试 {i}: {prompt[:50]}...")  
            print(f"  🔵 原始模型代码分: {orig_code_score}/100")  
            print(f"  🟢 LoRA模型代码分: {lora_code_score}/100")  
            print(f"  📈 代码改进: {lora_code_score - orig_code_score:+d}")  
            
            # 对关键用例进行图像评估  
            if prompt in key_prompts:  
                print(f"  🎨 正在渲染图像评估...")  
                
                # 渲染原始模型图像  
                orig_image = evaluator.run_blender_script_and_render(orig_resp, f"orig_{i}")  
                if orig_image:  
                    orig_image_score = evaluator.evaluate_image_with_openai(orig_image, prompt)  
                    total_original_image += orig_image_score.get('total_score', 0)  
                    print(f"  🔵 原始图像分: {orig_image_score.get('total_score', 0)}/100")  
                
                # 渲染LoRA模型图像  
                lora_image = evaluator.run_blender_script_and_render(lora_resp, f"lora_{i}")  
                if lora_image:  
                    lora_image_score = evaluator.evaluate_image_with_openai(lora_image, prompt)  
                    total_lora_image += lora_image_score.get('total_score', 0)  
                    print(f"  🟢 LoRA图像分: {lora_image_score.get('total_score', 0)}/100")  
                    print(f"  📈 图像改进: {lora_image_score.get('total_score', 0) - orig_image_score.get('total_score', 0):+d}")  
        
        # 综合评估结果  
        print(f"\n🎯 综合评估结果:")  
        print(f"  📝 代码质量:")  
        print(f"    原始模型: {total_original_code}/{len(self.test_prompts)*100}")  
        print(f"    LoRA模型: {total_lora_code}/{len(self.test_prompts)*100}")  
        print(f"    改进: {total_lora_code - total_original_code:+d} ({(total_lora_code-total_original_code)/max(total_original_code,1)*100:+.1f}%)")  
        
        if total_original_image > 0:  
            print(f"  🎨 图像质量:")  
            print(f"    原始模型: {total_original_image}/{len(key_prompts)*100}")  
            print(f"    LoRA模型: {total_lora_image}/{len(key_prompts)*100}")  
            print(f"    改进: {total_lora_image - total_original_image:+d} ({(total_lora_image-total_original_image)/max(total_original_image,1)*100:+.1f}%)")  
        
        return {  
            'code_scores': {'original': total_original_code, 'lora': total_lora_code},  
            'image_scores': {'original': total_original_image, 'lora': total_lora_image}  
        }

    def create_comparison_report(self, comparison_result):  
        """创建对比报告"""  
        report = f"""# BlenderLLM 模型对比报告  

## 📊 评估结果  

**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**测试样本**: {len(self.test_prompts)} 个椅子设计提示  

### 🎯 总体性能  

| 模型 | 总分 | 平均分 |  
|------|------|--------|  
| 原始BlenderLLM | {comparison_result['original_total']}/{len(self.test_prompts)*10} | {comparison_result['original_total']/len(self.test_prompts):.1f}/10 |  
| LoRA增强版本 | {comparison_result['lora_total']}/{len(self.test_prompts)*10} | {comparison_result['lora_total']/len(self.test_prompts):.1f}/10 |  

### 📈 改进效果  

- **总分改进**: {comparison_result['improvement']:+d}  
- **百分比改进**: {comparison_result['improvement']/comparison_result['original_total']*100:+.1f}%  

### 🎯 结论  

{'🎉 LoRA微调显著提升了模型性能！' if comparison_result['improvement'] > 2 else '✅ LoRA微调有效改善了模型能力' if comparison_result['improvement'] > 0 else '⚠️ 需要进一步优化训练策略'}  

## 📝 详细响应  

详细的模型响应请查看:  
- `original_responses.json` - 原始模型响应  
- `lora_responses.json` - LoRA模型响应  
"""  
        
        with open('./output/evaluation_results/comparison_report.md', 'w') as f:  
            f.write(report)  
        
        print("📋 对比报告已保存: ./output/evaluation_results/comparison_report.md")  


    def create_comprehensive_report(self, simple_result, enhanced_result):  
        """创建综合评估报告"""  
        print("\n" + "="*60)  
        print("📊 BlenderLLM椅子设计微调 - 综合评估报告")  
        print("="*60)  
        
        # 简单评估结果  
        print(f"\n📝 基础代码评估:")  
        print(f"  原始模型: {simple_result['original_total']}/200")  
        print(f"  LoRA模型: {simple_result['lora_total']}/200")   
        print(f"  改进幅度: {simple_result['improvement']:+d} ({simple_result['improvement']/simple_result['original_total']*100:+.1f}%)")  
        
        # 增强评估结果  
        if enhanced_result.get('code_scores'):  
            code_scores = enhanced_result['code_scores']  
            print(f"\n🔧 增强代码质量评估:")  
            print(f"  原始模型: {code_scores['original']}/{len(self.test_prompts)*100}")  
            print(f"  LoRA模型: {code_scores['lora']}/{len(self.test_prompts)*100}")  
            improvement = code_scores['lora'] - code_scores['original']  
            print(f"  改进幅度: {improvement:+d} ({improvement/max(code_scores['original'],1)*100:+.1f}%)")  
        
        # 图像质量评估结果  
        if enhanced_result.get('image_scores') and enhanced_result['image_scores']['original'] > 0:  
            image_scores = enhanced_result['image_scores']  
            print(f"\n🎨 图像质量评估:")  
            print(f"  原始模型: {image_scores['original']}/500")  # 5个关键样本 * 100分  
            print(f"  LoRA模型: {image_scores['lora']}/500")  
            improvement = image_scores['lora'] - image_scores['original']  
            print(f"  改进幅度: {improvement:+d} ({improvement/max(image_scores['original'],1)*100:+.1f}%)")  
        
        # 保存详细报告到文件  
        report_path = "./output/evaluation_report.txt"  
        with open(report_path, 'w', encoding='utf-8') as f:  
            f.write("BlenderLLM椅子设计微调 - 详细评估报告\n")  
            f.write("="*50 + "\n\n")  
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  
            f.write(f"测试样本数: {len(self.test_prompts)}\n\n")  
            
            # 写入详细结果...  
        
        print(f"\n📄 详细报告已保存: {report_path}")  

def main():  
    """主函数"""  
    print("🚀 分离式BlenderLLM模型对比评估")  
    print("=" * 50)  
    
    evaluator = SeparateModelEvaluator()  
    
    try:  
        # 评估原始模型  
        original_responses = evaluator.evaluate_original_model()  
        
        # 等待一段时间确保内存完全释放  
        print("\n⏳ 等待内存释放...")  
        time.sleep(3)  
        
        # 评估LoRA模型  
        lora_responses = evaluator.evaluate_lora_model()  
        
        # 对比结果  
        comparison_result = evaluator.simple_compare_responses(original_responses, lora_responses)  

        # 更仔细对比
        comparison_result_more = evaluator.enhanced_compare_responses(original_responses, lora_responses) 
        
        # 生成报告  
        # evaluator.create_comparison_report(comparison_result)  
        evaluator.create_comprehensive_report(comparison_result, comparison_result_more)  
        
        print("\n✅ 评估完成！")  
        
    except Exception as e:  
        print(f"❌ 评估失败: {e}")  
        import traceback  
        traceback.print_exc()  

if __name__ == "__main__":  
    main()  
