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

class SeparateModelEvaluator:  
    """分离式模型评估器"""  
    
    def __init__(self):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        self.test_prompts = [  
            "Design a chair: modern minimalist office chair with wheels",  
            "Design a chair: comfortable recliner with armrests and cushions",   
            "Design a chair: dining chair with tall backrest and no armrests",  
            "Design a chair: ergonomic gaming chair with adjustable height",  
            "Design a chair: vintage wooden chair with carved details"  
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
        
        # 生成报告  
        evaluator.create_comparison_report(comparison_result)  
        
        print("\n✅ 评估完成！")  
        
    except Exception as e:  
        print(f"❌ 评估失败: {e}")  
        import traceback  
        traceback.print_exc()  

if __name__ == "__main__":  
    main()  
