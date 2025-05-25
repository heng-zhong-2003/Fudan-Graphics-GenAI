"""  
对比原始模型和微调模型的性能  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json  
import os  
from datetime import datetime  

class ModelComparator:  
    def __init__(self):  
        self.test_prompts = [  
            "Generate chair design: modern minimalist chair",  
            "Generate chair design: vintage wooden armchair",  
            "Generate chair design: ergonomic office chair with lumbar support",  
            "Generate chair design: outdoor metal bar stool",  
            "Generate chair design: luxury leather executive chair"  
        ]  
    
    def load_model(self, model_path, model_name):  
        """加载模型"""  
        print(f"🔄 加载 {model_name}...")  
        try:  
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
            model = AutoModelForCausalLM.from_pretrained(  
                model_path,  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                device_map="auto"  
            )  
            print(f"✅ {model_name} 加载成功")  
            return model, tokenizer  
        except Exception as e:  
            print(f"❌ {model_name} 加载失败: {e}")  
            return None, None  
    
    def generate_and_evaluate(self, model, tokenizer, prompt):  
        """生成并评估单个提示"""  
        try:  
            inputs = tokenizer(prompt, return_tensors="pt")  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 400,  
                    temperature=0.7,  
                    do_sample=True,  
                    top_p=0.9,  
                    pad_token_id=tokenizer.eos_token_id,  
                    repetition_penalty=1.1  
                )  
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(prompt):].strip()  
            
            # 评估生成质量  
            score = self.evaluate_quality(generated)  
            
            return {  
                'prompt': prompt,  
                'generated': generated,  
                'score': score,  
                'length': len(generated),  
                'success': True  
            }  
            
        except Exception as e:  
            return {  
                'prompt': prompt,  
                'generated': '',  
                'score': 0,  
                'length': 0,  
                'success': False,  
                'error': str(e)  
            }  
    
    def evaluate_quality(self, generated_code):  
        """评估生成代码的质量"""  
        score = 0  
        max_score = 10  
        
        # 基础检查  
        if 'import bpy' in generated_code:  
            score += 2  
        
        # 椅子相关元素  
        chair_elements = ['seat', 'leg', 'back', 'chair', 'armrest']  
        if any(elem in generated_code.lower() for elem in chair_elements):  
            score += 2  
        
        # Blender操作  
        blender_ops = ['add_object', 'primitive', 'mesh.', 'location', 'scale']  
        if any(op in generated_code for op in blender_ops):  
            score += 2  
        
        # 代码结构  
        if generated_code.count('\n') >= 5:  # 多行代码  
            score += 1  
        
        if '#' in generated_code:  # 有注释  
            score += 1  
        
        # 语法检查（简单）  
        if generated_code.count('(') == generated_code.count(')'):  
            score += 1  
        
        # 长度合理性  
        if 100 < len(generated_code) < 2000:  
            score += 1  
        
        return min(score, max_score)  
    
    def compare_models(self):  
        """对比两个模型"""  
        print("🔍 椅子设计模型对比评估")  
        print("=" * 60)  
        
        # 加载原始模型  
        original_model, original_tokenizer = self.load_model(  
            "../models/BlenderLLM", "原始模型"  
        )  
        
        # 加载微调模型  
        finetuned_model, finetuned_tokenizer = self.load_model(  
            "./output/memory_optimized_model", "微调模型"  
        )  
        
        if not original_model or not finetuned_model:  
            print("❌ 模型加载失败，无法进行对比")  
            return  
        
        results = {  
            'original': [],  
            'finetuned': [],  
            'comparison_time': datetime.now().isoformat()  
        }  
        
        print("\n🧪 开始对比测试...")  
        
        for i, prompt in enumerate(self.test_prompts):  
            print(f"\n--- 测试 {i+1}/{len(self.test_prompts)} ---")  
            print(f"📝 提示: {prompt}")  
            
            # 测试原始模型  
            print("🔄 测试原始模型...")  
            original_result = self.generate_and_evaluate(  
                original_model, original_tokenizer, prompt  
            )  
            results['original'].append(original_result)  
            
            if original_result['success']:  
                print(f"✅ 原始模型 - 分数: {original_result['score']}/10, 长度: {original_result['length']}")  
            else:  
                print(f"❌ 原始模型生成失败")  
            
            # 测试微调模型  
            print("🔄 测试微调模型...")  
            finetuned_result = self.generate_and_evaluate(  
                finetuned_model, finetuned_tokenizer, prompt  
            )  
            results['finetuned'].append(finetuned_result)  
            
            if finetuned_result['success']:  
                print(f"✅ 微调模型 - 分数: {finetuned_result['score']}/10, 长度: {finetuned_result['length']}")  
            else:  
                print(f"❌ 微调模型生成失败")  
            
            # 对比  
            if original_result['success'] and finetuned_result['success']:  
                if finetuned_result['score'] > original_result['score']:  
                    print("🎯 微调模型表现更好!")  
                elif finetuned_result['score'] < original_result['score']:  
                    print("📉 原始模型表现更好")  
                else:  
                    print("🤝 两模型表现相当")  
        
        # 生成对比报告  
        self.generate_comparison_report(results)  
        
        return results  
    
    def generate_comparison_report(self, results):  
        """生成详细的对比报告"""  
        print("\n" + "=" * 60)  
        print("📊 对比报告")  
        print("=" * 60)  
        
        # 计算总体统计  
        original_scores = [r['score'] for r in results['original'] if r['success']]  
        finetuned_scores = [r['score'] for r in results['finetuned'] if r['success']]  
        
        original_success_rate = len(original_scores) / len(results['original']) * 100  
        finetuned_success_rate = len(finetuned_scores) / len(results['finetuned']) * 100  
        
        original_avg_score = sum(original_scores) / len(original_scores) if original_scores else 0  
        finetuned_avg_score = sum(finetuned_scores) / len(finetuned_scores) if finetuned_scores else 0  
        
        print(f"📈 成功率对比:")  
        print(f"  原始模型: {original_success_rate:.1f}% ({len(original_scores)}/{len(results['original'])})")  
        print(f"  微调模型: {finetuned_success_rate:.1f}% ({len(finetuned_scores)}/{len(results['finetuned'])})")  
        
        print(f"\n🎯 平均质量分数:")  
        print(f"  原始模型: {original_avg_score:.2f}/10")  
        print(f"  微调模型: {finetuned_avg_score:.2f}/10")  
        
        # 详细对比  
        print(f"\n📋 详细对比:")  
        for i, (orig, fine) in enumerate(zip(results['original'], results['finetuned'])):  
            prompt = orig['prompt'][:50] + "..." if len(orig['prompt']) > 50 else orig['prompt']  
            
            orig_status = f"{orig['score']}/10" if orig['success'] else "失败"  
            fine_status = f"{fine['score']}/10" if fine['success'] else "失败"  
            
            winner = ""  
            if orig['success'] and fine['success']:  
                if fine['score'] > orig['score']:  
                    winner = "📈 微调胜"  
                elif fine['score'] < orig['score']:  
                    winner = "📉 原始胜"  
                else:  
                    winner = "🤝 平局"  
            
            print(f"  {i+1}. {prompt}")  
            print(f"     原始: {orig_status} | 微调: {fine_status} {winner}")  
        
        # 保存详细结果  
        os.makedirs("./output/model_comparison", exist_ok=True)  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        
        with open(f"./output/model_comparison/comparison_{timestamp}.json", 'w', encoding='utf-8') as f:  
            json.dump(results, f, indent=2, ensure_ascii=False)  
        
        # 生成代码示例对比  
        self.generate_code_examples(results, timestamp)  
        
        print(f"\n💾 详细结果保存到: ./output/model_comparison/comparison_{timestamp}.json")  
    
    def generate_code_examples(self, results, timestamp):  
        """生成代码示例对比文件"""  
        examples_file = f"./output/model_comparison/code_examples_{timestamp}.md"  
        
        with open(examples_file, 'w', encoding='utf-8') as f:  
            f.write("# 椅子设计代码生成对比\n\n")  
            f.write(f"生成时间: {results['comparison_time']}\n\n")  
            
            for i, (orig, fine) in enumerate(zip(results['original'], results['finetuned'])):  
                f.write(f"## 测试 {i+1}: {orig['prompt']}\n\n")  
                
                f.write(f"### 原始模型 (分数: {orig['score'] if orig['success'] else '失败'}/10)\n")  
                if orig['success']:  
                    f.write("```python\n")  
                    f.write(orig['generated'])  
                    f.write("\n```\n\n")  
                else:  
                    f.write("❌ 生成失败\n\n")  
                
                f.write(f"### 微调模型 (分数: {fine['score'] if fine['success'] else '失败'}/10)\n")  
                if fine['success']:  
                    f.write("```python\n")  
                    f.write(fine['generated'])  
                    f.write("\n```\n\n")  
                else:  
                    f.write("❌ 生成失败\n\n")  
                
                f.write("---\n\n")  
        
        print(f"📝 代码示例保存到: {examples_file}")  

if __name__ == "__main__":  
    comparator = ModelComparator()  
    comparator.compare_models()  
