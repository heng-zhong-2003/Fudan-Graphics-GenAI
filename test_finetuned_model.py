#!/usr/bin/env python3  
"""  
测试微调后的模型效果  
"""  

import os  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json  

def load_original_model():  
    """加载原始模型"""  
    print("🔄 加载原始BlenderLLM模型...")  
    try:  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        return model, tokenizer  
    except Exception as e:  
        print(f"❌ 原始模型加载失败: {e}")  
        return None, None  

def load_finetuned_model():  
    """加载微调后的模型"""  
    print("🔄 加载微调后的模型...")  
    
    # 先加载原始模型  
    base_model, tokenizer = load_original_model()  
    if base_model is None:  
        return None, None  
    
    # 加载微调的参数  
    try:  
        finetuned_params_path = "./output/emergency_model/trainable_params.pt"  
        if os.path.exists(finetuned_params_path):  
            finetuned_params = torch.load(finetuned_params_path, map_location="cpu")  
            
            # 更新模型参数  
            updated_count = 0  
            for name, param in base_model.named_parameters():  
                if name in finetuned_params:  
                    param.data = finetuned_params[name].to(param.device)  
                    updated_count += 1  
            
            print(f"✅ 更新了 {updated_count} 个参数")  
            return base_model, tokenizer  
        else:  
            print("❌ 微调参数文件不存在")  
            return None, None  
            
    except Exception as e:  
        print(f"❌ 微调模型加载失败: {e}")  
        return None, None  

def test_generation(model, tokenizer, model_name, test_prompts):  
    """测试生成效果"""  
    print(f"\n🧪 测试 {model_name}")  
    print("-" * 50)  
    
    results = []  
    
    for i, prompt in enumerate(test_prompts):  
        print(f"\n📝 测试 {i+1}: {prompt}")  
        
        try:  
            # 使用提示模板  
            full_prompt = f"Generate chair design: {prompt}"  
            
            inputs = tokenizer(full_prompt, return_tensors="pt")  
            device = next(model.parameters()).device  
            inputs = {k: v.to(device) for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 150,  
                    temperature=0.7,  
                    do_sample=True,  
                    top_p=0.9,  
                    pad_token_id=tokenizer.eos_token_id,  
                    repetition_penalty=1.1  
                )  
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(full_prompt):].strip()  
            
            print(f"🤖 生成结果:")  
            print(generated[:300] + "..." if len(generated) > 300 else generated)  
            
            # 简单评分  
            score = evaluate_output(generated, prompt)  
            print(f"📊 评分: {score}/10")  
            
            results.append({  
                'prompt': prompt,  
                'output': generated,  
                'score': score  
            })  
            
        except Exception as e:  
            print(f"❌ 生成失败: {e}")  
            results.append({  
                'prompt': prompt,  
                'output': "",  
                'score': 0  
            })  
    
    return results  

def evaluate_output(output, prompt):  
    """简单的输出评估"""  
    score = 0  
    output_lower = output.lower()  
    prompt_lower = prompt.lower()  
    
    # 检查是否包含Blender代码  
    if 'import bpy' in output:  
        score += 3  
    
    # 检查是否包含椅子相关词汇  
    chair_keywords = ['chair', 'seat', 'leg', 'back', 'arm']  
    if any(keyword in output_lower for keyword in chair_keywords):  
        score += 2  
    
    # 检查是否包含Blender操作  
    blender_ops = ['primitive', 'cube', 'cylinder', 'mesh', 'add', 'location', 'scale']  
    if any(op in output_lower for op in blender_ops):  
        score += 2  
    
    # 检查是否响应了特定提示  
    if any(word in output_lower for word in prompt_lower.split()):  
        score += 2  
    
    # 检查代码质量  
    if 'bpy.ops' in output:  
        score += 1  
    
    return min(score, 10)  

def compare_models():  
    """对比模型效果"""  
    print("🔬 BlenderLLM 微调效果对比测试")  
    print("=" * 60)  
    
    # 测试提示  
    test_prompts = [  
        "wooden dining chair",  
        "modern office chair with wheels",  
        "simple stool",  
        "comfortable armchair",  
        "bar stool with back support"  
    ]  
    
    # 加载原始模型  
    original_model, tokenizer = load_original_model()  
    if original_model is None:  
        print("❌ 无法加载原始模型")  
        return  
    
    # 测试原始模型  
    print("\n" + "="*60)  
    original_results = test_generation(original_model, tokenizer, "原始BlenderLLM", test_prompts)  
    
    # 清理内存  
    del original_model  
    torch.cuda.empty_cache()  
    
    # 加载微调模型  
    finetuned_model, tokenizer = load_finetuned_model()  
    if finetuned_model is None:  
        print("❌ 无法加载微调模型")  
        return  
    
    # 测试微调模型  
    print("\n" + "="*60)  
    finetuned_results = test_generation(finetuned_model, tokenizer, "微调后BlenderLLM", test_prompts)  
    
    # 对比结果  
    print("\n" + "="*60)  
    print("📊 对比结果汇总")  
    print("="*60)  
    
    original_avg = sum(r['score'] for r in original_results) / len(original_results)  
    finetuned_avg = sum(r['score'] for r in finetuned_results) / len(finetuned_results)  
    
    print(f"📈 原始模型平均分: {original_avg:.2f}/10")  
    print(f"📈 微调模型平均分: {finetuned_avg:.2f}/10")  
    print(f"📈 改进幅度: {finetuned_avg - original_avg:+.2f}")  
    
    if finetuned_avg > original_avg:  
        print("✅ 微调有效！模型性能提升")  
    elif finetuned_avg == original_avg:  
        print("➡️ 微调效果中性")  
    else:  
        print("⚠️ 微调可能导致性能下降")  
    
    # 保存详细结果  
    results = {  
        'original': original_results,  
        'finetuned': finetuned_results,  
        'summary': {  
            'original_avg': original_avg,  
            'finetuned_avg': finetuned_avg,  
            'improvement': finetuned_avg - original_avg  
        }  
    }  
    
    os.makedirs("./output/evaluation", exist_ok=True)  
    with open("./output/evaluation/comparison_results.json", 'w') as f:  
        json.dump(results, f, indent=2, ensure_ascii=False)  
    
    print(f"\n💾 详细结果已保存到: ./output/evaluation/comparison_results.json")  

if __name__ == "__main__":  
    compare_models()  
