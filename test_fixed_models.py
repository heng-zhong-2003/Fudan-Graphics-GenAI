#!/usr/bin/env python3  
"""  
测试修复后的模型效果  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  
import json  

def load_model_with_params(params_path, model_name):  
    """加载带有特定参数的模型"""  
    print(f"🔄 加载 {model_name}...")  
    
    try:  
        # 加载基础模型  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        
        # 加载参数  
        if os.path.exists(params_path):  
            custom_params = torch.load(params_path, map_location="cpu")  
            
            updated_count = 0  
            for name, param in model.named_parameters():  
                if name in custom_params:  
                    # 检查参数是否安全  
                    new_param = custom_params[name]  
                    if not (torch.isnan(new_param).any() or torch.isinf(new_param).any()):  
                        param.data = new_param.to(param.device)  
                        updated_count += 1  
                        print(f"✅ 更新参数: {name}")  
                    else:  
                        print(f"⚠️ 跳过损坏参数: {name}")  
            
            print(f"✅ 总共更新了 {updated_count} 个参数")  
        else:  
            print(f"⚠️ 参数文件不存在: {params_path}")  
        
        return model, tokenizer  
        
    except Exception as e:  
        print(f"❌ 加载失败: {e}")  
        return None, None  

def safe_generate(model, tokenizer, prompt):  
    """安全生成（避免CUDA错误）"""  
    try:  
        inputs = tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)  
        device = next(model.parameters()).device  
        inputs = {k: v.to(device) for k, v in inputs.items()}  
        
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,  
                max_length=len(inputs['input_ids'][0]) + 100,  
                temperature=0.5,  # 更保守的温度  
                do_sample=True,  
                top_p=0.8,  
                pad_token_id=tokenizer.eos_token_id,  
                repetition_penalty=1.1,  
                num_return_sequences=1  
            )  
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
        generated = result[len(prompt):].strip()  
        return generated  
        
    except Exception as e:  
        print(f"❌ 生成失败: {e}")  
        return ""  

def test_all_available_models():  
    """测试所有可用的模型版本"""  
    print("🧪 测试所有可用模型")  
    print("=" * 60)  
    
    test_prompt = "Generate chair design: simple wooden chair"  
    
    # 模型列表  
    models_to_test = [  
        ("原始模型", None),  
        ("修复模型", "./output/fixed_model/trainable_params.pt"),  
        ("安全模型", "./output/safe_model/trainable_params.pt"),  
    ]  
    
    results = {}  
    
    for model_name, params_path in models_to_test:  
        print(f"\n{'='*20} {model_name} {'='*20}")  
        
        if params_path is None:  
            # 原始模型  
            try:  
                tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
                model = AutoModelForCausalLM.from_pretrained(  
                    "../models/BlenderLLM",  
                    trust_remote_code=True,  
                    torch_dtype=torch.float16,  
                    device_map={"": 0}  
                )  
                print("✅ 原始模型加载成功")  
            except Exception as e:  
                print(f"❌ 原始模型加载失败: {e}")  
                continue  
        else:  
            # 带参数的模型  
            if not os.path.exists(params_path):  
                print(f"⚠️ 跳过不存在的模型: {params_path}")  
                continue  
            
            model, tokenizer = load_model_with_params(params_path, model_name)  
            if model is None:  
                continue  
        
        # 测试生成  
        print(f"🎯 测试提示: {test_prompt}")  
        generated = safe_generate(model, tokenizer, test_prompt)  
        
        if generated:  
            print(f"✅ 生成成功 ({len(generated)} 字符)")  
            print("生成内容预览:")  
            print(generated[:200] + "..." if len(generated) > 200 else generated)  
            
            # 简单评分  
            score = 0  
            if 'import bpy' in generated: score += 3  
            if any(word in generated.lower() for word in ['chair', 'cube', 'cylinder']): score += 2  
            if 'bpy.ops' in generated: score += 2  
            if len(generated) > 50: score += 1  
            if not any(word in generated for word in ['error', 'failed']): score += 2  
            
            print(f"📊 质量评分: {score}/10")  
            results[model_name] = {"success": True, "score": score, "length": len(generated)}  
        else:  
            print("❌ 生成失败")  
            results[model_name] = {"success": False, "score": 0, "length": 0}  
        
        # 清理内存  
        del model  
        torch.cuda.empty_cache()  
        print("🧹 内存已清理")  
    
    # 总结  
    print("\n" + "="*60)  
    print("📊 测试结果总结")  
    print("="*60)  
    
    for model_name, result in results.items():  
        status = "✅ 成功" if result["success"] else "❌ 失败"  
        print(f"{model_name}: {status}, 评分: {result['score']}/10")  
    
    # 推荐  
    successful_models = [name for name, result in results.items() if result["success"]]  
    if successful_models:  
        best_model = max(successful_models, key=lambda x: results[x]["score"])  
        print(f"\n🏆 推荐使用: {best_model}")  
    else:  
        print("\n⚠️ 所有模型都有问题，建议检查环境")  
    
    return results  

if __name__ == "__main__":  
    test_all_available_models()  
