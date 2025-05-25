#!/usr/bin/env python3  
"""  
测试所有训练的模型  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  
from pathlib import Path  

def test_model(model_path, model_name):  
    """测试单个模型"""  
    print(f"\n🧪 测试模型: {model_name}")  
    print(f"📁 路径: {model_path}")  
    
    try:  
        # 检查模型文件是否存在  
        if not Path(model_path).exists():  
            print(f"❌ 模型路径不存在")  
            return  
        
        # 检查必要文件  
        config_file = Path(model_path) / "config.json"  
        model_file = Path(model_path) / "pytorch_model.bin"  
        safetensors_file = Path(model_path) / "model.safetensors"  
        
        if not config_file.exists():  
            print(f"❌ 缺少 config.json")  
            return  
        
        if not model_file.exists() and not any(Path(model_path).glob("*.safetensors")):  
            print(f"❌ 缺少模型权重文件")  
            return  
        
        # 加载模型  
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
        model = AutoModelForCausalLM.from_pretrained(  
            model_path,   
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map="auto"  
        )  
        
        print(f"✅ 模型加载成功")  
        
        # 测试生成  
        test_prompts = [  
            "User: Generate chair design: modern\nAssistant:",  
            "User: Create a simple wooden chair\nAssistant:",  
            "User: Design a comfortable office chair\nAssistant:"  
        ]  
        
        for i, prompt in enumerate(test_prompts):  
            try:  
                inputs = tokenizer(prompt, return_tensors="pt")  
                
                with torch.no_grad():  
                    outputs = model.generate(  
                        **inputs,  
                        max_length=len(inputs['input_ids'][0]) + 150,  
                        temperature=0.7,  
                        do_sample=True,  
                        pad_token_id=tokenizer.eos_token_id,  
                        repetition_penalty=1.1,  
                        top_p=0.9  
                    )  
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
                generated = result[len(prompt):].strip()  
                
                print(f"📝 测试 {i+1}:")  
                print(f"   输入: {prompt.split('Assistant:')[0].replace('User: ', '').strip()}")  
                print(f"   生成: {generated[:200]}...")  
                
                # 检查是否包含椅子相关内容  
                chair_keywords = ['chair', 'seat', 'backrest', 'leg', 'armrest', 'cushion', 'wood', 'metal']  
                if any(keyword.lower() in generated.lower() for keyword in chair_keywords):  
                    print(f"   ✅ 包含椅子相关内容")  
                else:  
                    print(f"   ⚠️  可能不是椅子相关内容")  
                
            except Exception as e:  
                print(f"❌ 生成失败: {e}")  
        
        # 清理内存  
        del model, tokenizer  
        torch.cuda.empty_cache()  
        
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  

def main():  
    print("🧪 测试所有训练的模型")  
    
    # 模型列表  
    models = [  
        ("output/fixed_minimal_model", "Fixed Minimal Model"),  
        ("output/minimal_model", "Minimal Model"),  
        ("output/ultra_light_model", "Ultra Light Model"),  
        ("output/simple_model", "Simple Model"),  
        ("output/fixed_model", "Fixed Model"),  
        ("output/test_model", "Test Model"),  
    ]  
    
    # 基础模型作为对比  
    base_model = ("../models/BlenderLLM", "Original BlenderLLM")  
    
    print(f"\n🔍 首先测试原始模型作为基准:")  
    test_model(base_model[0], base_model[1])  
    
    print(f"\n" + "="*60)  
    print(f"测试微调后的模型:")  
    
    for model_path, model_name in models:  
        test_model(model_path, model_name)  
        print("-" * 40)  

if __name__ == "__main__":  
    main()  
