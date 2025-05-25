#!/usr/bin/env python3  
"""  
测试训练好的LoRA模型  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from peft import PeftModel  
import os  

def test_lora_model():  
    print("🧪 测试LoRA增强的BlenderLLM")  
    print("=" * 40)  
    
    # 设置设备  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print(f"🎯 使用设备: {device}")  
    
    try:  
        # 1. 加载基础模型  
        print("\n🔄 加载基础模型...")  
        base_model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        
        # 2. 加载tokenizer  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        # 3. 加载LoRA适配器  
        print("🔧 加载LoRA适配器...")  
        model = PeftModel.from_pretrained(base_model, "./output/lora_blender_enhanced")  
        model.eval()  
        
        print("✅ LoRA模型加载成功!")  
        
        # 4. 测试不同类型的椅子设计  
        test_prompts = [  
            "Design a chair: modern minimalist office chair",  
            "Design a chair: comfortable recliner with armrests",  
            "Design a chair: dining chair with tall backrest"  
        ]  
        
        print("\n🎯 测试椅子设计生成:")  
        print("=" * 40)  
        
        for i, prompt in enumerate(test_prompts, 1):  
            print(f"\n🪑 测试 {i}: {prompt}")  
            
            # 构造输入  
            input_text = f"User: {prompt}\n\nAssistant:"  
            inputs = tokenizer(input_text, return_tensors="pt").to(device)  
            
            # 生成响应  
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_new_tokens=150,  
                    temperature=0.7,  
                    do_sample=True,  
                    pad_token_id=tokenizer.pad_token_id,  
                    eos_token_id=tokenizer.eos_token_id  
                )  
            
            # 解码响应  
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            assistant_response = response.split("Assistant:")[-1].strip()  
            
            print(f"🔧 生成的Blender代码:")  
            print("-" * 30)  
            print(assistant_response[:300] + "..." if len(assistant_response) > 300 else assistant_response)  
            print("-" * 30)  
        
        print("\n✅ LoRA模型测试完成!")  
        
    except Exception as e:  
        print(f"❌ 测试失败: {e}")  
        import traceback  
        traceback.print_exc()  

if __name__ == "__main__":  
    test_lora_model()  
