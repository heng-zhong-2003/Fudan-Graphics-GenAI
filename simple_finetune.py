#!/usr/bin/env python3  
"""  
简化的微调方法 - 避免device错误  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json  
import os  
from torch.optim import AdamW  
from torch.nn import CrossEntropyLoss  

def simple_finetune():  
    print("🔧 简化微调椅子设计模型...")  
    
    # 清理GPU内存  
    torch.cuda.empty_cache()  
    
    # 加载数据  
    data_file = "./output/new_training_data/chair_training_data.json"  
    with open(data_file, 'r', encoding='utf-8') as f:  
        training_data = json.load(f)  
    
    # 只用很少的数据  
    training_data = training_data[:5]  
    print(f"📊 使用 {len(training_data)} 个样本")  
    
    # 加载模型 - 不使用device_map  
    print("🔄 加载模型...")  
    tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
    
    # 直接加载到GPU  
    model = AutoModelForCausalLM.from_pretrained(  
        "../models/BlenderLLM",  
        trust_remote_code=True,  
        torch_dtype=torch.float32,  
        low_cpu_mem_usage=True  
    ).cuda()  
    
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
    
    # 只训练最后的输出层  
    for param in model.parameters():  
        param.requires_grad = False  
    
    # 只训练lm_head  
    if hasattr(model, 'lm_head'):  
        for param in model.lm_head.parameters():  
            param.requires_grad = True  
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"🎯 可训练参数: {trainable_params:,}")  
    
    # 优化器  
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)  
    loss_fn = CrossEntropyLoss()  
    
    model.train()  
    
    print("🏋️ 开始训练...")  
    
    for epoch in range(2):  # 只训练2个epoch  
        total_loss = 0  
        
        for i, item in enumerate(training_data):  
            text = f"Human: {item['input']}\n\nAssistant: {item['output']}"  
            
            # Tokenize  
            inputs = tokenizer(  
                text,  
                return_tensors="pt",  
                max_length=512,  
                truncation=True,  
                padding=True  
            ).to('cuda')  
            
            # 前向传播  
            outputs = model(**inputs, labels=inputs['input_ids'])  
            loss = outputs.loss  
            
            if torch.isnan(loss):  
                print(f"⚠️ NaN loss at step {i}, skipping...")  
                continue  
            
            # 反向传播  
            optimizer.zero_grad()  
            loss.backward()  
            
            # 梯度裁剪  
            torch.nn.utils.clip_grad_norm_(  
                [p for p in model.parameters() if p.requires_grad],   
                max_norm=1.0  
            )  
            
            optimizer.step()  
            
            total_loss += loss.item()  
            print(f"Epoch {epoch+1}, Step {i+1}/{len(training_data)}, Loss: {loss.item():.4f}")  
        
        avg_loss = total_loss / len(training_data)  
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")  
    
    print("✅ 训练完成")  
    
    # 保存模型  
    output_dir = "./output/simple_finetuned_model"  
    os.makedirs(output_dir, exist_ok=True)  
    
    model.save_pretrained(output_dir)  
    tokenizer.save_pretrained(output_dir)  
    
    print(f"💾 模型保存到: {output_dir}")  
    
    # 测试  
    print("\n🧪 测试微调后的模型...")  
    model.eval()  
    
    test_prompt = "Generate chair design: simple wooden chair"  
    
    try:  
        inputs = tokenizer(test_prompt, return_tensors="pt").to('cuda')  
        
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,  
                max_length=len(inputs['input_ids'][0]) + 200,  
                temperature=0.7,  
                do_sample=True,  
                pad_token_id=tokenizer.eos_token_id  
            )  
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
        generated = result[len(test_prompt):].strip()  
        
        print(f"📝 测试输入: {test_prompt}")  
        print(f"🤖 生成结果: {generated[:300]}...")  
        
    except Exception as e:  
        print(f"❌ 测试失败: {e}")  

if __name__ == "__main__":  
    simple_finetune()  
