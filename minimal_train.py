#!/usr/bin/env python3  
"""  
最小化训练 - 仅训练部分层  
"""  

import torch  
import json  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  

# 强制单GPU  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  

def minimal_train():  
    print("🪑 Minimal Chair Training...")  
    
    # 清理内存  
    torch.cuda.empty_cache()  
    
    # 加载模型  
    print("Loading model...")  
    tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
    
    # 只加载模型的某些层进行训练  
    model = AutoModelForCausalLM.from_pretrained(  
        "../models/BlenderLLM",  
        trust_remote_code=True,  
        torch_dtype=torch.float16,  
        device_map="auto",  
        low_cpu_mem_usage=True  
    )  
    
    # 冻结大部分参数，只训练最后几层  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
    
    # 只解冻最后2层  
    layers_to_train = []  
    for name, param in model.named_parameters():  
        if 'layers.31' in name or 'layers.30' in name or 'lm_head' in name:  
            param.requires_grad = True  
            layers_to_train.append(name)  
    
    print(f"Training layers: {len(layers_to_train)}")  
    
    # 启用梯度检查点  
    model.gradient_checkpointing_enable()  
    
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
    
    # 加载少量数据  
    with open('./output/cleaned_data/cleaned_data.json', 'r') as f:  
        data = json.load(f)  
    
    # 只用前50个样本  
    data = data[:50]  
    print(f"Using {len(data)} samples")  
    
    # 简单的训练循环  
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)  
    
    model.train()  
    
    for epoch in range(1):  
        total_loss = 0  
        for i, item in enumerate(data):  
            if i % 10 == 0:  
                print(f"Step {i}/{len(data)}")  
                torch.cuda.empty_cache()  
            
            text = f"{item.get('input', '')}\n{item.get('output', '')}"  
            if len(text) > 500:  
                text = text[:500]  
            
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            outputs = model(**inputs, labels=inputs['input_ids'])  
            loss = outputs.loss  
            
            loss.backward()  
            
            if (i + 1) % 4 == 0:  # 梯度累积  
                optimizer.step()  
                optimizer.zero_grad()  
            
            total_loss += loss.item()  
        
        print(f"Epoch loss: {total_loss/len(data):.4f}")  
    
    # 保存模型  
    print("Saving model...")  
    os.makedirs("./output/minimal_model", exist_ok=True)  
    model.save_pretrained("./output/minimal_model")  
    tokenizer.save_pretrained("./output/minimal_model")  
    
    print("✅ Minimal training done!")  

if __name__ == "__main__":  
    minimal_train()  
