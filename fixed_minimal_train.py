#!/usr/bin/env python3  
"""  
修复版最小化训练  
"""  

import torch  
import json  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  
import numpy as np  

# 强制单GPU  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  

def fixed_minimal_train():  
    print("🪑 Fixed Minimal Chair Training...")  
    
    # 清理内存  
    torch.cuda.empty_cache()  
    
    # 加载模型  
    print("Loading model...")  
    tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
    
    model = AutoModelForCausalLM.from_pretrained(  
        "../models/BlenderLLM",  
        trust_remote_code=True,  
        torch_dtype=torch.float16,  
        device_map="auto",  
        low_cpu_mem_usage=True  
    )  
    
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.pad_token_id = tokenizer.eos_token_id  
    
    # 冻结大部分参数  
    total_params = 0  
    trainable_params = 0  
    
    for name, param in model.named_parameters():  
        total_params += param.numel()  
        param.requires_grad = False  
    
    # 只解冻最后几层和输出层  
    for name, param in model.named_parameters():  
        if any(layer in name for layer in ['layers.31', 'layers.30', 'lm_head', 'embed_tokens']):  
            param.requires_grad = True  
            trainable_params += param.numel()  
            print(f"✅ Training: {name}")  
    
    print(f"📊 Total params: {total_params:,}")  
    print(f"🎯 Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")  
    
    # 启用梯度检查点  
    model.gradient_checkpointing_enable()  
    
    # 加载和清理数据  
    with open('./output/cleaned_data/cleaned_data.json', 'r') as f:  
        data = json.load(f)  
    
    # 过滤和清理数据  
    clean_data = []  
    for item in data:  
        input_text = item.get('input', '').strip()  
        output_text = item.get('output', '').strip()  
        
        # 跳过空数据  
        if not input_text or not output_text:  
            continue  
        
        # 限制长度  
        if len(input_text) > 200:  
            input_text = input_text[:200]  
        if len(output_text) > 300:  
            output_text = output_text[:300]  
        
        clean_data.append({  
            'input': input_text,  
            'output': output_text  
        })  
    
    # 只用前30个样本进行快速测试  
    clean_data = clean_data[:30]  
    print(f"Using {len(clean_data)} clean samples")  
    
    # 显示样本  
    if clean_data:  
        sample = clean_data[0]  
        print(f"�� Sample input: {sample['input'][:80]}...")  
        print(f"📝 Sample output: {sample['output'][:80]}...")  
    
    # 优化器设置  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],   
        lr=1e-6,  # 更小的学习率  
        weight_decay=0.01,  
        eps=1e-8  
    )  
    
    model.train()  
    
    print("🏋️ Starting training...")  
    
    for epoch in range(1):  
        total_loss = 0  
        valid_steps = 0  
        
        for i, item in enumerate(clean_data):  
            if i % 5 == 0:  
                print(f"Step {i}/{len(clean_data)}")  
                torch.cuda.empty_cache()  
            
            # 构建训练文本  
            input_text = item['input']  
            output_text = item['output']  
            full_text = f"User: {input_text}\nAssistant: {output_text}"  
            
            # Tokenize  
            inputs = tokenizer(  
                full_text,   
                return_tensors="pt",   
                max_length=256,  # 增加长度  
                truncation=True,   
                padding=True  
            )  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            # 前向传播  
            try:  
                outputs = model(**inputs, labels=inputs['input_ids'])  
                loss = outputs.loss  
                
                # 检查损失是否有效  
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"⚠️  Skipping step {i} due to invalid loss: {loss}")  
                    continue  
                
                # 梯度裁剪前的反向传播  
                loss.backward()  
                
                # 梯度裁剪  
                torch.nn.utils.clip_grad_norm_(  
                    [p for p in model.parameters() if p.requires_grad],   
                    max_norm=1.0  
                )  
                
                if (i + 1) % 2 == 0:  # 每2步更新一次  
                    optimizer.step()  
                    optimizer.zero_grad()  
                
                total_loss += loss.item()  
                valid_steps += 1  
                
            except Exception as e:  
                print(f"⚠️  Error at step {i}: {e}")  
                optimizer.zero_grad()  
                continue  
        
        avg_loss = total_loss / max(valid_steps, 1)  
        print(f"Epoch completed. Average loss: {avg_loss:.4f} (valid steps: {valid_steps})")  
    
    # 保存模型  
    print("💾 Saving model...")  
    os.makedirs("./output/fixed_minimal_model", exist_ok=True)  
    model.save_pretrained("./output/fixed_minimal_model")  
    tokenizer.save_pretrained("./output/fixed_minimal_model")  
    
    # 测试生成  
    print("🧪 Testing generation...")  
    model.eval()  
    
    test_prompts = [  
        "User: Generate chair design: modern minimalist\nAssistant:",  
        "User: Create a comfortable office chair\nAssistant:",  
        "User: Design a vintage wooden chair\nAssistant:"  
    ]  
    
    for prompt in test_prompts:  
        try:  
            inputs = tokenizer(prompt, return_tensors="pt")  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 100,  
                    temperature=0.7,  
                    do_sample=True,  
                    pad_token_id=tokenizer.pad_token_id,  
                    eos_token_id=tokenizer.eos_token_id,  
                    repetition_penalty=1.1  
                )  
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(prompt):].strip()  
            print(f"📝 Prompt: {prompt.split('Assistant:')[0]}...")  
            print(f"   Generated: {generated[:100]}...")  
            print()  
            
        except Exception as e:  
            print(f"⚠️  Generation failed for prompt: {e}")  
    
    print("✅ Fixed minimal training completed!")  

if __name__ == "__main__":  
    fixed_minimal_train()  
