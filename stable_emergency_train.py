#!/usr/bin/env python3  
"""  
稳定版紧急训练 - 修复NaN/Inf问题  
"""  

import os  
import torch  
import torch.nn as nn  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json  
import gc  
import numpy as np  

def safe_load_model():  
    """安全加载模型"""  
    print("🔄 安全加载模型...")  
    
    try:  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0},  
            low_cpu_mem_usage=True  
        )  
        
        return model, tokenizer  
        
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  
        return None, None  

def check_gradients(model):  
    """检查梯度是否正常"""  
    total_norm = 0  
    param_count = 0  
    
    for name, param in model.named_parameters():  
        if param.grad is not None:  
            param_norm = param.grad.data.norm(2)  
            total_norm += param_norm.item() ** 2  
            param_count += 1  
            
            # 检查异常值  
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():  
                print(f"⚠️ 异常梯度在 {name}")  
                param.grad.zero_()  # 清零异常梯度  
    
    total_norm = total_norm ** (1. / 2)  
    print(f"📊 梯度范数: {total_norm:.4f}, 参数数: {param_count}")  
    return total_norm  

def stable_train():  
    """稳定训练函数"""  
    print("🏋️ 稳定版椅子设计微调")  
    print("=" * 50)  
    
    # 1. 加载模型  
    model, tokenizer = safe_load_model()  
    if model is None:  
        return create_dummy_model()  
    
    device = torch.device("cuda:0")  
    
    # 2. 设置训练参数 - 更保守的设置  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
        # 只训练输出层的一小部分  
        if 'lm_head' in name and 'weight' in name:  
            param.requires_grad = True  
            print(f"�� 训练参数: {name}")  
    
    # 3. 准备训练数据  
    chair_data = [  
        ("wooden dining chair", "import bpy\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.5))"),  
        ("office chair", "import bpy\nbpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0.3))"),  
        ("simple stool", "import bpy\nbpy.ops.mesh.primitive_cube_add(scale=(0.5, 0.5, 0.3))"),  
    ]  
    
    # 4. 优化器 - 非常小的学习率  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],  
        lr=1e-7,  # 极小的学习率  
        weight_decay=0.01,  
        eps=1e-8  
    )  
    
    # 5. 训练循环  
    model.train()  
    total_loss = 0  
    valid_steps = 0  
    
    for epoch in range(2):  # 只训练2个epoch  
        print(f"\n📚 Epoch {epoch + 1}/2")  
        
        for i, (prompt, target) in enumerate(chair_data):  
            try:  
                # 清理GPU内存  
                torch.cuda.empty_cache()  
                
                # 准备输入  
                text = f"Generate chair design: {prompt}\n{target}"  
                inputs = tokenizer(  
                    text,   
                    return_tensors="pt",   
                    max_length=128,  # 更短的序列  
                    truncation=True,  
                    padding=True  
                ).to(device)  
                
                # 前向传播  
                optimizer.zero_grad()  
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):  
                    outputs = model(**inputs, labels=inputs['input_ids'])  
                    loss = outputs.loss  
                
                # 检查损失是否有效  
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"  ⚠️ 跳过无效损失: {loss.item()}")  
                    continue  
                
                # 反向传播，使用梯度缩放  
                scaler = torch.cuda.amp.GradScaler()  
                scaler.scale(loss).backward()  
                
                # 检查梯度  
                grad_norm = check_gradients(model)  
                
                # 梯度裁剪  
                if grad_norm > 1.0:  
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
                
                # 优化器步骤  
                scaler.step(optimizer)  
                scaler.update()  
                
                total_loss += loss.item()  
                valid_steps += 1  
                
                print(f"  步骤 {i+1}: 损失={loss.item():.4f}, 梯度范数={grad_norm:.4f}")  
                
            except Exception as e:  
                print(f"  ❌ 训练步骤失败: {e}")  
                continue  
    
    # 6. 保存模型  
    if valid_steps > 0:  
        avg_loss = total_loss / valid_steps  
        print(f"\n📊 平均损失: {avg_loss:.4f}")  
        
        # 保存训练好的参数  
        save_model(model, avg_loss)  
    else:  
        print("❌ 没有有效的训练步骤")  
        create_dummy_model()  

def save_model(model, avg_loss):  
    """保存模型"""  
    print("💾 保存稳定训练模型...")  
    
    output_dir = "./output/stable_model"  
    os.makedirs(output_dir, exist_ok=True)  
    
    # 只保存训练过的参数  
    trainable_params = {}  
    for name, param in model.named_parameters():  
        if param.requires_grad:  
            # 检查参数是否正常  
            if not (torch.isnan(param).any() or torch.isinf(param).any()):  
                trainable_params[name] = param.cpu().clone()  
                print(f"✅ 保存参数: {name}")  
            else:  
                print(f"⚠️ 跳过异常参数: {name}")  
    
    if trainable_params:  
        torch.save(trainable_params, os.path.join(output_dir, "trainable_params.pt"))  
        
        # 保存训练信息  
        info = {  
            "status": "stable_training_completed",  
            "avg_loss": avg_loss,  
            "saved_params": list(trainable_params.keys()),  
            "timestamp": str(torch.cuda.memory_allocated(0))  
        }  
        
        with open(os.path.join(output_dir, "training_info.json"), 'w') as f:  
            json.dump(info, f, indent=2)  
        
        print(f"✅ 模型保存到: {output_dir}")  
    else:  
        print("❌ 没有有效参数可保存")  

def create_dummy_model():  
    """创建虚拟模型（当训练失败时）"""  
    print("🔧 创建虚拟模型...")  
    
    output_dir = "./output/dummy_model"  
    os.makedirs(output_dir, exist_ok=True)  
    
    info = {  
        "status": "dummy_model",  
        "message": "训练失败，使用原始模型",  
        "recommendation": "直接使用原始BlenderLLM模型"  
    }  
    
    with open(os.path.join(output_dir, "info.json"), 'w') as f:  
        json.dump(info, f, indent=2)  
    
    print("✅ 虚拟模型创建完成")  

if __name__ == "__main__":  
    # 清理GPU内存  
    torch.cuda.empty_cache()  
    gc.collect()  
    
    stable_train()  
    
    print("\n🎯 建议:")  
    print("1. 如果稳定训练成功，使用 ./output/stable_model")  
    print("2. 如果训练失败，直接使用原始BlenderLLM模型")  
    print("3. 原始模型已经表现很好 (平均分8.4/10)")  
