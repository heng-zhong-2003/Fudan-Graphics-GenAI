#!/usr/bin/env python3  
"""  
使用新生成的数据进行训练  
"""  

import torch  
import json  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  

def train_with_new_data():  
    print("🪑 使用新数据训练椅子设计模型...")  
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    torch.cuda.empty_cache()  
    
    # 加载新数据  
    data_file = "./output/new_training_data/chair_training_data.json"  
    if not os.path.exists(data_file):  
        print(f"❌ 数据文件不存在: {data_file}")  
        print("请先运行 regenerate_training_data.py")  
        return  
    
    with open(data_file, 'r', encoding='utf-8') as f:  
        training_data = json.load(f)  
    
    print(f"📊 训练数据量: {len(training_data)}")  
    
    # 验证数据质量  
    valid_data = []  
    for item in training_data:  
        input_text = item.get('input', '').strip()  
        output_text = item.get('output', '').strip()  
        
        if input_text and output_text and len(output_text) > 50:  
            valid_data.append(item)  
    
    print(f"✅ 有效数据量: {len(valid_data)}")  
    
    # 限制训练数据量以加快训练  
    if len(valid_data) > 100:  
        valid_data = valid_data[:100]  
        print(f"🎯 使用前100个样本进行训练")  
    
    # 显示样本  
    print("\n📝 数据样本:")  
    for i in range(min(2, len(valid_data))):  
        sample = valid_data[i]  
        print(f"\n样本 {i+1}:")  
        print(f"  输入: {sample['input']}")  
        print(f"  输出: {sample['output'][:200]}...")  
    
    # 加载模型  
    print("\n🔄 加载模型...")  
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
    
    # 只训练部分参数以加快训练  
    for param in model.parameters():  
        param.requires_grad = False  
    
    # 解冻最后几层  
    if hasattr(model, 'lm_head'):  
        for param in model.lm_head.parameters():  
            param.requires_grad = True  
    
    # 如果有transformer层，解冻最后1-2层  
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  
        for layer in model.transformer.h[-2:]:  # 最后2层  
            for param in layer.parameters():  
                param.requires_grad = True  
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  
        for layer in model.model.layers[-2:]:  # 最后2层  
            for param in layer.parameters():  
                param.requires_grad = True  
    
    # 统计可训练参数  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    total_params = sum(p.numel() for p in model.parameters())  
    print(f"🎯 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")  
    
    # 优化器  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],   
        lr=1e-5,  
        weight_decay=0.01,  
        eps=1e-8  
    )  
    
    # 学习率调度器  
    from transformers import get_linear_schedule_with_warmup  
    total_steps = len(valid_data)  
    scheduler = get_linear_schedule_with_warmup(  
        optimizer,  
        num_warmup_steps=max(1, total_steps // 10),  
        num_training_steps=total_steps  
    )  
    
    model.train()  
    
    print(f"\n🏋️ 开始训练 {total_steps} 步...")  
    
    total_loss = 0  
    valid_steps = 0  
    log_interval = max(1, len(valid_data) // 10)  
    
    for i, item in enumerate(valid_data):  
        if i % log_interval == 0:  
            print(f"步骤 {i}/{len(valid_data)} ({100*i/len(valid_data):.1f}%)")  
            torch.cuda.empty_cache()  
        
        # 构建训练文本 - 使用更清晰的格式  
        conversation = f"Human: {item['input']}\n\nAssistant: {item['output']}"  
        
        # Tokenize  
        try:  
            inputs = tokenizer(  
                conversation,  
                return_tensors="pt",  
                max_length=1024,  # 增加最大长度  
                truncation=True,  
                padding=True  
            )  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            # 前向传播  
            outputs = model(**inputs, labels=inputs['input_ids'])  
            loss = outputs.loss  
            
            # 检查损失有效性  
            if torch.isnan(loss) or torch.isinf(loss):  
                print(f"⚠️  步骤 {i} 损失无效: {loss}")  
                optimizer.zero_grad()  
                continue  
            
            # 反向传播  
            loss.backward()  
            
            # 梯度裁剪  
            torch.nn.utils.clip_grad_norm_(  
                [p for p in model.parameters() if p.requires_grad],  
                max_norm=1.0  
            )  
            
            # 更新参数  
            optimizer.step()  
            scheduler.step()  
            optimizer.zero_grad()  
            
            total_loss += loss.item()  
            valid_steps += 1  
            
            # 定期显示损失  
            if i % log_interval == 0:  
                current_lr = scheduler.get_last_lr()[0]  
                print(f"  当前损失: {loss.item():.4f}, 学习率: {current_lr:.2e}")  
                
        except Exception as e:  
            print(f"⚠️  步骤 {i} 出错: {e}")  
            optimizer.zero_grad()  
            continue  
    
    avg_loss = total_loss / max(valid_steps, 1)  
    print(f"\n✅ 训练完成!")  
    print(f"📊 平均损失: {avg_loss:.4f}")  
    print(f"📊 有效步骤: {valid_steps}/{len(valid_data)}")  
    
    # 保存模型  
    output_dir = "./output/new_chair_model"  
    print(f"\n💾 保存模型到: {output_dir}")  
    os.makedirs(output_dir, exist_ok=True)  
    
    model.save_pretrained(output_dir)  
    tokenizer.save_pretrained(output_dir)  
    
    # 保存训练信息  
    training_info = {  
        "training_samples": len(valid_data),  
        "valid_steps": valid_steps,  
        "average_loss": avg_loss,  
        "trainable_params": trainable_params,  
        "total_params": total_params  
    }  
    
    with open(os.path.join(output_dir, "training_info.json"), 'w') as f:  
        json.dump(training_info, f, indent=2)  
    
    # 立即测试模型  
    print(f"\n🧪 测试训练后的模型...")  
    model.eval()  
    
    test_cases = [  
        "Generate chair design: modern office chair",  
        "Generate chair design: vintage wooden dining chair",   
        "Generate chair design: comfortable armchair",  
        "Generate chair design: sleek bar stool",  
        "Generate chair design: ergonomic gaming chair"  
    ]  
    
    for j, test_input in enumerate(test_cases):  
        print(f"\n--- 测试 {j+1} ---")  
        prompt = f"Human: {test_input}\n\nAssistant:"  
        
        try:  
            inputs = tokenizer(prompt, return_tensors="pt")  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 300,  
                    temperature=0.7,  
                    do_sample=True,  
                    top_p=0.9,  
                    pad_token_id=tokenizer.pad_token_id,  
                    eos_token_id=tokenizer.eos_token_id,  
                    repetition_penalty=1.1  
                )  
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(prompt):].strip()  
            
            print(f"📝 输入: {test_input}")  
            print(f"🤖 生成 ({len(generated)} 字符):")  
            print(f"   {generated[:400]}...")  
            
            # 检查是否包含Blender相关内容  
            blender_keywords = ['bpy.', 'import bpy', 'mesh.', 'object.', 'location', 'scale']  
            if any(keyword in generated for keyword in blender_keywords):  
                print("   ✅ 包含Blender代码")  
            else:  
                print("   ⚠️  可能不包含Blender代码")  
                
        except Exception as e:  
            print(f"❌ 生成失败: {e}")  
    
    print(f"\n🎉 训练完成! 模型保存在: {output_dir}")  
    
    return output_dir  

if __name__ == "__main__":  
    train_with_new_data()  
