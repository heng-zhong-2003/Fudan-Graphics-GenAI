#!/usr/bin/env python3  
"""  
内存优化的椅子设计微调 - 结合最小化训练策略  
"""  

import torch  
import json  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  
import gc  

# 强制单GPU和内存优化  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  

def clear_gpu_memory():  
    """彻底清理GPU内存"""  
    gc.collect()  
    torch.cuda.empty_cache()  
    torch.cuda.synchronize()  

def memory_optimized_train():  
    print("🪑 内存优化椅子设计微调...")  
    
    # 彻底清理内存  
    clear_gpu_memory()  
    
    # 检查GPU状态  
    print(f"🔍 GPU内存状态:")  
    print(f"  总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")  
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")  
    print(f"  缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")  
    
    # 加载tokenizer  
    print("🔄 加载tokenizer...")  
    tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
    
    # 使用最保守的方式加载模型  
    print("🔄 加载模型（内存优化模式）...")  
    try:  
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  # 使用半精度  
            device_map="auto",          # 自动分配设备  
            low_cpu_mem_usage=True,     # 低CPU内存使用  
            offload_folder="./temp_offload",  # 临时卸载文件夹  
            max_memory={0: "20GB", "cpu": "30GB"}  # 限制GPU内存使用  
        )  
        print("✅ 模型加载成功")  
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  
        print("🔄 尝试更激进的内存优化...")  
        
        # 更激进的内存优化  
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map="auto",  
            low_cpu_mem_usage=True,  
            offload_folder="./temp_offload",  
            max_memory={0: "15GB", "cpu": "50GB"}  # 更保守的GPU内存限制  
        )  
    
    # 检查模型参数分布  
    gpu_params = 0  
    cpu_params = 0  
    for name, param in model.named_parameters():  
        if param.device.type == 'cuda':  
            gpu_params += param.numel()  
        else:  
            cpu_params += param.numel()  
    
    print(f"📊 参数分布:")  
    print(f"  GPU参数: {gpu_params:,}")  
    print(f"  CPU参数: {cpu_params:,}")  
    
    # 冻结大部分参数，只训练最后几层  
    print("🔒 冻结大部分参数...")  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
    
    # 只解冻最后2层和输出层  
    layers_to_train = []  
    for name, param in model.named_parameters():  
        # 根据实际模型结构调整层名  
        if any(layer in name for layer in ['layers.31', 'layers.30', 'lm_head', 'embed_out']):  
            param.requires_grad = True  
            layers_to_train.append(name)  
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    total_params = sum(p.numel() for p in model.parameters())  
    
    print(f"🎯 训练参数:")  
    print(f"  可训练: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")  
    print(f"  训练层: {len(layers_to_train)}")  
    
    # 启用梯度检查点以节省内存  
    if hasattr(model, 'gradient_checkpointing_enable'):  
        model.gradient_checkpointing_enable()  
        print("✅ 启用梯度检查点")  
    
    # 加载训练数据  
    print("📊 加载训练数据...")  
    try:  
        with open('./output/new_training_data/chair_training_data.json', 'r', encoding='utf-8') as f:  
            data = json.load(f)  
    except FileNotFoundError:  
        print("❌ 训练数据文件不存在，创建示例数据...")  
        data = create_sample_data()  
    
    # 只使用少量数据进行训练  
    data = data[:20]  # 进一步减少数据量  
    print(f"📈 使用 {len(data)} 个训练样本")  
    
    # 设置优化器  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],   
        lr=5e-6,  # 更小的学习率  
        weight_decay=0.01  
    )  
    
    model.train()  
    
    print("🏋️ 开始训练...")  
    
    for epoch in range(1):  # 只训练1个epoch  
        total_loss = 0  
        valid_steps = 0  
        
        for i, item in enumerate(data):  
            if i % 5 == 0:  # 更频繁的内存清理  
                clear_gpu_memory()  
                print(f"  步骤 {i}/{len(data)} - GPU内存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")  
            
            try:  
                # 构建训练文本  
                text = f"Human: {item.get('input', '')}\n\nAssistant: {item.get('output', '')}"  
                
                # 限制文本长度以节省内存  
                if len(text) > 400:  
                    text = text[:400]  
                
                # Tokenize  
                inputs = tokenizer(  
                    text,   
                    return_tensors="pt",   
                    max_length=256,  # 更短的序列长度  
                    truncation=True,   
                    padding=True  
                )  
                
                # 将输入移动到正确的设备  
                inputs = {k: v.to(model.device) if hasattr(model, 'device') else v.cuda()   
                         for k, v in inputs.items()}  
                
                # 前向传播  
                outputs = model(**inputs, labels=inputs['input_ids'])  
                loss = outputs.loss  
                
                # 检查损失是否有效  
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"  ⚠️ 跳过无效损失: {loss.item()}")  
                    continue  
                
                # 反向传播  
                loss.backward()  
                
                # 梯度累积（每4步更新一次）  
                if (i + 1) % 4 == 0:  
                    # 梯度裁剪  
                    torch.nn.utils.clip_grad_norm_(  
                        [p for p in model.parameters() if p.requires_grad],   
                        max_norm=1.0  
                    )  
                    
                    optimizer.step()  
                    optimizer.zero_grad()  
                
                total_loss += loss.item()  
                valid_steps += 1  
                
            except Exception as e:  
                print(f"  ❌ 步骤 {i} 失败: {e}")  
                clear_gpu_memory()  
                continue  
        
        avg_loss = total_loss / valid_steps if valid_steps > 0 else float('inf')  
        print(f"📈 Epoch {epoch+1} 平均损失: {avg_loss:.4f} (有效步骤: {valid_steps}/{len(data)})")  
    
    # 保存模型  
    print("💾 保存模型...")  
    output_dir = "./output/memory_optimized_model"  
    os.makedirs(output_dir, exist_ok=True)  
    
    try:  
        model.save_pretrained(output_dir)  
        tokenizer.save_pretrained(output_dir)  
        print(f"✅ 模型保存到: {output_dir}")  
    except Exception as e:  
        print(f"❌ 保存失败: {e}")  
    
    # 测试微调后的模型  
    print("\n🧪 测试微调后的模型...")  
    test_model(model, tokenizer)  
    
    # 清理临时文件  
    if os.path.exists("./temp_offload"):  
        import shutil  
        shutil.rmtree("./temp_offload")  
        print("🗑️ 清理临时文件")  

def create_sample_data():  
    """创建示例训练数据"""  
    return [  
        {  
            "input": "Generate chair design: modern office chair",  
            "output": "import bpy\n\n# Create modern office chair\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.5))\nchair_seat = bpy.context.active_object\nchair_seat.scale = (1, 1, 0.1)\n\n# Add backrest\nbpy.ops.mesh.primitive_cube_add(location=(0, -0.9, 1.2))\nbackrest = bpy.context.active_object\nbackrest.scale = (1, 0.1, 0.7)"  
        },  
        {  
            "input": "Generate chair design: wooden dining chair",  
            "output": "import bpy\n\n# Create wooden dining chair\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.4))\nseat = bpy.context.active_object\nseat.scale = (0.8, 0.8, 0.05)\n\n# Add wooden legs\nfor i, pos in enumerate([(-0.7, -0.7, 0.2), (0.7, -0.7, 0.2), (-0.7, 0.7, 0.2), (0.7, 0.7, 0.2)]):\n    bpy.ops.mesh.primitive_cylinder_add(location=pos)\n    leg = bpy.context.active_object\n    leg.scale = (0.05, 0.05, 0.4)"  
        }  
    ]  

def test_model(model, tokenizer):  
    """测试微调后的模型"""  
    model.eval()  
    
    test_prompts = [  
        "Generate chair design: simple wooden chair",  
        "Generate chair design: ergonomic office chair"  
    ]  
    
    for prompt in test_prompts:  
        try:  
            inputs = tokenizer(prompt, return_tensors="pt")  
            inputs = {k: v.to(model.device) if hasattr(model, 'device') else v.cuda()   
                     for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 200,  
                    temperature=0.7,  
                    do_sample=True,  
                    pad_token_id=tokenizer.eos_token_id,  
                    repetition_penalty=1.1  
                )  
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(prompt):].strip()  
            
            print(f"📝 提示: {prompt}")  
            print(f"🤖 生成: {generated[:200]}...")  
            print("-" * 50)  
            
        except Exception as e:  
            print(f"❌ 测试失败: {e}")  

if __name__ == "__main__":  
    memory_optimized_train()  
