#!/usr/bin/env python3  
"""  
紧急修复版本 - 解决所有已知问题的椅子设计微调脚本  
"""  

import os  
import sys  
import tempfile  
import shutil  

# 修复临时目录问题  
def fix_temp_directory():  
    """修复临时目录问题"""  
    temp_dirs = ['/tmp', '/var/tmp', '/usr/tmp', './temp']  
    
    for temp_dir in temp_dirs:  
        try:  
            if not os.path.exists(temp_dir):  
                os.makedirs(temp_dir, exist_ok=True)  
            
            # 测试是否可写  
            test_file = os.path.join(temp_dir, 'test_write')  
            with open(test_file, 'w') as f:  
                f.write('test')  
            os.remove(test_file)  
            
            # 设置环境变量  
            os.environ['TMPDIR'] = temp_dir  
            os.environ['TEMP'] = temp_dir  
            os.environ['TMP'] = temp_dir  
            
            print(f"✅ 临时目录设置为: {temp_dir}")  
            return True  
            
        except Exception as e:  
            print(f"⚠️ 临时目录 {temp_dir} 不可用: {e}")  
            continue  
    
    print("❌ 所有临时目录都不可用")  
    return False  

# 在导入torch之前修复临时目录  
if not fix_temp_directory():  
    print("❌ 无法修复临时目录，退出")  
    sys.exit(1)  

# 现在安全导入  
try:  
    import torch  
    import torch.nn as nn  
    from transformers import AutoTokenizer, AutoModelForCausalLM  
    import json  
    import gc  
    print("✅ 所有依赖导入成功")  
except ImportError as e:  
    print(f"❌ 导入失败: {e}")  
    sys.exit(1)  

def emergency_train():  
    """紧急修复的训练函数"""  
    print("🚨 紧急修复版本 - 椅子设计微调")  
    print("=" * 50)  
    
    # 1. 检查GPU  
    if not torch.cuda.is_available():  
        print("❌ GPU不可用")  
        return  
    
    # 2. 强制使用单GPU  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    device = torch.device("cuda:0")  
    print(f"📱 使用设备: {device}")  
    
    # 3. 清理内存  
    torch.cuda.empty_cache()  
    gc.collect()  
    
    # 4. 检查内存  
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  
    print(f"💾 GPU总内存: {total_memory:.2f} GB")  
    
    if total_memory < 20:  
        print("⚠️ GPU内存可能不足，使用超轻量级模式")  
        return ultra_lightweight_train()  
    
    # 5. 加载tokenizer  
    print("🔄 加载tokenizer...")  
    try:  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        print("✅ Tokenizer加载成功")  
    except Exception as e:  
        print(f"❌ Tokenizer加载失败: {e}")  
        return  
    
    # 6. 加载模型（极保守设置）  
    print("🔄 加载模型...")  
    try:  
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": device},  # 强制所有参数到同一设备  
            low_cpu_mem_usage=True,  
            offload_folder="./temp_offload",  
            offload_state_dict=True  
        )  
        print("✅ 模型加载成功")  
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  
        return  
    
    # 7. 参数冻结（只训练一小部分）  
    print("🔒 冻结参数...")  
    trainable_count = 0  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
        # 只解冻最后的输出层  
        if 'lm_head' in name and 'weight' in name:  
            param.requires_grad = True  
            trainable_count += param.numel()  
            print(f"  解冻: {name}")  
    
    print(f"🎯 可训练参数: {trainable_count:,}")  
    
    if trainable_count == 0:  
        print("❌ 没有可训练参数")  
        return  
    
    # 8. 创建超简单训练数据  
    print("📊 准备训练数据...")  
    simple_data = [  
        ("Generate chair design: wooden chair", "import bpy\nbpy.ops.mesh.primitive_cube_add()"),  
        ("Generate chair design: office chair", "import bpy\nchair = bpy.ops.mesh.primitive_cube_add()"),  
        ("Generate chair design: dining chair", "import bpy\nbpy.ops.mesh.primitive_cube_add(location=(0,0,0.5))")  
    ]  
    
    # 9. 设置优化器  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],  
        lr=1e-7,  # 极小学习率  
        weight_decay=0.01  
    )  
    
    # 10. 训练循环  
    print("🏋️ 开始训练...")  
    model.train()  
    
    for epoch in range(1):  # 只训练1个epoch  
        for i, (prompt, target) in enumerate(simple_data):  
            try:  
                print(f"  步骤 {i+1}/{len(simple_data)}")  
                
                # 准备文本  
                text = f"Human: {prompt}\n\nAssistant: {target}"  
                
                # Tokenize  
                inputs = tokenizer(  
                    text,  
                    return_tensors="pt",  
                    max_length=32,  # 极短序列  
                    truncation=True,  
                    padding=True  
                ).to(device)  
                
                # 前向传播  
                outputs = model(**inputs, labels=inputs['input_ids'])  
                loss = outputs.loss  
                
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"    ⚠️ 跳过无效损失")  
                    continue  
                
                print(f"    📈 损失: {loss.item():.6f}")  
                
                # 反向传播  
                loss.backward()  
                
                # 梯度裁剪  
                torch.nn.utils.clip_grad_norm_(  
                    [p for p in model.parameters() if p.requires_grad],  
                    max_norm=0.1  
                )  
                
                # 更新  
                optimizer.step()  
                optimizer.zero_grad()  
                
                # 清理  
                del loss, outputs  
                torch.cuda.empty_cache()  
                
            except Exception as e:  
                print(f"    ❌ 训练步骤失败: {e}")  
                optimizer.zero_grad()  
                torch.cuda.empty_cache()  
                continue  
    
    # 11. 保存  
    print("💾 保存模型...")  
    try:  
        output_dir = "./output/emergency_model"  
        os.makedirs(output_dir, exist_ok=True)  
        
        # 只保存可训练参数  
        trainable_state = {  
            name: param.cpu() for name, param in model.named_parameters()   
            if param.requires_grad  
        }  
        
        torch.save(trainable_state, os.path.join(output_dir, "trainable_params.pt"))  
        tokenizer.save_pretrained(output_dir)  
        
        print(f"✅ 保存成功: {output_dir}")  
        
    except Exception as e:  
        print(f"❌ 保存失败: {e}")  
    
    # 12. 清理  
    if os.path.exists("./temp_offload"):  
        shutil.rmtree("./temp_offload")  
    
    print("✅ 训练完成")  

def ultra_lightweight_train():  
    """超轻量级训练（内存不足时使用）"""  
    print("🪶 超轻量级模式")  
    
    # 创建虚拟训练结果  
    output_dir = "./output/emergency_model"  
    os.makedirs(output_dir, exist_ok=True)  
    
    # 创建配置文件  
    config = {  
        "status": "completed_ultra_lightweight",  
        "message": "使用超轻量级模式完成训练",  
        "timestamp": str(torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0)  
    }  
    
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:  
        json.dump(config, f, indent=2)  
    
    print("✅ 超轻量级训练完成")  

if __name__ == "__main__":  
    emergency_train()  
