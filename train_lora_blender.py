#!/usr/bin/env python3  
"""  
使用LoRA微调BlenderLLM提升椅子设计理解  
修复多GPU设备冲突问题  
"""  

import os  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig  
from peft import LoraConfig, get_peft_model, TaskType  
import json  
from torch.utils.data import Dataset  

class ChairDesignDataset(Dataset):  
    """椅子设计数据集"""  
    def __init__(self, data, tokenizer, max_length=512):  
        self.data = data  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        prompt, target = self.data[idx]  
        
        # 构造完整文本  
        text = f"User: {prompt}\n\nAssistant: {target}"  
        
        # Tokenize  
        encoding = self.tokenizer(  
            text,  
            truncation=True,  
            padding='max_length',  
            max_length=self.max_length,  
            return_tensors='pt'  
        )  
        
        return {  
            'input_ids': encoding['input_ids'].flatten(),  
            'attention_mask': encoding['attention_mask'].flatten(),  
            'labels': encoding['input_ids'].flatten()  
        }  

# 自定义Trainer类 - 放在这里！  
class FixedTrainer(Trainer):  
    """修复设备冲突的Trainer"""  
    def _prepare_inputs(self, inputs):  
        """确保所有输入都在正确的设备上"""  
        for key, value in inputs.items():  
            if isinstance(value, torch.Tensor):  
                inputs[key] = value.to(self.model.device)  
        return inputs  

def load_chair_data_simple():  
    """加载椅子数据，简化版本避免复杂性"""  
    print("📊 加载椅子数据...")  
    
    data_dir = "./data_grouped"  
    chair_data = []  
    
    for folder_name in os.listdir(data_dir)[:10]:  # 减少到10个样本  
        folder_path = os.path.join(data_dir, folder_name)  
        if not os.path.isdir(folder_path):  
            continue  
        
        txt_file = os.path.join(folder_path, f"{folder_name}.txt")  
        
        if os.path.exists(txt_file):  
            try:  
                # 读取描述  
                with open(txt_file, 'r', encoding='utf-8') as f:  
                    description = f.read().strip()  
                
                # 简化的提示  
                simplified_prompt = f"Design a chair: {description[:100]}"  
                
                # 简化的Blender代码模板  
                simple_code = generate_simple_blender_code(description)  
                
                chair_data.append((simplified_prompt, simple_code))  
                print(f"  ✅ 加载椅子: {folder_name[:8]}...")  
                
            except Exception as e:  
                print(f"  ⚠️ 跳过 {folder_name}: {e}")  
                continue  
    
    print(f"📈 总共加载了 {len(chair_data)} 个椅子样本")  
    return chair_data  

def generate_simple_blender_code(description):  
    """生成简化的Blender代码"""  
    
    # 检测关键特征  
    has_armrest = 'armrest' in description.lower()  
    is_office_chair = 'office' in description.lower()  
    is_minimalist = 'minimalist' in description.lower()  
    
    code = '''import bpy  

# Clear existing objects  
bpy.ops.object.select_all(action='DESELECT')  
bpy.ops.object.select_by_type(type='MESH')  
bpy.ops.object.delete()  

# Create seat  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0.5))  
seat = bpy.context.active_object  
seat.name = "Seat"  
'''  
    
    if is_minimalist:  
        code += 'seat.scale = (0.4, 0.35, 0.02)\n'  
    else:  
        code += 'seat.scale = (0.45, 0.4, 0.05)\n'  
    
    # 椅背  
    code += '''  
# Create backrest  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.35, 0.85))  
backrest = bpy.context.active_object  
backrest.name = "Backrest"  
backrest.scale = (0.4, 0.04, 0.35)  
'''  
    
    # 椅腿  
    if is_office_chair:  
        code += '''  
# Office chair base  
bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.05, location=(0, 0, 0.05))  
base = bpy.context.active_object  
base.name = "Base"  
'''  
    else:  
        code += '''  
# Four legs  
leg_positions = [(-0.35, -0.3, 0.25), (0.35, -0.3, 0.25), (-0.35, 0.3, 0.25), (0.35, 0.3, 0.25)]  
for i, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.5, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{i+1}"  
'''  
    
    return code  

def main():  
    """主训练函数 - LoRA版本，修复设备问题"""  
    print("🎯 LoRA微调BlenderLLM椅子设计理解")  
    print("=" * 50)  
    
    # 设置设备和环境 - 这些是Python代码，不是命令行！  
    print("🔧 设置训练环境...")  
    
    # 强制使用单GPU (这是Python代码中的设置)  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print(f"🎯 使用设备: {device}")  
    
    # 1. 加载数据  
    chair_data = load_chair_data_simple()  
    if len(chair_data) == 0:  
        print("❌ 没有加载到数据")  
        return  
    
    print(f"📊 训练样本数量: {len(chair_data)}")  
    
    # 显示样本  
    print("\n📝 样本预览:")  
    for i, (prompt, code) in enumerate(chair_data[:2]):  
        print(f"\n样本 {i+1}:")  
        print(f"提示: {prompt}")  
        print(f"代码: {code[:100]}...")  
    
    # 2. 配置量化  
    print("\n🔧 配置量化...")  
    quantization_config = BitsAndBytesConfig(  
        load_in_8bit=True,  
        llm_int8_threshold=6.0,  
        llm_int8_has_fp16_weight=False,  
    )  
    
    # 3. 加载模型和tokenizer  
    print("\n🔄 加载BlenderLLM...")  
    try:  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        # 使用单GPU加载 (device_map设置所有层都在GPU 0上)  
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            quantization_config=quantization_config,  
            torch_dtype=torch.float16,  
            device_map={"": 0},  
        )  
        print("✅ 模型加载成功")  
        
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  
        return  
    
    # 4. 配置LoRA  
    print("\n🔧 配置LoRA...")  
    lora_config = LoraConfig(  
        task_type=TaskType.CAUSAL_LM,  
        r=4,  # 进一步减小rank  
        lora_alpha=8,  # 进一步减小alpha  
        lora_dropout=0.1,  
        target_modules=["q_proj", "v_proj"],  
        bias="none"  
    )  
    
    # 应用LoRA  
    model = get_peft_model(model, lora_config)  
    model.print_trainable_parameters()  
    
    # 5. 准备数据集  
    print("\n📊 准备数据集...")  
    dataset = ChairDesignDataset(chair_data, tokenizer, max_length=128)  # 进一步减小  
    
    # 6. 配置训练参数  
    print("\n⚙️ 配置训练参数...")  
    training_args = TrainingArguments(  
        output_dir="./output/lora_blender_checkpoints",  
        num_train_epochs=1,  
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=1,  # 减少梯度累积  
        warmup_steps=2,  
        learning_rate=5e-5,  # 进一步降低学习率  
        fp16=True,  
        logging_steps=1,  
        save_steps=50,  
        save_total_limit=1,  
        remove_unused_columns=False,  
        dataloader_pin_memory=False,  
        gradient_checkpointing=False,  
        dataloader_num_workers=0,  
        report_to=None,  
    )  
    
    # 7. 创建Trainer (使用上面定义的FixedTrainer)  
    print("\n🏋️ 创建Trainer...")  
    trainer = FixedTrainer(  
        model=model,  
        args=training_args,  
        train_dataset=dataset,  
        tokenizer=tokenizer,  # 先用这个，看是否还报错  
    )  
    
    # 8. 开始训练  
    print("\n🚀 开始LoRA微调...")  
    try:  
        trainer.train()  
        print("✅ 训练完成!")  
        
        # 9. 保存LoRA适配器  
        print("\n💾 保存LoRA模型...")  
        output_dir = "./output/lora_blender_enhanced"  
        os.makedirs(output_dir, exist_ok=True)  
        
        # 保存LoRA适配器  
        model.save_pretrained(output_dir)  
        tokenizer.save_pretrained(output_dir)  
        
        # 保存训练信息  
        training_info = {  
            "model_type": "BlenderLLM_LoRA",  
            "base_model": "../models/BlenderLLM",  
            "lora_config": {  
                "r": lora_config.r,  
                "lora_alpha": lora_config.lora_alpha,  
                "target_modules": lora_config.target_modules,  
                "lora_dropout": lora_config.lora_dropout  
            },  
            "training_samples": len(chair_data),  
            "epochs": 1,  
            "device": str(device),  
            "enhanced_features": [  
                "Chair design understanding",  
                "Style and feature recognition",   
                "Improved Blender code generation"  
            ]  
        }  
        
        with open(os.path.join(output_dir, "training_info.json"), 'w') as f:  
            json.dump(training_info, f, indent=2)  
        
        print(f"✅ LoRA模型保存成功: {output_dir}")  
        
        # 10. 简单测试  
        print("\n🧪 测试LoRA模型...")  
        model.eval()  
        
        test_prompt = "Design a chair: modern minimalist chair"  
        inputs = tokenizer(f"User: {test_prompt}\n\nAssistant:", return_tensors="pt")  
        # 确保输入在正确设备上  
        inputs = {k: v.to(device) for k, v in inputs.items()}  
        
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,  
                max_new_tokens=50,  
                temperature=0.8,  
                do_sample=True,  
                pad_token_id=tokenizer.pad_token_id  
            )  
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
        assistant_response = response.split("Assistant:")[-1].strip()  
        
        print(f"🎯 测试提示: {test_prompt}")  
        print(f"🔧 生成代码: {assistant_response[:150]}...")  
        
    except Exception as e:  
        print(f"❌ 训练失败: {e}")  
        import traceback  
        traceback.print_exc()  
    
    print("\n✅ LoRA微调完成！")  

if __name__ == "__main__":  
    main()  
