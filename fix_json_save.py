import json  
import os  

# 修复后的保存函数  
def save_training_info_fixed(lora_config, chair_data, device, output_dir):  
    training_info = {  
        "model_type": "BlenderLLM_LoRA",  
        "base_model": "../models/BlenderLLM",  
        "lora_config": {  
            "r": lora_config.r,  
            "lora_alpha": lora_config.lora_alpha,  
            "target_modules": list(lora_config.target_modules),  # 转换为list  
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
    
    print(f"✅ 训练信息保存成功")  

# 手动保存训练信息  
output_dir = "./output/lora_blender_enhanced"  
if os.path.exists(output_dir):  
    # 创建一个简单的训练信息  
    training_info = {  
        "model_type": "BlenderLLM_LoRA",  
        "base_model": "../models/BlenderLLM",  
        "lora_config": {  
            "r": 4,  
            "lora_alpha": 8,  
            "target_modules": ["q_proj", "v_proj"],  # 直接用list  
            "lora_dropout": 0.1  
        },  
        "training_samples": 10,  
        "epochs": 1,  
        "training_status": "SUCCESS",  
        "final_loss": 1.5535,  
        "enhanced_features": [  
            "Chair design understanding",  
            "Style and feature recognition",   
            "Improved Blender code generation"  
        ]  
    }  
    
    with open(os.path.join(output_dir, "training_info.json"), 'w') as f:  
        json.dump(training_info, f, indent=2)  
    
    print(f"✅ 训练信息已修复保存到: {output_dir}/training_info.json")  
else:  
    print("❌ 输出目录不存在")  
