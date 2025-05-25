#!/usr/bin/env python3  
"""  
使用自然语言描述进行椅子设计微调  
"""  

import os  
import sys  
import json  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import gc  

def load_chair_data_natural():  
    """加载椅子数据并转换为自然语言格式"""  
    print("📊 加载椅子数据（自然语言版本）...")  
    
    data_dir = "./data_grouped"  
    chair_data = []  
    
    for folder_name in os.listdir(data_dir):  
        folder_path = os.path.join(data_dir, folder_name)  
        if not os.path.isdir(folder_path):  
            continue  
        
        txt_file = os.path.join(folder_path, f"{folder_name}.txt")  
        tags_file = os.path.join(folder_path, "tags.txt")  
        
        if os.path.exists(txt_file):  
            try:  
                # 读取椅子描述  
                with open(txt_file, 'r', encoding='utf-8') as f:  
                    description = f.read().strip()  
                
                # 读取并解析标签  
                tags_dict = {}  
                if os.path.exists(tags_file):  
                    with open(tags_file, 'r', encoding='utf-8') as f:  
                        tags_content = f.read().strip()  
                        tags_dict = parse_tags_to_natural(tags_content)  
                
                # 构造自然语言格式  
                prompt = f"Design a chair: {description}"  
                
                # 生成自然语言回答  
                response = generate_natural_response(description, tags_dict)  
                
                chair_data.append((prompt, response))  
                print(f"  ✅ 加载椅子: {folder_name[:8]}...")  
                
            except Exception as e:  
                print(f"  ⚠️ 跳过 {folder_name}: {e}")  
                continue  
    
    print(f"📈 总共加载了 {len(chair_data)} 个椅子样本")  
    return chair_data  

def parse_tags_to_natural(tags_content):  
    """将标签解析为自然语言友好的字典"""  
    tags_dict = {}  
    
    lines = tags_content.strip().split('\n')  
    for line in lines:  
        if ':' in line:  
            key, value = line.split(':', 1)  
            key = key.strip()  
            value = value.strip()  
            
            if value.lower() != 'null':  
                tags_dict[key] = value  
    
    return tags_dict  

def generate_natural_response(description, tags_dict):  
    """生成自然语言的椅子设计回答"""  
    
    response_parts = []  
    
    # 开始设计描述  
    response_parts.append("I'll design this chair with the following features:")  
    
    # 基础结构  
    response_parts.append("\n**Structure:**")  
    response_parts.append("- Four legs for stability")  
    response_parts.append("- A horizontal seat surface")  
    response_parts.append("- A backrest for support")  
    
    # 根据标签添加特色  
    style = tags_dict.get('现代风格', '')  
    if '极简主义' in style:  
        response_parts.append("\n**Style - Minimalist:**")  
        response_parts.append("- Clean, simple lines")  
        response_parts.append("- Thin, elegant proportions")  
        response_parts.append("- Minimal decorative elements")  
        response_parts.append("- Focus on functionality")  
    
    material = tags_dict.get('材质相关描述', '')  
    if '实木' in material:  
        response_parts.append("\n**Material - Solid Wood:**")  
        response_parts.append("- Natural wood construction")  
        response_parts.append("- Visible wood grain texture")  
        response_parts.append("- Warm, natural color")  
        response_parts.append("- Durable and sturdy build")  
    
    # 功能特性  
    response_parts.append("\n**Functional Features:**")  
    
    ergonomic = tags_dict.get('人体工学符合性', '').lower()  
    if ergonomic and ergonomic != '无':  
        response_parts.append("- Ergonomically designed for comfort")  
        response_parts.append("- Proper back support angle")  
    
    adjustable_height = tags_dict.get('高度可调节性', '')  
    if adjustable_height and adjustable_height != '无':  
        response_parts.append("- Height adjustable mechanism")  
    
    adjustable_angle = tags_dict.get('角度可调节性', '')  
    if adjustable_angle and adjustable_angle != '无':  
        response_parts.append("- Adjustable backrest angle")  
    
    foldable = tags_dict.get('折叠性', '')  
    if foldable and foldable != '无':  
        response_parts.append("- Foldable design for storage")  
    
    # 尺寸建议  
    response_parts.append("\n**Dimensions:**")  
    response_parts.append("- Seat height: 45cm from ground")  
    response_parts.append("- Seat depth: 40cm")  
    response_parts.append("- Seat width: 45cm")  
    response_parts.append("- Backrest height: 35cm above seat")  
    
    # 基于描述的特殊考虑  
    desc_lower = description.lower()  
    if 'dining' in desc_lower:  
        response_parts.append("\n**Dining Chair Specifics:**")  
        response_parts.append("- Standard dining height")  
        response_parts.append("- Easy to move and stack")  
        response_parts.append("- Matches dining table aesthetics")  
    
    elif 'office' in desc_lower:  
        response_parts.append("\n**Office Chair Specifics:**")  
        response_parts.append("- Swivel base for mobility")  
        response_parts.append("- Suitable for desk work")  
        response_parts.append("- Professional appearance")  
    
    elif 'lounge' in desc_lower or 'relax' in desc_lower:  
        response_parts.append("\n**Lounge Chair Specifics:**")  
        response_parts.append("- Lower, more relaxed seating position")  
        response_parts.append("- Wider seat for comfort")  
        response_parts.append("- Emphasis on relaxation")  
    
    # 最终总结  
    response_parts.append("\n**Design Summary:**")  
    response_parts.append(f"This chair combines {material.lower() if material else 'quality materials'} with "  
                         f"{'minimalist aesthetics' if '极简主义' in style else 'functional design'} "  
                         f"to create a {'versatile' if not tags_dict else 'specialized'} seating solution.")  
    
    return " ".join(response_parts)  

def train_natural_language():  
    """使用自然语言进行训练"""  
    print("🗣️ 自然语言椅子设计微调")  
    print("=" * 50)  
    
    # 1. 加载数据  
    chair_data = load_chair_data_natural()  
    if len(chair_data) == 0:  
        print("❌ 没有加载到数据")  
        return  
    
    print(f"📊 训练样本数量: {len(chair_data)}")  
    
    # 显示样本  
    print("\n📝 样本预览:")  
    for i, (prompt, response) in enumerate(chair_data[:2]):  
        print(f"\n样本 {i+1}:")  
        print(f"提示: {prompt}")  
        print(f"回答: {response[:200]}...")  
    
    # 2. 检查GPU  
    if not torch.cuda.is_available():  
        print("❌ GPU不可用")  
        return  
    
    device = torch.device("cuda:0")  
    torch.cuda.empty_cache()  
    
    # 3. 加载模型  
    print("\n🔄 加载模型...")  
    try:  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": device},  
            low_cpu_mem_usage=True  
        )  
        print("✅ 模型加载成功")  
        
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  
        return  
    
    # 4. 设置可训练参数  
    print("🔒 设置可训练参数...")  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
        if 'lm_head' in name and 'weight' in name:  
            param.requires_grad = True  
            print(f"  解冻: {name}")  
    
    # 5. 优化器  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],  
        lr=1e-6,  # 更小的学习率  
        weight_decay=0.01  
    )  
    
    # 6. 训练  
    print("\n🏋️ 开始训练...")  
    model.train()  
    
    successful_steps = 0  
    total_loss = 0  
    
    # 只取前20个样本进行训练（避免过长）  
    train_samples = chair_data[:20]  
    
    for epoch in range(1):  # 只训练1个epoch  
        print(f"\n📅 Epoch {epoch + 1}/1")  
        
        for i, (prompt, target) in enumerate(train_samples):  
            try:  
                # 构造训练文本（更短的格式）  
                text = f"Human: {prompt}\n\nAssistant: {target}"  
                
                # 限制长度  
                if len(text) > 1000:  # 限制字符数  
                    text = text[:1000] + "..."  
                
                # Tokenize  
                inputs = tokenizer(  
                    text,  
                    return_tensors="pt",  
                    max_length=256,  # 较短的序列  
                    truncation=True,  
                    padding=True  
                ).to(device)  
                
                # 检查输入长度  
                if inputs['input_ids'].shape[1] > 256:  
                    print(f"    ⚠️ 步骤 {i+1}: 序列过长，跳过")  
                    continue  
                
                # 前向传播  
                outputs = model(**inputs, labels=inputs['input_ids'])  
                loss = outputs.loss  
                
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"    ⚠️ 步骤 {i+1}: 跳过无效损失")  
                    continue  
                
                print(f"    📈 步骤 {i+1}/{len(train_samples)}: 损失 {loss.item():.6f}")  
                
                # 反向传播  
                loss.backward()  
                
                # 梯度裁剪  
                torch.nn.utils.clip_grad_norm_(  
                    [p for p in model.parameters() if p.requires_grad],  
                    max_norm=0.5  
                )  
                
                # 更新  
                optimizer.step()  
                optimizer.zero_grad()  
                
                successful_steps += 1  
                total_loss += loss.item()  
                
                # 清理  
                del loss, outputs, inputs  
                torch.cuda.empty_cache()  
                
            except Exception as e:  
                print(f"    ❌ 步骤 {i+1} 失败: {e}")  
                optimizer.zero_grad()  
                torch.cuda.empty_cache()  
                continue  
    
    # 7. 保存  
    print(f"\n💾 保存模型...")  
    print(f"📊 成功训练步骤: {successful_steps}")  
    
    if successful_steps > 0:  
        print(f"📊 平均损失: {total_loss/successful_steps:.6f}")  
        
        try:  
            output_dir = "./output/natural_language_model"  
            os.makedirs(output_dir, exist_ok=True)  
            
            # 只保存可训练参数的state_dict  
            trainable_params = {}  
            for name, param in model.named_parameters():  
                if param.requires_grad:  
                    trainable_params[name] = param.detach().cpu().clone()  
            
            torch.save(trainable_params, os.path.join(output_dir, "trainable_params.pt"))  
            
            # 保存tokenizer  
            tokenizer.save_pretrained(output_dir)  
            
            # 保存训练信息  
            training_info = {  
                "total_samples": len(chair_data),  
                "trained_samples": len(train_samples),  
                "successful_steps": successful_steps,  
                "average_loss": total_loss/successful_steps,  
                "format": "natural_language"  
            }  
            
            with open(os.path.join(output_dir, "training_info.json"), 'w') as f:  
                json.dump(training_info, f, indent=2)  
            
            print(f"✅ 保存成功: {output_dir}")  
            
        except Exception as e:  
            print(f"❌ 保存失败: {e}")  
    else:  
        print("❌ 没有成功的训练步骤")  
    
    print("✅ 训练完成")  

if __name__ == "__main__":  
    train_natural_language()  
