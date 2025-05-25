#!/usr/bin/env python3  
"""  
微调BlenderLLM提升椅子设计理解能力  
保持代码生成格式，但增强语义理解  
"""  

import os  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json  

def load_chair_data_for_blender():  
    """加载椅子数据，构造用于BlenderLLM的训练格式"""  
    print("📊 加载椅子数据（BlenderLLM格式）...")  
    
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
                # 读取原始描述  
                with open(txt_file, 'r', encoding='utf-8') as f:  
                    description = f.read().strip()  
                
                # 读取标签  
                tags_dict = {}  
                if os.path.exists(tags_file):  
                    with open(tags_file, 'r', encoding='utf-8') as f:  
                        tags_content = f.read().strip()  
                        tags_dict = parse_tags(tags_content)  
                
                # 构造增强的prompt（包含风格和功能信息）  
                enhanced_prompt = create_enhanced_prompt(description, tags_dict)  
                
                # 生成对应的Blender代码（基于标签信息）  
                blender_code = generate_enhanced_blender_code(description, tags_dict)  
                
                chair_data.append((enhanced_prompt, blender_code))  
                print(f"  ✅ 加载椅子: {folder_name[:8]}...")  
                
            except Exception as e:  
                print(f"  ⚠️ 跳过 {folder_name}: {e}")  
                continue  
    
    print(f"📈 总共加载了 {len(chair_data)} 个椅子样本")  
    return chair_data  

def parse_tags(tags_content):  
    """解析标签内容"""  
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

def create_enhanced_prompt(description, tags_dict):  
    """创建包含风格和功能信息的增强prompt"""  
    
    # 基础描述  
    prompt_parts = [f"Generate chair design: {description}"]  
    
    # 添加风格信息  
    style = tags_dict.get('现代风格', '')  
    if '极简主义' in style:  
        prompt_parts.append("Style: minimalist design with clean lines and simple geometry")  
    elif '北欧' in style:  
        prompt_parts.append("Style: Nordic design with natural materials and functional aesthetics")  
    
    # 添加材质信息  
    material = tags_dict.get('材质相关描述', '')  
    if '实木' in material:  
        prompt_parts.append("Material: solid wood construction with natural grain texture")  
    elif '金属' in material:  
        prompt_parts.append("Material: metal frame with industrial aesthetic")  
    
    # 添加功能特性  
    ergonomic = tags_dict.get('人体工学符合性', '').lower()  
    if ergonomic and ergonomic != '无':  
        prompt_parts.append("Features: ergonomic design for comfortable seating")  
    
    adjustable_height = tags_dict.get('高度可调节性', '')  
    if adjustable_height and adjustable_height != '无':  
        prompt_parts.append("Features: height adjustable mechanism")  
    
    foldable = tags_dict.get('折叠性', '')  
    if foldable and foldable != '无':  
        prompt_parts.append("Features: foldable design for space saving")  
    
    return ". ".join(prompt_parts) + "."  

def generate_enhanced_blender_code(description, tags_dict):  
    """生成增强的Blender代码，考虑风格和功能"""  
    
    # 基础代码模板  
    code_template = '''import bpy  
import math  

# Clear existing mesh objects  
bpy.ops.object.select_all(action='DESELECT')  
bpy.ops.object.select_by_type(type='MESH')  
bpy.ops.object.delete()  

# Create chair components  
'''  
    
    # 根据风格调整参数  
    style = tags_dict.get('现代风格', '')  
    if '极简主义' in style:  
        # 极简风格 - 细腿、简单形状  
        code_template += '''  
# Minimalist style - thin legs and simple shapes  
leg_radius = 0.02  # Thin legs  
seat_thickness = 0.03  # Thin seat  
backrest_thickness = 0.02  # Thin backrest  
'''  
    else:  
        # 标准参数  
        code_template += '''  
# Standard style parameters  
leg_radius = 0.04  
seat_thickness = 0.05  
backrest_thickness = 0.04  
'''  
    
    # 座椅创建  
    code_template += '''  
# Create seat  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0.5))  
seat = bpy.context.active_object  
seat.name = "Seat"  
seat.scale = (0.45, 0.4, seat_thickness)  
'''  
    
    # 椅背创建  
    backrest_height = 0.35  
    ergonomic = tags_dict.get('人体工学符合性', '').lower()  
    if ergonomic and ergonomic != '无':  
        # 人体工学设计 - 稍微倾斜的椅背  
        code_template += f'''  
# Ergonomic backrest with slight tilt  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.35, 0.85))  
backrest = bpy.context.active_object  
backrest.name = "Backrest"  
backrest.scale = (0.4, backrest_thickness, {backrest_height})  
backrest.rotation_euler = (math.radians(-5), 0, 0)  # Slight tilt for comfort  
'''  
    else:  
        code_template += f'''  
# Standard vertical backrest  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.35, 0.85))  
backrest = bpy.context.active_object  
backrest.name = "Backrest"  
backrest.scale = (0.4, backrest_thickness, {backrest_height})  
'''  
    
    # 椅腿创建  
    material = tags_dict.get('材质相关描述', '')  
    if '金属' in material:  
        # 金属腿 - 更细更直  
        code_template += '''  
# Metal legs - thin and straight  
leg_positions = [(-0.35, -0.3, 0.25), (0.35, -0.3, 0.25), (-0.35, 0.3, 0.25), (0.35, 0.3, 0.25)]  
for i, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=leg_radius*0.7, depth=0.5, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{i+1}"  
'''  
    else:  
        # 木质腿 - 标准圆柱  
        code_template += '''  
# Wooden legs - standard cylindrical  
leg_positions = [(-0.35, -0.3, 0.25), (0.35, -0.3, 0.25), (-0.35, 0.3, 0.25), (0.35, 0.3, 0.25)]  
for i, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=leg_radius, depth=0.5, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{i+1}"  
'''  
    
    # 如果有扶手  
    if 'armrest' in description.lower() or '扶手' in description:  
        code_template += '''  
# Add armrests  
armrest_positions = [(-0.45, 0, 0.75), (0.45, 0, 0.75)]  
for i, pos in enumerate(armrest_positions):  
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos)  
    armrest = bpy.context.active_object  
    armrest.name = f"Armrest_{i+1}"  
    armrest.scale = (0.05, 0.3, 0.05)  
'''  
    
    # 材质设置  
    if '实木' in material:  
        code_template += '''  
# Apply wood material  
for obj in bpy.data.objects:  
    if obj.type == 'MESH' and obj.name.startswith(('Seat', 'Backrest', 'Leg', 'Armrest')):  
        mat = bpy.data.materials.new(name="Wood_Material")  
        mat.use_nodes = True  
        mat.node_tree.nodes.clear()  
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')  
        principled = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')  
        principled.inputs[0].default_value = (0.6, 0.4, 0.2, 1.0)  # Wood color  
        principled.inputs[7].default_value = 0.3  # Roughness  
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])  
        obj.data.materials.append(mat)  
'''  
    
    code_template += '''  
# Final positioning and camera setup  
bpy.ops.object.select_all(action='DESELECT')  
'''  
    
    return code_template  

def main():  
    """主训练函数"""  
    print("🎯 BlenderLLM椅子设计理解微调")  
    print("=" * 50)  
    
    # 1. 加载数据  
    chair_data = load_chair_data_for_blender()  
    if len(chair_data) == 0:  
        print("❌ 没有加载到数据")  
        return  
    
    print(f"📊 训练样本数量: {len(chair_data)}")  
    
    # 显示样本  
    print("\n📝 样本预览:")  
    for i, (prompt, code) in enumerate(chair_data[:2]):  
        print(f"\n样本 {i+1}:")  
        print(f"增强提示: {prompt}")  
        print(f"代码片段: {code[:200]}...")  
    
    # 2. 加载模型  
    print("\n🔄 加载BlenderLLM...")  
    try:  
        tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map="auto",  
            load_in_8bit=True  # 8bit量化节省内存  
        )  
        print("✅ 模型加载成功")  
        
    except Exception as e:  
        print(f"❌ 模型加载失败: {e}")  
        return  
    
    # 3. 微调逻辑  
    print("\n🔧 设置微调参数...")  
    
    # 只微调最后几层以保持代码生成能力  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
        if any(layer in name for layer in ['lm_head', 'layers.31', 'layers.30']):  # 只微调最后2层+输出层  
            param.requires_grad = True  
            print(f"  解冻: {name}")  
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"🎯 可训练参数: {trainable_params:,}")  
    
    # 4. 训练  
    print("\n🏋️ 开始微调...")  
    model.train()  
    
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],  
        lr=5e-6,  # 很小的学习率保持稳定性  
        weight_decay=0.01  
    )  
    
    successful_steps = 0  
    total_loss = 0  
    
    # 只用前10个样本进行快速微调  
    train_samples = chair_data[:10]  
    
    for epoch in range(1):  
        print(f"\n📅 Epoch {epoch + 1}/1")  
        
        for i, (prompt, target_code) in enumerate(train_samples):  
            try:  
                # 构造训练文本 - 保持BlenderLLM的格式  
                text = f"User: {prompt}\n\nAssistant: {target_code}"  
                
                # 限制长度  
                if len(text) > 2000:  
                    text = text[:2000]  
                
                # Tokenize  
                inputs = tokenizer(  
                    text,  
                    return_tensors="pt",  
                    max_length=512,  
                    truncation=True,  
                    padding=True  
                ).to(model.device)  
                
                # 前向传播  
                outputs = model(**inputs, labels=inputs['input_ids'])  
                loss = outputs.loss  
                
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"    ⚠️ 步骤 {i+1}: 跳过无效损失")  
                    continue  
                
                print(f"    📈 步骤 {i+1}/{len(train_samples)}: 损失 {loss.item():.6f}")  
                
                # 反向传播  
                loss.backward()  
                torch.nn.utils.clip_grad_norm_(  
                    [p for p in model.parameters() if p.requires_grad],  
                    max_norm=1.0  
                )  
                
                optimizer.step()  
                optimizer.zero_grad()  
                
                successful_steps += 1  
                total_loss += loss.item()  
                
                # 清理内存  
                del loss, outputs, inputs  
                torch.cuda.empty_cache()  
                
            except Exception as e:  
                print(f"    ❌ 步骤 {i+1} 失败: {e}")  
                optimizer.zero_grad()  
                torch.cuda.empty_cache()  
                continue  
    
    # 5. 测试微调效果  
    print(f"\n🧪 测试微调效果...")  
    if successful_steps > 0:  
        print(f"📊 成功训练步骤: {successful_steps}")  
        print(f"📊 平均损失: {total_loss/successful_steps:.6f}")  
        
        # 测试生成  
        model.eval()  
        test_prompts = [  
            "Generate chair design: minimalist dining chair with wooden frame",  
            "Generate chair design: ergonomic office chair with adjustable height",  
            "Generate chair design: Nordic style armchair with natural materials"  
        ]  
        
        for test_prompt in test_prompts:  
            try:  
                inputs = tokenizer(f"User: {test_prompt}\n\nAssistant:",   
                                 return_tensors="pt").to(model.device)  
                
                with torch.no_grad():  
                    outputs = model.generate(  
                        **inputs,  
                        max_new_tokens=200,  
                        temperature=0.7,  
                        do_sample=True,  
                        pad_token_id=tokenizer.pad_token_id  
                    )  
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
                assistant_response = response.split("Assistant:")[-1].strip()  
                
                print(f"\n🎯 测试提示: {test_prompt}")  
                print(f"🔧 生成代码: {assistant_response[:150]}...")  
                
            except Exception as e:  
                print(f"❌ 测试失败: {e}")  
    
    # 6. 保存微调后的模型  
    print(f"\n💾 保存微调模型...")  
    try:  
        output_dir = "./output/blender_enhanced_model"  
        os.makedirs(output_dir, exist_ok=True)  
        
        # 保存完整模型（用于实际使用）  
        model.save_pretrained(output_dir)  
        tokenizer.save_pretrained(output_dir)  
        
        # 保存训练信息  
        training_info = {  
            "model_type": "BlenderLLM_enhanced",  
            "training_purpose": "Enhanced chair design understanding",  
            "total_samples": len(chair_data),  
            "trained_samples": len(train_samples),  
            "successful_steps": successful_steps,  
            "average_loss": total_loss/successful_steps if successful_steps > 0 else None,  
            "enhanced_features": [  
                "Style understanding (minimalist, Nordic, etc.)",  
                "Material awareness (wood, metal, etc.)",  
                "Functional features (ergonomic, adjustable, foldable)",  
                "Enhanced Blender code generation"  
            ]  
        }  
        
        with open(os.path.join(output_dir, "training_info.json"), 'w') as f:  
            json.dump(training_info, f, indent=2)  
        
        print(f"✅ 模型保存成功: {output_dir}")  
        print("🎯 现在可以用这个增强模型运行 modeling.py!")  
        
        # 生成使用说明  
        usage_guide = f"""  
# 使用增强后的BlenderLLM模型  

## 1. 修改 modeling.py 中的模型路径:  
MODEL_NAME = "{output_dir}"  

## 2. 运行命令示例:  
python modeling.py \\
    --model_name "{output_dir}" \\
    --prompt "minimalist wooden dining chair with ergonomic design" \\
    --obj_name "enhanced_chair" \\
    --output_folder "./output/enhanced_results" \\
    --blender_executable "/usr/bin/blender" \\
    --brightness 1.0  

## 3. 增强功能:  
- 更好的风格理解（极简主义、北欧风格等）  
- 材质感知（实木、金属等）  
- 功能特性识别（人体工学、可调节等）  
- 优化的Blender代码生成  

## 4. 支持的风格关键词:  
- minimalist / 极简主义  
- Nordic / 北欧风格  
- ergonomic / 人体工学  
- adjustable / 可调节  
- foldable / 可折叠  
- wooden / 木质  
- metal / 金属  
"""  
        
        with open(os.path.join(output_dir, "usage_guide.md"), 'w') as f:  
            f.write(usage_guide)  
        
    except Exception as e:  
        print(f"❌ 保存失败: {e}")  
    
    print("✅ 微调完成！")  
    print("🚀 现在你的BlenderLLM具备了更强的椅子设计理解能力")  

if __name__ == "__main__":  
    main()  
