#!/usr/bin/env python3  
"""  
使用真实的50个椅子样本进行微调  
"""  

import os  
import sys  
import json  
import torch  
import torch.nn as nn  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import gc  

def load_chair_data():  
    """加载真实的椅子数据"""  
    print("📊 加载真实椅子数据...")  
    
    data_dir = "./data_grouped"  
    if not os.path.exists(data_dir):  
        print(f"❌ 数据目录不存在: {data_dir}")  
        return []  
    
    chair_data = []  
    
    # 遍历所有椅子文件夹  
    for folder_name in os.listdir(data_dir):  
        folder_path = os.path.join(data_dir, folder_name)  
        if not os.path.isdir(folder_path):  
            continue  
        
        # 查找描述文件  
        txt_file = os.path.join(folder_path, f"{folder_name}.txt")  
        tags_file = os.path.join(folder_path, "tags.txt")  
        
        if os.path.exists(txt_file):  
            try:  
                # 读取椅子描述  
                with open(txt_file, 'r', encoding='utf-8') as f:  
                    description = f.read().strip()  
                
                # 读取标签（如果存在）  
                tags = ""  
                if os.path.exists(tags_file):  
                    with open(tags_file, 'r', encoding='utf-8') as f:  
                        tags = f.read().strip()  
                
                # 构造训练样本  
                prompt = f"Generate chair design: {description}"  
                if tags:  
                    prompt += f" (tags: {tags})"  
                
                # 生成对应的Blender代码（基于描述）  
                blender_code = generate_blender_code_from_description(description, tags)  
                
                chair_data.append((prompt, blender_code))  
                print(f"  ✅ 加载椅子: {folder_name[:8]}...")  
                
            except Exception as e:  
                print(f"  ⚠️ 跳过损坏文件 {folder_name}: {e}")  
                continue  
    
    print(f"📈 总共加载了 {len(chair_data)} 个椅子样本")  
    return chair_data  

def generate_blender_code_from_description(description, tags=""):  
    """根据描述生成对应的Blender代码"""  
    
    # 基础模板  
    code = """import bpy  
import math  

# Clear existing mesh objects  
bpy.ops.object.select_all(action='DESELECT')  
bpy.ops.object.select_by_type(type='MESH')  
bpy.ops.object.delete()  

"""  
    
    # 根据描述添加特定代码  
    desc_lower = description.lower()  
    
    # 椅子腿  
    if any(word in desc_lower for word in ['wooden', 'wood', 'timber']):  
        code += """# Create wooden chair legs  
for i in range(4):  
    x = 0.4 if i % 2 == 0 else -0.4  
    y = 0.4 if i < 2 else -0.4  
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.4))  
    leg = bpy.context.active_object  
    leg.scale = (0.05, 0.05, 0.4)  
    leg.name = f"leg_{i+1}"  

"""  
    elif any(word in desc_lower for word in ['metal', 'steel', 'chrome']):  
        code += """# Create metal chair legs  
for i in range(4):  
    x = 0.4 if i % 2 == 0 else -0.4  
    y = 0.4 if i < 2 else -0.4  
    bpy.ops.mesh.primitive_cylinder_add(location=(x, y, 0.4))  
    leg = bpy.context.active_object  
    leg.scale = (0.03, 0.03, 0.4)  
    leg.name = f"leg_{i+1}"  

"""  
    else:  
        code += """# Create chair legs  
for i in range(4):  
    x = 0.4 if i % 2 == 0 else -0.4  
    y = 0.4 if i < 2 else -0.4  
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.4))  
    leg = bpy.context.active_object  
    leg.scale = (0.05, 0.05, 0.4)  
    leg.name = f"leg_{i+1}"  

"""  
    
    # 椅子座面  
    if 'cushion' in desc_lower or 'soft' in desc_lower:  
        code += """# Create cushioned seat  
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.82))  
seat = bpy.context.active_object  
seat.scale = (0.45, 0.45, 0.05)  
seat.name = "seat"  

# Add cushion effect  
bpy.ops.object.modifier_add(type='SUBSURF')  
seat.modifiers["Subdivision Surface"].levels = 2  

"""  
    else:  
        code += """# Create seat  
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.82))  
seat = bpy.context.active_object  
seat.scale = (0.45, 0.45, 0.03)  
seat.name = "seat"  

"""  
    
    # 椅背  
    if 'high back' in desc_lower or 'tall' in desc_lower:  
        code += """# Create high backrest  
bpy.ops.mesh.primitive_cube_add(location=(0, 0.42, 1.2))  
backrest = bpy.context.active_object  
backrest.scale = (0.4, 0.03, 0.4)  
backrest.name = "backrest"  

"""  
    elif 'no back' in desc_lower or 'stool' in desc_lower:  
        pass  # 不添加椅背  
    else:  
        code += """# Create backrest  
bpy.ops.mesh.primitive_cube_add(location=(0, 0.42, 1.0))  
backrest = bpy.context.active_object  
backrest.scale = (0.4, 0.03, 0.2)  
backrest.name = "backrest"  

"""  
    
    # 扶手  
    if 'armchair' in desc_lower or 'armrest' in desc_lower:  
        code += """# Create armrests  
for i in range(2):  
    x = 0.5 if i == 0 else -0.5  
    bpy.ops.mesh.primitive_cube_add(location=(x, 0, 0.9))  
    armrest = bpy.context.active_object  
    armrest.scale = (0.03, 0.3, 0.03)  
    armrest.name = f"armrest_{i+1}"  

"""  
    
    # 材质  
    if 'wooden' in desc_lower:  
        code += """# Add wood material  
wood_material = bpy.data.materials.new(name="Wood")  
wood_material.diffuse_color = (0.55, 0.27, 0.07, 1.0)  # Brown wood color  

"""  
    elif 'metal' in desc_lower:  
        code += """# Add metal material  
metal_material = bpy.data.materials.new(name="Metal")  
metal_material.diffuse_color = (0.8, 0.8, 0.9, 1.0)  # Metallic color  
metal_material.metallic = 0.9  
metal_material.roughness = 0.1  

"""  
    
    return code.strip()  

def train_with_real_data():  
    """使用真实数据进行训练"""  
    print("🚀 使用真实50个椅子样本进行微调")  
    print("=" * 60)  
    
    # 1. 加载真实数据  
    chair_data = load_chair_data()  
    if len(chair_data) == 0:  
        print("❌ 没有加载到任何数据")  
        return  
    
    print(f"📊 训练样本数量: {len(chair_data)}")  
    
    # 显示几个样本  
    print("\n📝 样本预览:")  
    for i, (prompt, code) in enumerate(chair_data[:3]):  
        print(f"\n样本 {i+1}:")  
        print(f"提示: {prompt[:100]}...")  
        print(f"代码: {code[:200]}...")  
    
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
    
    # 4. 参数设置  
    print("🔒 设置可训练参数...")  
    trainable_count = 0  
    for name, param in model.named_parameters():  
        param.requires_grad = False  
        if 'lm_head' in name and 'weight' in name:  
            param.requires_grad = True  
            trainable_count += param.numel()  
            print(f"  解冻: {name}")  
    
    print(f"🎯 可训练参数: {trainable_count:,}")  
    
    # 5. 优化器  
    optimizer = torch.optim.AdamW(  
        [p for p in model.parameters() if p.requires_grad],  
        lr=5e-6,  # 稍微大一点的学习率  
        weight_decay=0.01  
    )  
    
    # 6. 训练  
    print("\n🏋️ 开始训练...")  
    model.train()  
    
    successful_steps = 0  
    total_loss = 0  
    
    for epoch in range(2):  # 训练2个epoch  
        print(f"\n📅 Epoch {epoch + 1}/2")  
        
        for i, (prompt, target) in enumerate(chair_data):  
            try:  
                # 构造训练文本  
                text = f"Human: {prompt}\n\nAssistant: {target}"  
                
                # Tokenize  
                inputs = tokenizer(  
                    text,  
                    return_tensors="pt",  
                    max_length=512,  # 增加长度以容纳完整代码  
                    truncation=True,  
                    padding=True  
                ).to(device)  
                
                # 前向传播  
                outputs = model(**inputs, labels=inputs['input_ids'])  
                loss = outputs.loss  
                
                if torch.isnan(loss) or torch.isinf(loss):  
                    print(f"    ⚠️ 步骤 {i+1}: 跳过无效损失")  
                    continue  
                
                print(f"    📈 步骤 {i+1}/{len(chair_data)}: 损失 {loss.item():.6f}")  
                
                # 反向传播  
                loss.backward()  
                
                # 梯度裁剪  
                torch.nn.utils.clip_grad_norm_(  
                    [p for p in model.parameters() if p.requires_grad],  
                    max_norm=1.0  
                )  
                
                # 更新  
                optimizer.step()  
                optimizer.zero_grad()  
                
                successful_steps += 1  
                total_loss += loss.item()  
                
                # 清理  
                del loss, outputs  
                torch.cuda.empty_cache()  
                
            except Exception as e:  
                print(f"    ❌ 步骤 {i+1} 失败: {e}")  
                optimizer.zero_grad()  
                torch.cuda.empty_cache()  
                continue  
    
    # 7. 保存结果  
    print(f"\n💾 保存模型...")  
    print(f"📊 成功训练步骤: {successful_steps}")  
    print(f"📊 平均损失: {total_loss/max(successful_steps, 1):.6f}")  
    
    if successful_steps > 0:  
        try:  
            output_dir = "./output/real_data_model"  
            os.makedirs(output_dir, exist_ok=True)  
            
            # 保存可训练参数  
            trainable_state = {  
                name: param.cpu() for name, param in model.named_parameters()  
                if param.requires_grad  
            }  
            
            torch.save(trainable_state, os.path.join(output_dir, "trainable_params.pt"))  
            tokenizer.save_pretrained(output_dir)  
            
            # 保存训练信息  
            training_info = {  
                "total_samples": len(chair_data),  
                "successful_steps": successful_steps,  
                "average_loss": total_loss/max(successful_steps, 1),  
                "sample_data": chair_data[:3]  # 保存前3个样本作为示例  
            }  
            
            with open(os.path.join(output_dir, "training_info.json"), 'w', encoding='utf-8') as f:  
                json.dump(training_info, f, indent=2, ensure_ascii=False)  
            
            print(f"✅ 保存成功: {output_dir}")  
            
        except Exception as e:  
            print(f"❌ 保存失败: {e}")  
    else:  
        print("❌ 没有成功的训练步骤，不保存模型")  
    
    print("✅ 训练完成")  

def parse_tags(tags_content):  
    """解析你的标签格式"""  
    tag_info = {}  
    
    lines = tags_content.strip().split('\n')  
    for line in lines:  
        if ':' in line:  
            key, value = line.split(':', 1)  
            key = key.strip()  
            value = value.strip()  
            
            # 处理null值  
            if value.lower() == 'null':  
                value = None  
            
            tag_info[key] = value  
    
    return tag_info  

def generate_blender_code_from_description(description, tags=""):  
    """根据描述和标签生成对应的Blender代码"""  
    
    # 解析标签  
    tag_info = {}  
    if tags:  
        tag_info = parse_tags(tags)  
    
    # 基础模板  
    code = """import bpy  
import math  

# Clear existing mesh objects  
bpy.ops.object.select_all(action='DESELECT')  
bpy.ops.object.select_by_type(type='MESH')  
bpy.ops.object.delete()  

"""  
    
    # 根据标签信息调整设计  
    desc_lower = description.lower()  
    
    # 根据风格调整  
    style = tag_info.get('现代风格', '')  
    material = tag_info.get('材质相关描述', '')  
    traditional = tag_info.get('传统/古典风格', '')  
    
    # 椅子腿设计  
    if material and '实木' in material:  
        code += """# Create solid wood chair legs  
for i in range(4):  
    x = 0.4 if i % 2 == 0 else -0.4  
    y = 0.4 if i < 2 else -0.4  
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.4))  
    leg = bpy.context.active_object  
    leg.scale = (0.08, 0.08, 0.4)  # Thicker for solid wood  
    leg.name = f"wood_leg_{i+1}"  
    
    # Add wood grain effect  
    bpy.ops.object.modifier_add(type='BEVEL')  
    leg.modifiers["Bevel"].width = 0.01  

"""  
    elif style and '极简主义' in style:  
        code += """# Create minimalist chair legs  
for i in range(4):  
    x = 0.35 if i % 2 == 0 else -0.35  
    y = 0.35 if i < 2 else -0.35  
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.35))  
    leg = bpy.context.active_object  
    leg.scale = (0.03, 0.03, 0.35)  # Very thin for minimalist design  
    leg.name = f"minimal_leg_{i+1}"  

"""  
    else:  
        code += """# Create standard chair legs  
for i in range(4):  
    x = 0.4 if i % 2 == 0 else -0.4  
    y = 0.4 if i < 2 else -0.4  
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.4))  
    leg = bpy.context.active_object  
    leg.scale = (0.05, 0.05, 0.4)  
    leg.name = f"leg_{i+1}"  

"""  
    
    # 座面设计  
    ergonomic = tag_info.get('人体工学符合性', '').lower()  
    
    if style and '极简主义' in style:  
        code += """# Create minimalist seat  
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.72))  
seat = bpy.context.active_object  
seat.scale = (0.4, 0.4, 0.02)  # Very thin for minimalist  
seat.name = "minimal_seat"  

"""  
    elif material and '实木' in material:  
        code += """# Create solid wood seat  
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.82))  
seat = bpy.context.active_object  
seat.scale = (0.45, 0.45, 0.04)  # Thicker wood seat  
seat.name = "wood_seat"  

# Add wood texture  
bpy.ops.object.modifier_add(type='BEVEL')  
seat.modifiers["Bevel"].width = 0.005  

"""  
    else:  
        code += """# Create standard seat  
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.82))  
seat = bpy.context.active_object  
seat.scale = (0.45, 0.45, 0.03)  
seat.name = "seat"  

"""  
    
    # 椅背设计  
    function_type = tag_info.get('功能型椅子', '')  
    
    if '凳子' in desc_lower or function_type == '凳子':  
        # 凳子不需要椅背  
        pass  
    elif style and '极简主义' in style:  
        code += """# Create minimalist backrest  
bpy.ops.mesh.primitive_cube_add(location=(0, 0.38, 1.0))  
backrest = bpy.context.active_object  
backrest.scale = (0.35, 0.02, 0.15)  # Thin minimalist backrest  
backrest.name = "minimal_backrest"  

"""  
    elif material and '实木' in material:  
        code += """# Create solid wood backrest  
bpy.ops.mesh.primitive_cube_add(location=(0, 0.42, 1.1))  
backrest = bpy.context.active_object  
backrest.scale = (0.4, 0.04, 0.25)  # Thicker wood backrest  
backrest.name = "wood_backrest"  

# Add wood detail  
bpy.ops.object.modifier_add(type='BEVEL')  
backrest.modifiers["Bevel"].width = 0.003  

"""  
    else:  
        code += """# Create standard backrest  
bpy.ops.mesh.primitive_cube_add(location=(0, 0.42, 1.0))  
backrest = bpy.context.active_object  
backrest.scale = (0.4, 0.03, 0.2)  
backrest.name = "backrest"  

"""  
    
    # 材质设置  
    if material and '实木' in material:  
        code += """# Add natural wood material  
wood_material = bpy.data.materials.new(name="NaturalWood")  
wood_material.use_nodes = True  
bsdf = wood_material.node_tree.nodes["Principled BSDF"]  
bsdf.inputs[0].default_value = (0.55, 0.35, 0.15, 1.0)  # Natural wood color  
bsdf.inputs[7].default_value = 0.8  # Roughness for wood texture  

# Apply material to all objects  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH' and 'leg' in obj.name or 'seat' in obj.name or 'backrest' in obj.name:  
        obj.data.materials.append(wood_material)  

"""  
    elif style and '极简主义' in style:  
        code += """# Add minimalist material  
minimal_material = bpy.data.materials.new(name="Minimalist")  
minimal_material.use_nodes = True  
bsdf = minimal_material.node_tree.nodes["Principled BSDF"]  
bsdf.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0)  # Clean white/light gray  
bsdf.inputs[7].default_value = 0.3  # Low roughness for clean look  

# Apply material to all objects  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH':  
        obj.data.materials.append(minimal_material)  

"""  
    
    # 功能性调整  
    adjustable_height = tag_info.get('高度可调节性', '').lower()  
    foldable = tag_info.get('折叠性', '').lower()  
    
    if adjustable_height and adjustable_height != '无':  
        code += """# Add height adjustment mechanism  
bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0.2))  
adjustment = bpy.context.active_object  
adjustment.scale = (0.08, 0.08, 0.1)  
adjustment.name = "height_adjustment"  

"""  
    
    if foldable and foldable != '无':  
        code += """# Add folding hinges  
for i in range(2):  
    x = 0.2 if i == 0 else -0.2  
    bpy.ops.mesh.primitive_cylinder_add(location=(x, 0.4, 0.8))  
    hinge = bpy.context.active_object  
    hinge.scale = (0.02, 0.02, 0.05)  
    hinge.rotation_euler = (1.5708, 0, 0)  # Rotate 90 degrees  
    hinge.name = f"hinge_{i+1}"  

"""  
    
    # 添加最终定位和渲染设置  
    code += """# Final positioning and camera setup  
bpy.ops.object.select_all(action='DESELECT')  

# Add basic lighting  
bpy.ops.object.light_add(type='SUN', location=(2, 2, 5))  
sun = bpy.context.active_object  
sun.data.energy = 3  

# Position camera for good view  
bpy.ops.object.camera_add(location=(3, -3, 2))  
camera = bpy.context.active_object  
camera.rotation_euler = (1.1, 0, 0.785)  

# Set camera as active  
bpy.context.scene.camera = camera  
"""  
    
    return code.strip()  

if __name__ == "__main__":  
    train_with_real_data()  
