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
    
    for folder_name in os.listdir(data_dir)[:50]:  # 保持50个样本  
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
    """生成更详细的Blender代码 - 修复变量定义问题"""  
    
    # 转换为小写便于检测  
    desc_lower = description.lower()  
    
    # 检测风格特征  
    is_minimalist = any(word in desc_lower for word in ['minimalist', 'minimal', 'simple'])  
    is_modern = any(word in desc_lower for word in ['modern', 'contemporary', 'sleek'])  
    is_vintage = any(word in desc_lower for word in ['vintage', 'retro', 'classic', 'traditional'])  
    is_industrial = any(word in desc_lower for word in ['industrial', 'metal', 'steel'])  
    is_scandinavian = any(word in desc_lower for word in ['scandinavian', 'nordic', 'wood', 'wooden'])  
    is_ergonomic = any(word in desc_lower for word in ['ergonomic', 'comfortable', 'support'])  
    
    # 检测功能特征  
    has_armrest = any(word in desc_lower for word in ['armrest', 'arm rest', 'arms'])  
    has_wheels = any(word in desc_lower for word in ['wheel', 'caster', 'rolling', 'swivel'])  
    is_office_chair = any(word in desc_lower for word in ['office', 'desk', 'work'])  
    is_dining_chair = any(word in desc_lower for word in ['dining', 'kitchen', 'table'])  
    is_gaming_chair = any(word in desc_lower for word in ['gaming', 'game', 'racing'])  
    is_recliner = any(word in desc_lower for word in ['recliner', 'recline', 'lounge'])  
    is_bar_stool = any(word in desc_lower for word in ['bar', 'stool', 'high', 'counter'])  
    is_folding = any(word in desc_lower for word in ['folding', 'fold', 'portable'])  
    
    # 检测材质特征  
    is_leather = any(word in desc_lower for word in ['leather', 'hide'])  
    is_fabric = any(word in desc_lower for word in ['fabric', 'upholster', 'cushion', 'soft'])  
    is_plastic = any(word in desc_lower for word in ['plastic', 'acrylic', 'resin'])  
    
    # 检测尺寸特征  
    is_tall = any(word in desc_lower for word in ['tall', 'high back', 'high-back'])  
    is_wide = any(word in desc_lower for word in ['wide', 'broad', 'spacious'])  
    is_compact = any(word in desc_lower for word in ['compact', 'small', 'space-saving'])  
    
    code = '''import bpy  

# Clear existing objects  
bpy.ops.object.select_all(action='DESELECT')  
bpy.ops.object.select_by_type(type='MESH')  
bpy.ops.object.delete()  

'''  
    
    # 根据椅子类型调整基本尺寸  
    if is_bar_stool:  
        seat_height = 1.2  
        seat_scale = (0.35, 0.35, 0.03)  
        backrest_height = 1.5  
        leg_height = 1.0  
    elif is_compact:  
        seat_height = 0.4  
        seat_scale = (0.35, 0.3, 0.04)  
        backrest_height = 0.7  
        leg_height = 0.35  
    elif is_wide:  
        seat_height = 0.5  
        seat_scale = (0.55, 0.5, 0.05)  
        backrest_height = 0.9  
        leg_height = 0.45  
    else:  
        seat_height = 0.5  
        seat_scale = (0.45, 0.4, 0.05)  
        backrest_height = 0.85  
        leg_height = 0.45  
    
    # 创建座椅  
    code += f'''# Create seat  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, {seat_height}))  
seat = bpy.context.active_object  
seat.name = "Seat"  
'''  
    
    # 根据风格调整座椅形状  
    if is_minimalist:  
        code += f'seat.scale = ({seat_scale[0]-0.05}, {seat_scale[1]-0.05}, {seat_scale[2]-0.01})\n'  
    elif is_gaming_chair or is_ergonomic:  
        code += f'seat.scale = ({seat_scale[0]+0.1}, {seat_scale[1]+0.05}, {seat_scale[2]+0.02})\n'  
        # 添加座椅曲面  
        code += '''  
# Add seat curve for ergonomic design  
bpy.ops.object.modifier_add(type='BEVEL')  
seat.modifiers["Bevel"].width = 0.02  
'''  
    else:  
        code += f'seat.scale = {seat_scale}\n'  
    
    # 添加座椅材质效果  
    if is_leather:  
        code += '''  
# Add leather-like material  
mat_seat = bpy.data.materials.new(name="Leather_Seat")  
mat_seat.use_nodes = True  
mat_seat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.3, 0.2, 0.1, 1.0)  
mat_seat.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.8  
seat.data.materials.append(mat_seat)  
'''  
    elif is_fabric:  
        code += '''  
# Add fabric-like material  
mat_seat = bpy.data.materials.new(name="Fabric_Seat")  
mat_seat.use_nodes = True  
mat_seat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.4, 0.4, 0.6, 1.0)  
mat_seat.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.9  
seat.data.materials.append(mat_seat)  
'''  
    
    # 创建椅背  
    if not is_bar_stool or (is_bar_stool and 'back' in desc_lower):  
        backrest_location = f"(0, -0.35, {backrest_height})"  
        if is_tall:  
            backrest_scale = "(0.4, 0.04, 0.45)"  
        elif is_minimalist:  
            backrest_scale = "(0.35, 0.03, 0.3)"  
        elif is_gaming_chair:  
            backrest_scale = "(0.45, 0.06, 0.5)"  
        else:  
            backrest_scale = "(0.4, 0.04, 0.35)"  
            
        code += f'''  
# Create backrest  
bpy.ops.mesh.primitive_cube_add(size=2, location={backrest_location})  
backrest = bpy.context.active_object  
backrest.name = "Backrest"  
backrest.scale = {backrest_scale}  
'''  
        
        # 为人体工学椅子添加腰部支撑  
        if is_ergonomic or is_gaming_chair:  
            code += f'''  
# Add lumbar support  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.32, {backrest_height-0.1}))  
lumbar = bpy.context.active_object  
lumbar.name = "Lumbar_Support"  
lumbar.scale = (0.25, 0.03, 0.08)  
'''  
    
    # 创建扶手  
    if has_armrest:  
        if is_gaming_chair or is_office_chair:  
            code += f'''  
# Create adjustable armrests  
armrest_positions = [(-0.4, 0, {seat_height+0.2}), (0.4, 0, {seat_height+0.2})]  
for idx, pos in enumerate(armrest_positions):  
    # Armrest support  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.3, location=(pos[0], pos[1], pos[2]-0.1))  
    arm_support = bpy.context.active_object  
    arm_support.name = f"Armrest_Support_{{idx+1}}"  
    
    # Armrest pad  
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos)  
    armrest = bpy.context.active_object  
    armrest.name = f"Armrest_{{idx+1}}"  
    armrest.scale = (0.15, 0.25, 0.02)  
'''  
        else:  
            code += f'''  
# Create simple armrests  
armrest_positions = [(-0.35, 0, {seat_height+0.15}), (0.35, 0, {seat_height+0.15})]  
for idx, pos in enumerate(armrest_positions):  
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos)  
    armrest = bpy.context.active_object  
    armrest.name = f"Armrest_{{idx+1}}"  
    armrest.scale = (0.04, 0.3, 0.15)  
'''  
    
    # 创建支撑结构（腿部或底座）  
    if has_wheels or is_office_chair or is_gaming_chair:  
        code += f'''  
# Create office chair base with wheels  
bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=0.05, location=(0, 0, 0.05))  
base = bpy.context.active_object  
base.name = "Base"  

# Central column  
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth={seat_height-0.1}, location=(0, 0, {(seat_height-0.1)/2 + 0.05}))  
column = bpy.context.active_object  
column.name = "Central_Column"  

# Add wheels  
wheel_positions = [(0.35, 0, 0.05), (-0.35, 0, 0.05), (0, 0.35, 0.05), (0, -0.35, 0.05), (0.25, 0.25, 0.05)]  
for idx, pos in enumerate(wheel_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.02, location=pos)  
    wheel = bpy.context.active_object  
    wheel.name = f"Wheel_{{idx+1}}"  
    wheel.rotation_euler = (1.5708, 0, 0)  
'''  
    elif is_bar_stool:  
        code += f'''  
# Create bar stool legs with footrest  
leg_positions = [(-0.25, -0.25, {leg_height/2}), (0.25, -0.25, {leg_height/2}), (-0.25, 0.25, {leg_height/2}), (0.25, 0.25, {leg_height/2})]  
for idx, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth={leg_height}, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{{idx+1}}"  

# Add footrest  
bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.02, location=(0, 0, {leg_height*0.4}))  
footrest = bpy.context.active_object  
footrest.name = "Footrest"  
'''  
    elif is_folding:  
        code += f'''  
# Create folding chair legs  
leg_positions = [(-0.3, -0.3, {leg_height/2}), (0.3, -0.3, {leg_height/2}), (-0.3, 0.3, {leg_height/2}), (0.3, 0.3, {leg_height/2})]  
for idx, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth={leg_height}, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{{idx+1}}"  
    leg.rotation_euler = (0, 0.1, 0)  
    
# Add cross braces for stability  
bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.6, location=(0, 0, {leg_height*0.3}))  
brace1 = bpy.context.active_object  
brace1.name = "Cross_Brace_1"  
brace1.rotation_euler = (0, 0, 1.5708)  

bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.6, location=(0, 0, {leg_height*0.3}))  
brace2 = bpy.context.active_object  
brace2.name = "Cross_Brace_2"  
brace2.rotation_euler = (1.5708, 0, 0)  
'''  
    else:  
        # 标准四腿椅子  
        if is_industrial:  
            code += f'''  
# Create industrial style metal legs  
leg_positions = [(-0.35, -0.3, {leg_height/2}), (0.35, -0.3, {leg_height/2}), (-0.35, 0.3, {leg_height/2}), (0.35, 0.3, {leg_height/2})]  
for idx, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{{idx+1}}"  
    leg.scale = (0.02, 0.02, {leg_height/2})  
'''  
        elif is_scandinavian:  
            code += f'''  
# Create wooden style legs with slight taper  
leg_positions = [(-0.35, -0.3, {leg_height/2}), (0.35, -0.3, {leg_height/2}), (-0.35, 0.3, {leg_height/2}), (0.35, 0.3, {leg_height/2})]  
for idx, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.035, depth={leg_height}, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{{idx+1}}"  
    bpy.ops.object.modifier_add(type='BEVEL')  
    leg.modifiers["Bevel"].width = 0.005  
'''  
        else:  
            code += f'''  
# Create standard four legs  
leg_positions = [(-0.35, -0.3, {leg_height/2}), (0.35, -0.3, {leg_height/2}), (-0.35, 0.3, {leg_height/2}), (0.35, 0.3, {leg_height/2})]  
for idx, pos in enumerate(leg_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth={leg_height}, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"Leg_{{idx+1}}"  
'''  
    
    # 添加装饰性元素  
    if is_vintage or 'carved' in desc_lower:  
        code += '''  
# Add decorative elements for vintage style  
bpy.ops.mesh.primitive_torus_add(major_radius=0.08, minor_radius=0.01, location=(0, -0.35, 1.1))  
decoration = bpy.context.active_object  
decoration.name = "Backrest_Decoration"  

# Add carved details on backrest  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.33, 0.9))  
carving = bpy.context.active_object  
carving.name = "Carved_Detail"  
carving.scale = (0.25, 0.02, 0.1)  
'''  
    
    # 为可调节椅子添加调节机构  
    if 'adjustable' in desc_lower or 'height adjustable' in desc_lower:  
        code += f'''  
# Add height adjustment mechanism  
bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.1, location=(0, 0, {seat_height-0.15}))  
adjustment = bpy.context.active_object  
adjustment.name = "Height_Adjustment"  

# Add adjustment lever  
bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=0.08, location=(0.2, 0, {seat_height-0.1}))  
lever = bpy.context.active_object  
lever.name = "Adjustment_Lever"  
lever.rotation_euler = (0, 1.5708, 0)  
'''  
    
    # 为躺椅添加可调节靠背  
    if is_recliner:  
        code += '''  
# Add reclining mechanism  
bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.05, location=(-0.4, -0.2, 0.6))  
hinge = bpy.context.active_object  
hinge.name = "Reclining_Hinge"  
hinge.rotation_euler = (0, 1.5708, 0)  

# Add footrest for recliner  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0.5, 0.3))  
footrest = bpy.context.active_object  
footrest.name = "Footrest"  
footrest.scale = (0.35, 0.25, 0.02)  
'''  
    
    # 添加头枕（针对高背椅和游戏椅）  
    if is_gaming_chair or (is_tall and is_ergonomic):  
        code += f'''  
# Add headrest  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.3, {backrest_height+0.3}))  
headrest = bpy.context.active_object  
headrest.name = "Headrest"  
headrest.scale = (0.25, 0.08, 0.12)  

# Headrest support  
bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.15, location=(0, -0.35, {backrest_height+0.15}))  
headrest_support = bpy.context.active_object  
headrest_support.name = "Headrest_Support"  
'''  
    
    # 添加靠垫（针对舒适性椅子）  
    if is_fabric or 'cushion' in desc_lower or 'padded' in desc_lower:  
        code += f'''  
# Add seat cushion  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, {seat_height+0.03}))  
cushion = bpy.context.active_object  
cushion.name = "Seat_Cushion"  
cushion.scale = ({seat_scale[0]-0.02}, {seat_scale[1]-0.02}, 0.03)  

# Add cushion material  
mat_cushion = bpy.data.materials.new(name="Cushion_Material")  
mat_cushion.use_nodes = True  
mat_cushion.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.7, 0.6, 1.0)  
mat_cushion.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.7  
cushion.data.materials.append(mat_cushion)  

# Add backrest cushion  
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, -0.33, {backrest_height}))  
back_cushion = bpy.context.active_object  
back_cushion.name = "Backrest_Cushion"  
back_cushion.scale = (0.35, 0.03, 0.3)  
back_cushion.data.materials.append(mat_cushion)  
'''  
    
    # 添加额外的支撑结构  
    if 'reinforced' in desc_lower or is_industrial:  
        code += f'''  
# Add reinforcement bars  
cross_brace_positions = [  
    (0, -0.15, {leg_height*0.5}),   
    (0, 0.15, {leg_height*0.5})  
]  
for idx, pos in enumerate(cross_brace_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.7, location=pos)  
    brace = bpy.context.active_object  
    brace.name = f"Cross_Brace_{{idx+1}}"  
    brace.rotation_euler = (0, 0, 1.5708)  

# Side braces  
side_brace_positions = [  
    (-0.15, 0, {leg_height*0.5}),   
    (0.15, 0, {leg_height*0.5})  
]  
for idx, pos in enumerate(side_brace_positions):  
    bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.6, location=pos)  
    brace = bpy.context.active_object  
    brace.name = f"Side_Brace_{{idx+1}}"  
    brace.rotation_euler = (1.5708, 0, 0)  
'''  
    
    # 添加透明材质（针对亚克力椅子）  
    if is_plastic or 'transparent' in desc_lower or 'acrylic' in desc_lower:  
        code += '''  
# Add transparent acrylic material  
mat_acrylic = bpy.data.materials.new(name="Acrylic_Material")  
mat_acrylic.use_nodes = True  
mat_acrylic.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.9, 0.9, 1.0, 1.0)  
mat_acrylic.node_tree.nodes["Principled BSDF"].inputs[15].default_value = 1.0  
mat_acrylic.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.0  
mat_acrylic.node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.95  

# Apply to seat and backrest  
seat.data.materials.append(mat_acrylic)  
for obj in bpy.context.scene.objects:  
    if obj.name == "Backrest":  
        obj.data.materials.append(mat_acrylic)  
'''  
    
    # 添加木纹材质（针对木质椅子）  
    if is_scandinavian or 'wooden' in desc_lower:  
        code += '''  
# Add wood material  
mat_wood = bpy.data.materials.new(name="Wood_Material")  
mat_wood.use_nodes = True  
mat_wood.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.4, 0.25, 0.1, 1.0)  
mat_wood.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.8  

# Apply wood material to relevant parts  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH' and 'Leg' in obj.name:  
        obj.data.materials.append(mat_wood)  
    elif obj.name in ['Seat', 'Backrest'] and not (is_fabric or is_leather):  
        obj.data.materials.append(mat_wood)  
'''  
    
    # 添加金属材质（针对工业风格）  
    if is_industrial:  
        code += '''  
# Add metal material  
mat_metal = bpy.data.materials.new(name="Metal_Material")  
mat_metal.use_nodes = True  
mat_metal.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.7, 0.7, 0.8, 1.0)  
mat_metal.node_tree.nodes["Principled BSDF"].inputs[6].default_value = 1.0  
mat_metal.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.2  

# Apply to frame elements  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH' and any(name in obj.name for name in ['Leg', 'Brace', 'Support', 'Base']):  
        obj.data.materials.append(mat_metal)  
'''  
    
    # 添加特殊功能元素  
    if 'massage' in desc_lower:  
        code += '''  
# Add massage chair elements  
bpy.ops.mesh.primitive_sphere_add(radius=0.03, location=(0, -0.32, 0.8))  
massage_node = bpy.context.active_object  
massage_node.name = "Massage_Node"  
'''  
    
    if 'cup holder' in desc_lower or 'cupholder' in desc_lower:  
        code += f'''  
# Add cup holder  
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.05, location=(0.5, 0, {seat_height+0.15}))  
cup_holder = bpy.context.active_object  
cup_holder.name = "Cup_Holder"  
'''  
    
    # 最终调整和优化  
    code += '''  
# Final adjustments and positioning  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  

# Set smooth shading for organic shapes  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH':  
        bpy.context.view_layer.objects.active = obj  
        bpy.ops.object.shade_smooth()  

# Create a collection for organization  
collection = bpy.data.collections.new("Chair_Components")  
bpy.context.scene.collection.children.link(collection)  

# Move all objects to the collection  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH':  
        bpy.context.scene.collection.objects.unlink(obj)  
        collection.objects.link(obj)  

print("Chair generation completed successfully!")  
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
