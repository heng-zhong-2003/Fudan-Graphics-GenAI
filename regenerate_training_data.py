#!/usr/bin/env python3  
"""  
重新生成椅子设计训练数据  
"""  

import json  
import os  

def create_chair_training_data():  
    """创建椅子设计的训练数据"""  
    
    training_data = []  
    
    # 椅子设计案例  
    chair_designs = [  
        {  
            "input": "Generate chair design: modern minimalist office chair",  
            "output": """import bpy  
import bmesh  

# Clear existing mesh objects  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# Create modern office chair base  
bpy.ops.mesh.add_object.cylinder_add(radius=0.3, depth=0.1, location=(0, 0, 0.05))  
base = bpy.context.active_object  
base.name = "ChairBase"  

# Create chair stem  
bpy.ops.mesh.add_object.cylinder_add(radius=0.05, depth=0.6, location=(0, 0, 0.4))  
stem = bpy.context.active_object  
stem.name = "ChairStem"  

# Create seat  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, 0, 0.8))  
seat = bpy.context.active_object  
seat.name = "ChairSeat"  
seat.scale = (0.4, 0.4, 0.05)  

# Create backrest  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, -0.35, 1.2))  
backrest = bpy.context.active_object  
backrest.name = "ChairBackrest"  
backrest.scale = (0.4, 0.05, 0.3)  

print("Modern minimalist office chair created successfully!")"""  
        },  
        {  
            "input": "Generate chair design: vintage wooden dining chair",  
            "output": """import bpy  

# Clear scene  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# Create wooden chair legs  
leg_positions = [(-0.15, -0.15, 0.4), (0.15, -0.15, 0.4), (-0.15, 0.15, 0.4), (0.15, 0.15, 0.4)]  

for i, pos in enumerate(leg_positions):  
    bpy.ops.mesh.add_object.cube_add(size=2, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"ChairLeg_{i+1}"  
    leg.scale = (0.02, 0.02, 0.4)  

# Create wooden seat  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, 0, 0.8))  
seat = bpy.context.active_object  
seat.name = "WoodenSeat"  
seat.scale = (0.35, 0.35, 0.03)  

# Create wooden backrest with vintage style  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, -0.3, 1.1))  
backrest = bpy.context.active_object  
backrest.name = "WoodenBackrest"  
backrest.scale = (0.35, 0.05, 0.25)  

# Add wood material  
material = bpy.data.materials.new(name="WoodMaterial")  
material.use_nodes = True  
material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.6, 0.4, 0.2, 1.0)  

print("Vintage wooden dining chair created!")"""  
        },  
        {  
            "input": "Generate chair design: comfortable armchair with cushions",  
            "output": """import bpy  

# Clear existing objects  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# Create armchair base  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, 0, 0.3))  
base = bpy.context.active_object  
base.name = "ArmchairBase"  
base.scale = (0.6, 0.6, 0.15)  

# Create comfortable seat cushion  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, 0, 0.5))  
seat_cushion = bpy.context.active_object  
seat_cushion.name = "SeatCushion"  
seat_cushion.scale = (0.55, 0.55, 0.08)  

# Create backrest  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, -0.45, 0.9))  
backrest = bpy.context.active_object  
backrest.name = "Backrest"  
backrest.scale = (0.55, 0.15, 0.4)  

# Create armrests  
armrest_positions = [(-0.4, 0, 0.7), (0.4, 0, 0.7)]  
for i, pos in enumerate(armrest_positions):  
    bpy.ops.mesh.add_object.cube_add(size=2, location=pos)  
    armrest = bpy.context.active_object  
    armrest.name = f"Armrest_{i+1}"  
    armrest.scale = (0.12, 0.5, 0.08)  

# Create cushion material  
cushion_material = bpy.data.materials.new(name="CushionMaterial")  
cushion_material.use_nodes = True  
cushion_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.6, 0.4, 1.0)  

print("Comfortable armchair with cushions created!")"""  
        },  
        {  
            "input": "Generate chair design: sleek bar stool",  
            "output": """import bpy  

# Clear scene  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# Create bar stool base  
bpy.ops.mesh.add_object.cylinder_add(radius=0.25, depth=0.05, location=(0, 0, 0.025))  
base = bpy.context.active_object  
base.name = "StoolBase"  

# Create central support pole  
bpy.ops.mesh.add_object.cylinder_add(radius=0.03, depth=1.2, location=(0, 0, 0.6))  
support_pole = bpy.context.active_object  
support_pole.name = "SupportPole"  

# Create circular seat  
bpy.ops.mesh.add_object.cylinder_add(radius=0.2, depth=0.05, location=(0, 0, 1.2))  
seat = bpy.context.active_object  
seat.name = "StoolSeat"  

# Create footrest ring  
bpy.ops.mesh.add_object.torus_add(major_radius=0.15, minor_radius=0.02, location=(0, 0, 0.4))  
footrest = bpy.context.active_object  
footrest.name = "Footrest"  

# Add metallic material  
metal_material = bpy.data.materials.new(name="MetalMaterial")  
metal_material.use_nodes = True  
bsdf = metal_material.node_tree.nodes["Principled BSDF"]  
bsdf.inputs[0].default_value = (0.7, 0.7, 0.7, 1.0)  
bsdf.inputs[4].default_value = 1.0  # Metallic  
bsdf.inputs[7].default_value = 0.1  # Roughness  

print("Sleek bar stool created successfully!")"""  
        },  
        {  
            "input": "Generate chair design: ergonomic gaming chair",  
            "output": """import bpy  

# Clear existing objects  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# Create gaming chair base with wheels  
bpy.ops.mesh.add_object.cylinder_add(radius=0.3, depth=0.1, location=(0, 0, 0.05))  
base = bpy.context.active_object  
base.name = "GamingChairBase"  

# Create gas cylinder  
bpy.ops.mesh.add_object.cylinder_add(radius=0.04, depth=0.5, location=(0, 0, 0.35))  
gas_cylinder = bpy.context.active_object  
gas_cylinder.name = "GasCylinder"  

# Create ergonomic seat  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, 0, 0.7))  
seat = bpy.context.active_object  
seat.name = "ErgonomicSeat"  
seat.scale = (0.45, 0.4, 0.08)  

# Create high backrest  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, -0.35, 1.2))  
backrest = bpy.context.active_object  
backrest.name = "HighBackrest"  
backrest.scale = (0.45, 0.1, 0.5)  

# Create adjustable armrests  
armrest_positions = [(-0.3, 0, 0.85), (0.3, 0, 0.85)]  
for i, pos in enumerate(armrest_positions):  
    bpy.ops.mesh.add_object.cube_add(size=2, location=pos)  
    armrest = bpy.context.active_object  
    armrest.name = f"AdjustableArmrest_{i+1}"  
    armrest.scale = (0.08, 0.25, 0.05)  

# Create headrest  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, -0.4, 1.6))  
headrest = bpy.context.active_object  
headrest.name = "Headrest"  
headrest.scale = (0.3, 0.08, 0.12)  

print("Ergonomic gaming chair created!")"""  
        }  
    ]  
    
    # 添加更多变化  
    styles = ["modern", "traditional", "industrial", "scandinavian", "vintage"]  
    materials = ["wood", "metal", "plastic", "leather", "fabric"]  
    chair_types = ["dining chair", "office chair", "lounge chair", "bar stool", "accent chair"]  
    
    for style in styles:  
        for material in materials:  
            for chair_type in chair_types:  
                input_text = f"Generate chair design: {style} {material} {chair_type}"  
                output_text = f"""import bpy  

# Clear scene  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# Create {style} {material} {chair_type}  
# Basic chair structure  

# Create seat  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, 0, 0.8))  
seat = bpy.context.active_object  
seat.name = "ChairSeat"  
seat.scale = (0.4, 0.4, 0.05)  

# Create backrest  
bpy.ops.mesh.add_object.cube_add(size=2, location=(0, -0.35, 1.1))  
backrest = bpy.context.active_object  
backrest.name = "ChairBackrest"   
backrest.scale = (0.4, 0.05, 0.25)  

# Create legs  
leg_positions = [(-0.15, -0.15, 0.4), (0.15, -0.15, 0.4), (-0.15, 0.15, 0.4), (0.15, 0.15, 0.4)]  
for i, pos in enumerate(leg_positions):  
    bpy.ops.mesh.add_object.cube_add(size=2, location=pos)  
    leg = bpy.context.active_object  
    leg.name = f"ChairLeg_{{i+1}}"  
    leg.scale = (0.03, 0.03, 0.4)  

print("{style} {material} {chair_type} created successfully!")"""  
                
                training_data.append({  
                    "input": input_text,  
                    "output": output_text  
                })  
    
    # 添加预定义的详细案例  
    training_data.extend(chair_designs)  
    
    return training_data  

def main():  
    print("🪑 重新生成椅子设计训练数据...")  
    
    # 生成训练数据  
    training_data = create_chair_training_data()  
    
    print(f"✅ 生成了 {len(training_data)} 个训练样本")  
    
    # 保存数据  
    output_dir = "./output/new_training_data"  
    os.makedirs(output_dir, exist_ok=True)  
    
    output_file = os.path.join(output_dir, "chair_training_data.json")  
    with open(output_file, 'w', encoding='utf-8') as f:  
        json.dump(training_data, f, ensure_ascii=False, indent=2)  
    
    print(f"💾 数据保存到: {output_file}")  
    
    # 显示样本  
    print("\n📝 样本预览:")  
    for i in range(min(3, len(training_data))):  
        sample = training_data[i]  
        print(f"\n样本 {i+1}:")  
        print(f"  输入: {sample['input']}")  
        print(f"  输出: {sample['output'][:150]}...")  

if __name__ == "__main__":  
    main()  
