"""  
Blender脚本运行器  
基于BlenderLLM项目修改，适配椅子设计评估需求  
"""  

import subprocess  
import os  
import tempfile  
import sys  
from typing import List, Tuple, Optional  

def run_blender_script(  
    script: str,  
    obj_name: str,  
    output_folder: str,  
    bounding_coords: Optional[List] = None,  
    camera_angles: Optional[List] = None,  
    brightness: Optional[Tuple] = None,  
    blender_executable: str = "blender",  
    save_obj: bool = False,  
    save_image: bool = True,  
    resolution: Tuple[int, int] = (512, 512)  
) -> bool:  
    """  
    运行Blender脚本生成3D模型和渲染图像  
    
    Args:  
        script: Blender Python脚本内容  
        obj_name: 对象名称  
        output_folder: 输出目录  
        bounding_coords: 边界框坐标（可选）  
        camera_angles: 相机角度（可选）  
        brightness: 亮度设置（可选）  
        blender_executable: Blender可执行文件路径  
        save_obj: 是否保存OBJ文件  
        save_image: 是否保存渲染图像  
        resolution: 渲染分辨率  
        
    Returns:  
        bool: 是否成功执行  
    """  
    
    # 确保输出目录存在  
    os.makedirs(output_folder, exist_ok=True)  
    
    # 构建完整的Blender脚本  
    full_script = _build_complete_script(  
        script, obj_name, output_folder, bounding_coords,  
        camera_angles, brightness, save_obj, save_image, resolution  
    )  
    
    # 创建临时脚本文件  
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:  
        temp_file.write(full_script)  
        temp_script_path = temp_file.name  
    
    try:  
        # 构建Blender命令  
        cmd = [  
            blender_executable,  
            "--background",  # 后台运行  
            "--python", temp_script_path  
        ]  
        
        print(f"🔄 正在运行Blender脚本: {obj_name}")  
        
        # 执行Blender命令  
        result = subprocess.run(  
            cmd,  
            capture_output=True,  
            text=True,  
            timeout=60  # 60秒超时  
        )  
        
        if result.returncode == 0:  
            # 检查输出文件是否生成  
            success = True  
            if save_image:  
                image_path = os.path.join(output_folder, f"{obj_name}.png")  
                if not os.path.exists(image_path):  
                    print(f"⚠️ 图像文件未生成: {image_path}")  
                    success = False  
            
            if save_obj:  
                obj_path = os.path.join(output_folder, f"{obj_name}.obj")  
                if not os.path.exists(obj_path):  
                    print(f"⚠️ OBJ文件未生成: {obj_path}")  
                    success = False  
            
            if success:  
                print(f"✅ Blender脚本执行成功: {obj_name}")  
                return True  
            else:  
                print(f"❌ 输出文件生成失败: {obj_name}")  
                return False  
        else:  
            print(f"❌ Blender脚本执行失败: {obj_name}")  
            print(f"错误输出: {result.stderr}")  
            return False  
            
    except subprocess.TimeoutExpired:  
        print(f"⏰ Blender脚本执行超时: {obj_name}")  
        return False  
    except Exception as e:  
        print(f"❌ 执行Blender脚本时发生错误: {e}")  
        return False  
    finally:  
        # 清理临时文件  
        try:  
            os.unlink(temp_script_path)  
        except:  
            pass  

def _build_complete_script(  
    user_script: str,  
    obj_name: str,  
    output_folder: str,  
    bounding_coords: Optional[List] = None,  
    camera_angles: Optional[List] = None,  
    brightness: Optional[Tuple] = None,  
    save_obj: bool = False,  
    save_image: bool = True,  
    resolution: Tuple[int, int] = (512, 512)  
) -> str:  
    """  
    构建完整的Blender脚本，包含用户脚本和渲染设置  
    """  
    
    # 默认相机角度  
    if camera_angles is None:  
        camera_angles = [(3, -3, 2), (1.1, 0, 0.785)]  # 位置和旋转  
    
    # 默认亮度  
    if brightness is None:  
        brightness = (3.0, 1.0)  # (sun_energy, world_strength)  
    
    complete_script = f'''  
import bpy  
import bmesh  
import mathutils  
import os  
from mathutils import Vector  

# 清除默认场景  
bpy.ops.object.select_all(action='SELECT')  
bpy.ops.object.delete(use_global=False)  

# 用户生成的脚本  
{user_script}  

# 设置渲染引擎  
bpy.context.scene.render.engine = 'CYCLES'  
bpy.context.scene.render.resolution_x = {resolution[0]}  
bpy.context.scene.render.resolution_y = {resolution[1]}  
bpy.context.scene.render.resolution_percentage = 100  

# 添加光照  
# 主光源 - 太阳光  
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))  
sun_light = bpy.context.active_object  
sun_light.data.energy = {brightness[0]}  
sun_light.rotation_euler = mathutils.Euler((0.5, 0.5, 0), 'XYZ')  

# 环境光  
bpy.context.scene.world.use_nodes = True  
world_nodes = bpy.context.scene.world.node_tree.nodes  
world_bg = world_nodes.get('Background')  
if world_bg:  
    world_bg.inputs[1].default_value = {brightness[1]}  # Strength  

# 添加相机  
bpy.ops.object.camera_add(location=({camera_angles[0][0]}, {camera_angles[0][1]}, {camera_angles[0][2]}))  
camera = bpy.context.active_object  
bpy.context.scene.camera = camera  

# 设置相机旋转  
camera.rotation_euler = mathutils.Euler(({camera_angles[1][0]}, {camera_angles[1][1]}, {camera_angles[1][2]}), 'XYZ')  

# 让相机对焦到场景中心  
# 计算所有物体的中心  
all_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']  
if all_objects:  
    # 计算所有物体的边界框中心  
    min_coords = [float('inf')] * 3  
    max_coords = [float('-inf')] * 3  
    
    for obj in all_objects:  
        for vertex in obj.data.vertices:  
            world_vertex = obj.matrix_world @ vertex.co  
            for i in range(3):  
                min_coords[i] = min(min_coords[i], world_vertex[i])  
                max_coords[i] = max(max_coords[i], world_vertex[i])  
    
    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]  
    
    # 设置相机约束，让其始终看向中心  
    constraint = camera.constraints.new(type='TRACK_TO')  
    
    # 创建一个空物体作为目标  
    bpy.ops.object.empty_add(location=center)  
    target = bpy.context.active_object  
    target.name = "camera_target"  
    
    constraint.target = target  
    constraint.track_axis = 'TRACK_NEGATIVE_Z'  
    constraint.up_axis = 'UP_Y'  

# 添加材质增强（如果对象没有材质）  
for obj in bpy.context.scene.objects:  
    if obj.type == 'MESH' and len(obj.data.materials) == 0:  
        # 创建默认材质  
        mat = bpy.data.materials.new(name=f"{{obj.name}}_material")  
        mat.use_nodes = True  
        
        # 设置基础材质属性  
        bsdf = mat.node_tree.nodes.get('Principled BSDF')  
        if bsdf:  
            bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # 浅灰色  
            bsdf.inputs['Roughness'].default_value = 0.3  
            bsdf.inputs['Metallic'].default_value = 0.1  
        
        obj.data.materials.append(mat)  

# 保存OBJ文件（如果需要）  
if {save_obj}:  
    bpy.ops.object.select_all(action='DESELECT')  
    for obj in bpy.context.scene.objects:  
        if obj.type == 'MESH':  
            obj.select_set(True)  
    
    if len(bpy.context.selected_objects) > 0:  
        obj_path = os.path.join(r"{output_folder}", "{obj_name}.obj")  
        bpy.ops.export_scene.obj(filepath=obj_path, use_selection=True)  
        print(f"OBJ文件已保存: {{obj_path}}")  

# 渲染图像（如果需要）  
if {save_image}:  
    image_path = os.path.join(r"{output_folder}", "{obj_name}.png")  
    bpy.context.scene.render.filepath = image_path  
    bpy.ops.render.render(write_still=True)  
    print(f"图像已渲染: {{image_path}}")  

print("Blender脚本执行完成!")  
'''  
    
    return complete_script  

def validate_blender_executable(blender_path: str) -> bool:  
    """  
    验证Blender可执行文件是否有效  
    
    Args:  
        blender_path: Blender可执行文件路径  
        
    Returns:  
        bool: 是否为有效的Blender可执行文件  
    """  
    try:  
        result = subprocess.run(  
            [blender_path, "--version"],  
            capture_output=True,  
            text=True,  
            timeout=10  
        )  
        return result.returncode == 0 and "Blender" in result.stdout  
    except:  
        return False  

def get_default_blender_path() -> str:  
    """  
    获取默认的Blender可执行文件路径  
    
    Returns:  
        str: Blender可执行文件路径  
    """  
    # 常见的Blender安装路径  
    possible_paths = [  
        "blender",  # 系统PATH中  
        "/usr/bin/blender",  # Linux默认  
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS  
        "../BlenderModel/blender-4.4.3-linux-x64/blender",  # 项目相对路径  
    ]  
    
    for path in possible_paths:  
        if validate_blender_executable(path):  
            return path  
    
    return "blender"  # 默认返回