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
    """运行Blender脚本生成3D模型和渲染图像"""  
    
    # 确保输出目录存在  
    os.makedirs(output_folder, exist_ok=True)  
    
    # 构建完整的Blender脚本  
    full_script = _build_complete_script(  
        script, obj_name, output_folder, bounding_coords,  
        camera_angles, brightness, save_obj, save_image, resolution  
    )  
    
    # 创建临时脚本文件  
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:  
        temp_file.write(full_script)  
        temp_script_path = temp_file.name  
    
    # 创建日志文件  
    log_path = os.path.join(output_folder, f"{obj_name}_blender.log")  
    
    try:  
        # 构建Blender命令  
        cmd = [  
            blender_executable,  
            "--background",  # 后台运行  
            "--factory-startup",  # 使用默认设置启动  
            "--python", temp_script_path  
        ]  
        
        print(f"🔄 正在运行Blender脚本: {obj_name}")  
        
        # 执行Blender命令  
        result = subprocess.run(  
            cmd,  
            capture_output=True,  
            text=True,  
            timeout=120  # 增加到120秒超时  
        )  
        
        # 保存日志  
        with open(log_path, 'w', encoding='utf-8') as log_file:  
            log_file.write(f"=== Blender执行日志: {obj_name} ===\n")  
            log_file.write(f"返回码: {result.returncode}\n")  
            log_file.write(f"标准输出:\n{result.stdout}\n")  
            log_file.write(f"错误输出:\n{result.stderr}\n")  
        
        if result.returncode == 0:  
            # 检查输出文件是否生成  
            success = True  
            missing_files = []  
            
            if save_image:  
                image_path = os.path.join(output_folder, f"{obj_name}.png")  
                if not os.path.exists(image_path):  
                    missing_files.append(f"图像文件: {image_path}")  
                    success = False  
                else:  
                    # 检查文件大小  
                    file_size = os.path.getsize(image_path)  
                    if file_size < 100:  # 小于100字节可能是空文件  
                        missing_files.append(f"图像文件过小: {image_path} ({file_size} bytes)")  
                        success = False  
            
            if save_obj:  
                obj_path = os.path.join(output_folder, f"{obj_name}.obj")  
                if not os.path.exists(obj_path):  
                    missing_files.append(f"OBJ文件: {obj_path}")  
                    success = False  
            
            if success:  
                print(f"✅ Blender脚本执行成功: {obj_name}")  
                return True  
            else:  
                print(f"❌ 输出文件生成失败: {obj_name}")  
                for missing in missing_files:  
                    print(f"   缺失: {missing}")  
                print(f"   详细日志: {log_path}")  
                return False  
        else:  
            print(f"❌ Blender脚本执行失败: {obj_name}")  
            print(f"   返回码: {result.returncode}")  
            print(f"   详细日志: {log_path}")  
            # 打印关键错误信息  
            if result.stderr:  
                error_lines = result.stderr.split('\n')  
                for line in error_lines[-10:]:  # 显示最后10行错误  
                    if line.strip():  
                        print(f"   错误: {line.strip()}")  
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
    """构建完整的Blender脚本，包含错误处理和调试信息"""  
    
    # 预处理用户脚本，修复常见语法错误  
    user_script = _fix_script_syntax(user_script)  
    
    # 默认相机角度  
    if camera_angles is None:  
        camera_angles = [(4, -4, 3), (1.1, 0, 0.785)]  
    
    # 默认亮度  
    if brightness is None:  
        brightness = (5.0, 1.5)  
    
    # 转义路径中的反斜杠  
    safe_output_folder = output_folder.replace('\\', '/')  
    
    # 缩进用户脚本  
    indented_user_script = '\n'.join('    ' + line for line in user_script.split('\n'))  
    
    complete_script = f'''  
import bpy  
import bmesh  
import mathutils  
import os  
import sys  
from mathutils import Vector  

print("=== 开始执行Blender脚本: {obj_name} ===")  

try:  
    # 清除默认场景  
    print("清除默认场景...")  
    bpy.ops.object.select_all(action='SELECT')  
    bpy.ops.object.delete(use_global=False)  
    
    # 清除所有材质和纹理  
    for material in bpy.data.materials:  
        bpy.data.materials.remove(material)  
    
    print("执行用户脚本...")  
    # 用户生成的脚本（已修复语法）  
{indented_user_script}  
    
    print("设置渲染参数...")  
    # 设置渲染引擎  
    bpy.context.scene.render.engine = 'CYCLES'  
    bpy.context.scene.render.resolution_x = {resolution[0]}  
    bpy.context.scene.render.resolution_y = {resolution[1]}  
    bpy.context.scene.render.resolution_percentage = 100  
    
    # Cycles设置优化  
    bpy.context.scene.cycles.samples = 64  # 提高采样数  
    bpy.context.scene.cycles.use_denoising = True  
    bpy.context.scene.cycles.device = 'CPU'  
    
    print("添加光照...")  
    # 添加光照系统  
    # 主光源 - 太阳光  
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))  
    sun_light = bpy.context.active_object  
    sun_light.data.energy = {brightness[0]}  
    sun_light.rotation_euler = mathutils.Euler((0.5, 0.5, 0), 'XYZ')  
    
    # 补充光源  
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))  
    area_light = bpy.context.active_object  
    area_light.data.energy = {brightness[0] * 0.5}  
    area_light.data.size = 2  
    
    # 环境光  
    bpy.context.scene.world.use_nodes = True  
    world_nodes = bpy.context.scene.world.node_tree.nodes  
    world_bg = world_nodes.get('Background')  
    if world_bg:  
        world_bg.inputs[1].default_value = {brightness[1]}  
    
    print("设置相机...")  
    # 添加相机  
    bpy.ops.object.camera_add(location=({camera_angles[0][0]}, {camera_angles[0][1]}, {camera_angles[0][2]}))  
    camera = bpy.context.active_object  
    bpy.context.scene.camera = camera  
    camera.rotation_euler = mathutils.Euler(({camera_angles[1][0]}, {camera_angles[1][1]}, {camera_angles[1][2]}), 'XYZ')  
    
    # 计算场景中心并调整相机  
    all_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']  
    print(f"找到 {{len(all_objects)}} 个网格对象")  
    
    if all_objects:  
        # 计算所有物体的边界框中心  
        min_coords = [float('inf')] * 3  
        max_coords = [float('-inf')] * 3  
        
        for obj in all_objects:  
            if obj.data and obj.data.vertices:  
                for vertex in obj.data.vertices:  
                    world_vertex = obj.matrix_world @ vertex.co  
                    for i in range(3):  
                        min_coords[i] = min(min_coords[i], world_vertex[i])  
                        max_coords[i] = max(max_coords[i], world_vertex[i])  
        
        if all(coord != float('inf') for coord in min_coords):  
            center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]  
            print(f"场景中心: {{center}}")  
            
            # 创建一个空物体作为目标  
            bpy.ops.object.empty_add(location=center)  
            target = bpy.context.active_object  
            target.name = "camera_target"  
            
            # 设置相机约束  
            constraint = camera.constraints.new(type='TRACK_TO')  
            constraint.target = target  
            constraint.track_axis = 'TRACK_NEGATIVE_Z'  
            constraint.up_axis = 'UP_Y'  
        else:  
            print("警告: 无法计算有效的场景边界框")  
    else:  
        print("警告: 场景中没有网格对象")  
    
    print("添加材质...")  
    # 为没有材质的对象添加默认材质  
    for obj in bpy.context.scene.objects:  
        if obj.type == 'MESH' and len(obj.data.materials) == 0:  
            mat = bpy.data.materials.new(name=f"{{obj.name}}_material")  
            mat.use_nodes = True  
            
            bsdf = mat.node_tree.nodes.get('Principled BSDF')  
            if bsdf:  
                bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)  
                bsdf.inputs['Roughness'].default_value = 0.4  
                bsdf.inputs['Metallic'].default_value = 0.0  
            
            obj.data.materials.append(mat)  
            print(f"为对象 {{obj.name}} 添加了材质")  
    
    # 保存OBJ文件（如果需要）  
    if {save_obj}:  
        print("保存OBJ文件...")  
        bpy.ops.object.select_all(action='DESELECT')  
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']  
        
        if mesh_objects:  
            for obj in mesh_objects:  
                obj.select_set(True)  
            
            obj_path = os.path.join(r"{safe_output_folder}", "{obj_name}.obj")  
            bpy.ops.export_scene.obj(filepath=obj_path, use_selection=True)  
            print(f"OBJ文件已保存: {{obj_path}}")  
        else:  
            print("警告: 没有网格对象可导出")  
    
    # 渲染图像（如果需要）  
    if {save_image}:  
        print("开始渲染...")  
        
        # 检查是否有可渲染的对象  
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']  
        if not mesh_objects:  
            print("错误: 没有网格对象可渲染，创建备用椅子...")  
            _create_fallback_chair()  
        
        image_path = os.path.join(r"{safe_output_folder}", "{obj_name}.png")  
        bpy.context.scene.render.filepath = image_path  
        
        # 执行渲染  
        bpy.ops.render.render(write_still=True)  
        
        # 验证文件是否生成  
        if os.path.exists(image_path):  
            file_size = os.path.getsize(image_path)  
            print(f"图像已渲染: {{image_path}} ({{file_size}} bytes)")  
        else:  
            print(f"错误: 图像文件未生成: {{image_path}}")  
    
    print("=== Blender脚本执行完成 ===")  

except Exception as e:  
    print(f"错误: {{str(e)}}")  
    import traceback  
    traceback.print_exc()  
    
    # 尝试创建一个简单的场景作为备用  
    try:  
        print("尝试创建备用场景...")  
        _create_fallback_chair()  
        
        # 重新尝试渲染  
        if {save_image}:  
            # 添加基础光照  
            bpy.ops.object.light_add(type='SUN', location=(3, 3, 5))  
            sun = bpy.context.active_object  
            sun.data.energy = 3.0  
            
            # 添加相机  
            bpy.ops.object.camera_add(location=(2, -2, 1.5))  
            camera = bpy.context.active_object  
            bpy.context.scene.camera = camera  
            camera.rotation_euler = mathutils.Euler((1.1, 0, 0.785), 'XYZ')  
            
            # 渲染  
            image_path = os.path.join(r"{safe_output_folder}", "{obj_name}.png")  
            bpy.context.scene.render.filepath = image_path  
            bpy.ops.render.render(write_still=True)  
            print(f"备用场景渲染完成: {{image_path}}")  
            
    except Exception as backup_error:  
        print(f"备用场景创建也失败: {{str(backup_error)}}")  

def _create_fallback_chair():  
    """创建备用椅子"""  
    bpy.ops.object.select_all(action='SELECT')  
    bpy.ops.object.delete(use_global=False)  
    
    # 创建简单的椅子形状作为备用  
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0.25), scale=(0.4, 0.4, 0.05))  
    seat = bpy.context.active_object  
    seat.name = "chair_seat"  
    
    bpy.ops.mesh.primitive_cube_add(location=(0, -0.35, 0.5), scale=(0.4, 0.05, 0.25))  
    backrest = bpy.context.active_object  
    backrest.name = "chair_backrest"  
    
    # 腿部  
    for i, pos in enumerate([(-0.3, -0.3, 0.125), (0.3, -0.3, 0.125), (-0.3, 0.3, 0.125), (0.3, 0.3, 0.125)]):  
        bpy.ops.mesh.primitive_cube_add(location=pos, scale=(0.03, 0.03, 0.125))  
        leg = bpy.context.active_object  
        leg.name = f"chair_leg_{{i+1}}"  
    
    print("备用椅子场景已创建")  
'''  
    
    return complete_script  

def _fix_script_syntax(script: str) -> str:  
    """修复脚本中的常见语法错误"""  
    lines = script.split('\n')  
    fixed_lines = []  
    
    for line in lines:  
        # 修复未闭合的字符串  
        if line.count('"') % 2 == 1:  
            line = line + '"'  
        if line.count("'") % 2 == 1:  
            line = line + "'"  
        
        # 修复f-string语法错误  
        if 'f"' in line and line.count('{') != line.count('}'):  
            # 简单修复：移除f前缀  
            line = line.replace('f"', '"').replace("f'", "'")  
        
        fixed_lines.append(line)  
    
    return '\n'.join(fixed_lines)  

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
        "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",  # Windows  
        "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",  # Windows  
        "../BlenderModel/blender-4.4.3-linux-x64/blender",  # 项目相对路径  
        "./blender/blender",  # 当前目录  
    ]  
    
    for path in possible_paths:  
        if validate_blender_executable(path):  
            return path  
    
    return "blender"  # 默认返回  

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