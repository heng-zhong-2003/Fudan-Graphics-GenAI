import bpy
import json
import math

#转换 BlendNet.jsonl 里的 CAD脚本 为 三视图， 三个视图独立输出为 {id}_[front/top/end]， 输出同名 txt prompt文件
#在 blender 内运行，需要安装 Freestyle SVG Exporter 插件

def process_cad_script(script_content, script_id, output_path, index):
    """处理单个CAD脚本并导出三视图"""
    try:
        # ===执行CAD脚本 === 
        exec(script_content)

        # === 设置纯白背景 ===
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()

        bg_node = nodes.new('ShaderNodeBackground')
        bg_node.inputs['Color'].default_value = (1,1,1,1) 
        bg_node.inputs['Strength'].default_value =5.0 #提高背景亮度
        output_node = nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

        bpy.context.scene.render.film_transparent = True 
        #透明背景模式
        bpy.context.scene.view_settings.look = 'AgX - Very High Contrast' #高对比色调映射

        #确保所有对象已加载
        bpy.context.view_layer.update()

        # ===== Freestyle SVG Exporter ===== #
        bpy.context.scene.render.use_freestyle = True

        # === 设置渲染器 ===
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' 

        bpy.context.scene.render.use_file_extension = True
        # === 启用Freestyle SVG Exporter ===
        if "Freestyle SVG Exporter" not in bpy.context.preferences.addons:
            try:
                bpy.ops.preferences.addon_enable(module="freestyle_svg_exporter") 
            except:
                print("警告：Freestyle SVG Exporter插件未安装")
        bpy.context.scene.svg_export.use_svg_export = True
        # ===== Freestyle SVG Exporter ===== #

        #获取场景中所有物体的总包围盒
        def get_scene_bounding_box():
            min_co = [float('inf')] *3 
            max_co = [-float('inf')] *3 
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                #获取物体在世界坐标系中的包围盒 
                    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
                    for i in range(3):
                        min_co[i] = min(min_co[i], min(v[i] for v in bbox))
                        max_co[i] = max(max_co[i], max(v[i] for v in bbox))
            return min_co, max_co
        
        #计算包围盒并确定物体尺寸
        min_co, max_co = get_scene_bounding_box()
        size = [max_co[i] - min_co[i] for i in range(3)]
        max_dim = max(size) #获取最大尺寸

        #创建相机并设置三视图
        def setup_ortho_camera(name, location, rotation, scale_factor=1.5):
            #如果相机已存在则删除 
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
            bpy.ops.object.camera_add(location=location, rotation=rotation)
            camera = bpy.context.object
            camera.name = name 
            camera.data.type = 'ORTHO' 
            #正交投影 
            camera.data.ortho_scale = max_dim * scale_factor #根据物体尺寸动态调整 
            return camera
        
        #定义三视图的相机位置和角度（X/Y/Z轴方向）
        #相机位置会根据物体中心点自动偏移
        center = [(min_co[i] + max_co[i]) /2 for i in range(3)]
        distance = max_dim *2 #相机距离基于物体大小
        cameras = {
            "front": ((center[0], center[1] - distance, center[2]), (math.pi/2,0,0)),        #主视图 
            "top": ((center[0], center[1], center[2] + distance), (0,0,0)),                  #俯视图 
            "side": ((center[0] + distance, center[1], center[2]), (math.pi/2,0, math.pi/2)) #侧视图
        }

        #创建相机
        for name, (loc, rot) in cameras.items():
            setup_ortho_camera(name, loc, rot, scale_factor=1.8) #1.8倍缩放确保留出边距

        #为每个相机渲染并保存图像
        for cam_name in cameras.keys():
            bpy.context.scene.camera = bpy.data.objects[cam_name]
            bpy.context.scene.render.filepath = f"{output_path}/{script_id}_{cam_name}"
            bpy.ops.render.render(write_still=True)

        return True 
    
    except Exception as e:
        print(f"处理脚本{index}时出错: {str(e)}")
        return False
    
def process_jsonl_file(jsonl_path, output_dir):
    """处理JSONL文件中的所有符合条件的CAD脚本"""
    success_count =0 
    total_count =0 
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                if "script" not in data:
                    print(f"第{i+1}行缺少'script'字段")
                    continue 
    
                if "chair" in data["name"] or "Chair" in data["name"]:      #过滤 椅子
                    total_count +=1 
                    print(f"处理第{total_count}个脚本...")
                    process_cad_script(data["script"], data["id"], output_dir, i+1)
                    #保存对应prompt
                    with open(f'{output_dir}/{data["id"]}.txt', 'w', encoding='utf-8') as f:
                        f.write(data["instruction"])

            except json.JSONDecodeError:
                print(f"第{i+1}行不是有效的JSON")
            except Exception as e:
                print(f"处理第{i+1}行时发生错误: {str(e)}")
            print(f"\n处理完成!成功: {success_count}/{total_count}")

#配置输入输出路径 
jsonl_path = "D:/CS/CG/PJ3/BlendNet.jsonl"  #JSONL文件路径 
output_dir = "D:/CS/CG/PJ3/output"          #输出目录 

#开始处理 
process_jsonl_file(jsonl_path, output_dir)