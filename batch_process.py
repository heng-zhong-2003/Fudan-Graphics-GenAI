# batch_process.py  
#!/usr/bin/env python3  
"""  
批量处理椅子风格生成任务  
"""  

import os  
import sys  
import json  
import argparse  
from pathlib import Path  
from concurrent.futures import ThreadPoolExecutor, as_completed  
import time  

# 添加项目路径  
sys.path.append(str(Path(__file__).parent))  

def process_single_style(args_tuple):  
    """处理单个风格描述"""  
    style_id, style_description, model_path, output_dir = args_tuple  
    
    try:  
        from fine_tune_blender_llm import BlenderLLMFineTuner  
        from inference_and_eval import generate_chair_views  
        
        # 加载模型  
        model = BlenderLLMFineTuner(model_path, output_dir)  
        model.model.eval()  
        
        # 创建输出目录  
        style_output_dir = Path(output_dir) / f"style_{style_id}"  
        style_output_dir.mkdir(exist_ok=True)  
        
        # 生成椅子视图  
        generated_views = generate_chair_views(  
            model, style_description, style_output_dir, f"chair_{style_id}"  
        )  
        
        result = {  
            'style_id': style_id,  
            'style_description': style_description,  
            'status': 'success',  
            'generated_views': {k: str(v) for k, v in generated_views.items()} if generated_views else None,  
            'output_dir': str(style_output_dir),  
            'timestamp': time.time()  
        }  
        
        # 保存结果  
        with open(style_output_dir / "generation_result.json", 'w') as f:  
            json.dump(result, f, indent=2)  
        
        return result  
        
    except Exception as e:  
        result = {  
            'style_id': style_id,  
            'style_description': style_description,  
            'status': 'failed',  
            'error': str(e),  
            'timestamp': time.time()  
        }  
        return result  

def main():  
    parser = argparse.ArgumentParser(description='Batch Process Chair Style Generation')  
    parser.add_argument('--model_path', type=str, required=True,  
                       help='Path to fine-tuned model')  
    parser.add_argument('--styles_file', type=str, required=True,  
                       help='JSON file containing style descriptions')  
    parser.add_argument('--output_dir', type=str, default='./batch_results',  
                       help='Output directory for batch results')  
    parser.add_argument('--max_workers', type=int, default=2,  
                       help='Maximum number of parallel workers')  
    parser.add_argument('--start_index', type=int, default=0,  
                       help='Start processing from this index')  
    parser.add_argument('--end_index', type=int, default=None,  
                       help='End processing at this index')  
    
    args = parser.parse_args()  
    
    # 验证输入  
    if not Path(args.model_path).exists():  
        print(f"Error: Model path does not exist: {args.model_path}")  
        sys.exit(1)  
    
    if not Path(args.styles_file).exists():  
        print(f"Error: Styles file does not exist: {args.styles_file}")  
        sys.exit(1)  
    
    # 创建输出目录   
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
    
    # 加载风格描述  
    with open(args.styles_file, 'r', encoding='utf-8') as f:  
        if args.styles_file.endswith('.json'):  
            styles_data = json.load(f)  
        else:  
            # 支持文本格式的风格描述  
            styles_data = parse_style_descriptions(f.read())  
    
    # 处理索引范围  
    if isinstance(styles_data, list):  
        total_styles = len(styles_data)  
        end_idx = args.end_index if args.end_index is not None else total_styles  
        styles_to_process = styles_data[args.start_index:end_idx]  
    else:  
        # 如果是字典格式  
        style_items = list(styles_data.items())  
        total_styles = len(style_items)  
        end_idx = args.end_index if args.end_index is not None else total_styles  
        styles_to_process = style_items[args.start_index:end_idx]  
    
    print(f"Processing {len(styles_to_process)} styles (index {args.start_index} to {end_idx-1})")  
    print(f"Output directory: {args.output_dir}")  
    print(f"Max workers: {args.max_workers}")  
    
    # 准备任务参数  
    tasks = []  
    for i, style_data in enumerate(styles_to_process):  
        if isinstance(style_data, dict):  
            style_id = style_data.get('id', f'style_{args.start_index + i}')  
            style_description = format_style_description(style_data)  
        elif isinstance(style_data, tuple):  
            style_id, style_description = style_data  
        else:  
            style_id = f'style_{args.start_index + i}'  
            style_description = str(style_data)  
        
        tasks.append((style_id, style_description, args.model_path, args.output_dir))  
    
    # 并行处理  
    results = []  
    successful = 0  
    failed = 0  
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:  
        # 提交所有任务  
        future_to_task = {executor.submit(process_single_style, task): task for task in tasks}  
        
        # 收集结果  
        for future in as_completed(future_to_task):  
            task = future_to_task[future]  
            try:  
                result = future.result()  
                results.append(result)  
                
                if result['status'] == 'success':  
                    successful += 1  
                    print(f"✓ Completed: {result['style_id']}")  
                else:  
                    failed += 1  
                    print(f"✗ Failed: {result['style_id']} - {result.get('error', 'Unknown error')}")  
                    
            except Exception as e:  
                failed += 1  
                error_result = {  
                    'style_id': task[0],  
                    'style_description': task[1],  
                    'status': 'failed',  
                    'error': str(e),  
                    'timestamp': time.time()  
                }  
                results.append(error_result)  
                print(f"✗ Exception: {task[0]} - {str(e)}")  
    
    # 保存批量处理结果  
    batch_summary = {  
        'total_processed': len(tasks),  
        'successful': successful,  
        'failed': failed,  
        'success_rate': successful / len(tasks) if tasks else 0,  
        'start_time': time.time(),  
        'results': results  
    }  
    
    with open(Path(args.output_dir) / "batch_summary.json", 'w') as f:  
        json.dump(batch_summary, f, indent=2, default=str)  
    
    print(f"\nBatch processing completed!")  
    print(f"Total: {len(tasks)}, Successful: {successful}, Failed: {failed}")  
    print(f"Success rate: {successful/len(tasks)*100:.1f}%")  
    print(f"Results saved to: {args.output_dir}")  

def parse_style_descriptions(text_content):  
    """解析文本格式的风格描述"""  
    styles = []  
    
    # 按段落分割  
    paragraphs = text_content.strip().split('\n\n')  
    
    for i, paragraph in enumerate(paragraphs):  
        if paragraph.strip():  
            # 尝试解析结构化格式  
            parsed_style = parse_structured_style(paragraph)  
            if parsed_style:  
                parsed_style['id'] = f'chair_{i+1}'  
                styles.append(parsed_style)  
            else:  
                # 作为纯文本处理  
                styles.append({  
                    'id': f'style_{i+1}',  
                    'description': paragraph.strip()  
                })  
    
    return styles  

def parse_structured_style(paragraph):  
    """解析结构化的椅子风格描述"""  
    try:  
        lines = paragraph.strip().split('\n')  
        style_data = {}  
        
        for line in lines:  
            line = line.strip()  
            if ':' in line:  
                key, value = line.split(':', 1)  
                key = key.strip()  
                value = value.strip()  
                
                # 标准化键名  
                key_mapping = {  
                    '传统/古典风格': 'traditional_style',  
                    '现代风格': 'modern_style',   
                    '其他特色风格': 'special_style',  
                    '材质相关描述': 'material',  
                    '功能型椅子': 'functional_type',  
                    '主要功能': 'main_function',  
                    '人体工学符合性': 'ergonomics',  
                    '高度可调节性': 'height_adjustable',  
                    '角度可调节性': 'angle_adjustable',  
                    '折叠性': 'foldable'  
                }  
                
                mapped_key = key_mapping.get(key, key.lower().replace(' ', '_'))  
                
                # 处理null值  
                if value.lower() in ['null', 'none', '无', '']:  
                    continue  
                
                style_data[mapped_key] = value  
        
        return style_data if style_data else None  
        
    except Exception as e:  
        print(f"Error parsing structured style: {e}")  
        return None  

def format_style_description(style_data):  
    """格式化椅子风格描述为自然语言"""  
    if isinstance(style_data, str):  
        return style_data  
    
    if 'description' in style_data:  
        return style_data['description']  
    
    # 构建自然语言描述  
    description_parts = []  
    
    # 风格描述  
    styles = []  
    if style_data.get('traditional_style'):  
        styles.append(f"传统{style_data['traditional_style']}")  
    if style_data.get('modern_style'):  
        styles.append(f"现代{style_data['modern_style']}")  
    if style_data.get('special_style'):  
        styles.append(style_data['special_style'])  
    
    if styles:  
        description_parts.append(f"设计一把{'/'.join(styles)}风格的椅子")  
    else:  
        description_parts.append("设计一把椅子")  
    
    # 材质  
    if style_data.get('material'):  
        description_parts.append(f"材质采用{style_data['material']}")  
    
    # 功能  
    if style_data.get('functional_type'):  
        description_parts.append(f"类型为{style_data['functional_type']}")  
    
    if style_data.get('main_function'):  
        description_parts.append(f"主要用于{style_data['main_function']}")  
    
    # 人体工学和调节性  
    features = []  
    if style_data.get('ergonomics'):  
        if style_data['ergonomics'] in ['高', 'high']:  
            features.append("符合人体工学设计")  
        elif style_data['ergonomics'] in ['中', 'medium']:  
            features.append("基本符合人体工学")  
    
    if style_data.get('height_adjustable') == '有':  
        features.append("高度可调节")  
    
    if style_data.get('angle_adjustable') == '有':  
        features.append("角度可调节")  
    
    if style_data.get('foldable') == '有':  
        features.append("可折叠")  
    
    if features:  
        description_parts.append(f"具备{'/'.join(features)}功能")  
    
    return "，".join(description_parts) + "。"  

if __name__ == "__main__":  
    main()