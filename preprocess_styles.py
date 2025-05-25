# preprocess_styles.py  
#!/usr/bin/env python3  
"""  
椅子风格描述预处理工具  
"""  

import argparse  
import json  
from pathlib import Path  
import re  

def main():  
    parser = argparse.ArgumentParser(description='Preprocess Chair Style Descriptions')  
    parser.add_argument('--input_file', type=str, required=True,  
                       help='Input file containing style descriptions')  
    parser.add_argument('--output_file', type=str, required=True,  
                       help='Output JSON file')  
    parser.add_argument('--format', type=str, choices=['structured', 'text'], default='auto',  
                       help='Input format type')  
    
    args = parser.parse_args()  
    
    if not Path(args.input_file).exists():  
        print(f"Error: Input file does not exist: {args.input_file}")  
        return  
    
    with open(args.input_file, 'r', encoding='utf-8') as f:  
        content = f.read()  
    
    # 自动检测格式  
    if args.format == 'auto':  
        if '传统/古典风格:' in content or 'traditional_style:' in content:  
            format_type = 'structured'  
        else:  
            format_type = 'text'  
    else:  
        format_type = args.format  
    
    print(f"Detected format: {format_type}")  
    
    if format_type == 'structured':  
        processed_styles = process_structured_format(content)  
    else:  
        processed_styles = process_text_format(content)  
    
    # 保存结果  
    with open(args.output_file, 'w', encoding='utf-8') as f:  
        json.dump(processed_styles, f, indent=2, ensure_ascii=False)  
    
    print(f"Processed {len(processed_styles)} style descriptions")  
    print(f"Output saved to: {args.output_file}")  

def process_structured_format(content):  
    """处理结构化格式的风格描述"""  
    styles = []  
    
    # 按空行分割不同的椅子描述  
    chair_descriptions = content.strip().split('\n\n')  
    
    for i, desc in enumerate(chair_descriptions):  
        if desc.strip():  
            chair_data = parse_single_chair_description(desc, i+1)  
            if chair_data:  
                styles.append(chair_data)  
    
    return styles  

def process_text_format(content):  
    """处理纯文本格式的风格描述"""  
    styles = []  
    
    # 简单按行或段落分割  
    descriptions = content.strip().split('\n')  
    descriptions = [d.strip() for d in descriptions if d.strip()]  
    
    for i, desc in enumerate(descriptions):  
        styles.append({  
            'id': f'chair_{i+1}',  
            'raw_description': desc,  
            'formatted_description': desc,  
            'metadata': {}  
        })  
    
    return styles  

def parse_single_chair_description(desc_text, chair_id):  
    """解析单个椅子的结构化描述"""  
    try:  
        lines = desc_text.strip().split('\n')  
        metadata = {}  
        
        for line in lines:  
            line = line.strip()  
            if ':' in line:  
                key, value = line.split(':', 1)  
                key = key.strip()  
                value = value.strip()  
                
                # 跳过null值  
                if value.lower() in ['null', 'none', '无', '']:  
                    continue  
                
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
                
                mapped_key = key_mapping.get(key, key.lower().replace(' ', '_').replace('/', '_'))  
                metadata[mapped_key] = value  
        
        # 生成格式化描述  
        formatted_desc = generate_formatted_description(metadata)  
        
        return {  
            'id': f'chair_{chair_id}',  
            'raw_description': desc_text,  
            'formatted_description': formatted_desc,  
            'metadata': metadata  
        }  
        
    except Exception as e:  
        print(f"Error parsing chair description {chair_id}: {e}")  
        return None  

def generate_formatted_description(metadata):  
    """根据元数据生成格式化的自然语言描述"""  
    if not metadata:  
        return "设计一把椅子。"  
    
    description_parts = []  
    
    # 基础描述  
    base_styles = []  
    if metadata.get('traditional_style'):  
        base_styles.append(f"传统{metadata['traditional_style']}")  
    if metadata.get('modern_style'):  
        base_styles.append(f"现代{metadata['modern_style']}")  
    if metadata.get('special_style'):  
        base_styles.append(metadata['special_style'])  
    
    if base_styles:  
        description_parts.append(f"设计一把{'/'.join(base_styles)}风格的椅子")  
    else:  
        description_parts.append("设计一把椅子")  
    
    # 材质描述  
    if metadata.get('material'):  
        material_desc = metadata['material']  
        # 处理括号内的英文  
        material_desc = re.sub(r'\s*$[^)]*$', '', material_desc)  
        description_parts.append(f"采用{material_desc}材质")  
    
    # 功能描述  
    if metadata.get('functional_type'):  
        func_type = metadata['functional_type']  
        func_type = re.sub(r'\s*$[^)]*$', '', func_type)  
        description_parts.append(f"属于{func_type}类型")  
    
    if metadata.get('main_function'):  
        description_parts.append(f"主要用于{metadata['main_function']}")  
    
        # 特殊功能  
    special_features = []  
    
    # 人体工学  
    if metadata.get('ergonomics'):  
        ergonomics = metadata['ergonomics']  
        if ergonomics in ['高', 'high']:  
            special_features.append("高度符合人体工学设计")  
        elif ergonomics in ['中', 'medium']:  
            special_features.append("中等程度符合人体工学")  
        elif ergonomics in ['低', 'low']:  
            special_features.append("基础人体工学设计")  
    
    # 可调节性  
    adjustable_features = []  
    if metadata.get('height_adjustable') == '有':  
        adjustable_features.append("高度可调节")  
    if metadata.get('angle_adjustable') == '有':  
        adjustable_features.append("角度可调节")  
    
    if adjustable_features:  
        special_features.append("具备" + "和".join(adjustable_features) + "功能")  
    
    # 折叠性  
    if metadata.get('foldable') == '有':  
        special_features.append("可折叠")  
    
    if special_features:  
        description_parts.append("，".join(special_features))  
    
    return "，".join(description_parts) + "。"  

if __name__ == "__main__":  
    main()