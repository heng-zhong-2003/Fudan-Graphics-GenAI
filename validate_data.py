#!/usr/bin/env python3  
"""  
数据验证脚本 - 检查和修复数据格式问题  
"""  

import json  
import os  
from pathlib import Path  
import argparse  

def validate_and_fix_data(data_path, output_path=None):  
    """验证并修复数据格式"""  
    data_path = Path(data_path)  
    
    if output_path:  
        output_path = Path(output_path)  
        output_path.mkdir(parents=True, exist_ok=True)  
    
    print(f"🔍 Validating data in: {data_path}")  
    
    all_data = []  
    problematic_files = []  
    
    # 处理JSON文件  
    for json_file in data_path.rglob('*.json'):  
        try:  
            with open(json_file, 'r', encoding='utf-8') as f:  
                data = json.load(f)  
                
                if isinstance(data, list):  
                    for item in data:  
                        fixed_item = fix_data_item(item)  
                        if fixed_item:  
                            all_data.append(fixed_item)  
                else:  
                    fixed_item = fix_data_item(data)  
                    if fixed_item:  
                        all_data.append(fixed_item)  
                        
                print(f"✅ Processed: {json_file}")  
                
        except Exception as e:  
            print(f"❌ Error processing {json_file}: {e}")  
            problematic_files.append(str(json_file))  
    
    # 处理文本文件  
    for txt_file in data_path.rglob('*.txt'):  
        try:  
            with open(txt_file, 'r', encoding='utf-8') as f:  
                lines = f.readlines()  
                for line in lines:  
                    line = line.strip()  
                    if line:  
                        all_data.append({  
                            "input": f"Generate chair design: {line}",  
                            "output": "",  
                            "source_file": str(txt_file)  
                        })  
            print(f"✅ Processed: {txt_file}")  
        except Exception as e:  
            print(f"❌ Error processing {txt_file}: {e}")  
            problematic_files.append(str(txt_file))  
    
    print(f"\n📊 Validation Results:")  
    print(f"   - Total valid samples: {len(all_data)}")  
    print(f"   - Problematic files: {len(problematic_files)}")  
    
    if problematic_files:  
        print(f"   - Failed files: {problematic_files}")  
    
    # 保存修复后的数据  
    if output_path and all_data:  
        output_file = output_path / 'cleaned_data.json'  
        with open(output_file, 'w', encoding='utf-8') as f:  
            json.dump(all_data, f, ensure_ascii=False, indent=2)  
        print(f"💾 Cleaned data saved to: {output_file}")  
        
        # 创建样本文件  
        sample_file = output_path / 'sample_data.json'  
        sample_data = all_data[:5] if len(all_data) >= 5 else all_data  
        with open(sample_file, 'w', encoding='utf-8') as f:  
            json.dump(sample_data, f, ensure_ascii=False, indent=2)  
        print(f"📝 Sample data saved to: {sample_file}")  
    
    return all_data  

def fix_data_item(item):  
    """修复单个数据项"""  
    if isinstance(item, str):  
        return {  
            "input": f"Generate chair design: {item}",  
            "output": "",  
            "text": item  
        }  
    
    if isinstance(item, dict):  
        fixed_item = {}  
        
        # 处理各种可能的字段  
        for key, value in item.items():  
            if isinstance(value, Path):  
                fixed_item[key] = str(value)  
            elif key in ['file_path', 'image_path', 'model_path'] and isinstance(value, str):  
                fixed_item[key] = value  
            elif isinstance(value, (list, dict)):  
                # 递归处理嵌套结构  
                fixed_item[key] = fix_nested_structure(value)  
            else:  
                fixed_item[key] = value  
        
        # 确保有基本的input/output结构  
        if 'input' not in fixed_item and 'output' not in fixed_item:  
            if 'style_description' in fixed_item:  
                fixed_item['input'] = f"Generate chair design: {fixed_item['style_description']}"  
                fixed_item['output'] = fixed_item.get('blender_code', '')  
            elif 'text' in fixed_item:  
                fixed_item['input'] = f"Generate chair design: {fixed_item['text']}"  
                fixed_item['output'] = ""  
        
        return fixed_item  
    
    return None  

def fix_nested_structure(obj):  
    """修复嵌套结构"""  
    if isinstance(obj, list):  
        return [fix_nested_structure(item) for item in obj]  
    elif isinstance(obj, dict):  
        return {k: str(v) if isinstance(v, Path) else fix_nested_structure(v) for k, v in obj.items()}  
    elif isinstance(obj, Path):  
        return str(obj)  
    else:  
        return obj  

def main():  
    parser = argparse.ArgumentParser(description='Validate and fix data format')  
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')  
    parser.add_argument('--output_path', type=str, help='Path to save cleaned data')  
    
    args = parser.parse_args()  
    
    validate_and_fix_data(args.data_path, args.output_path)  

if __name__ == '__main__':  
    main()  