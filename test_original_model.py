#!/usr/bin/env python3  
"""  
测试原始BlenderLLM模型的椅子生成能力  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import os  

def test_original_model():  
    print("🧪 测试原始BlenderLLM模型...")  
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    torch.cuda.empty_cache()  
    
    # 加载原始模型  
    tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
    model = AutoModelForCausalLM.from_pretrained(  
        "../models/BlenderLLM",  
        trust_remote_code=True,  
        torch_dtype=torch.float16,  
        device_map="auto"  
    )  
    
    print("✅ 模型加载成功")  
    
    # 详细测试椅子生成  
    test_prompts = [  
        "Generate chair design: modern minimalist",  
        "Generate chair design: vintage wooden dining chair",  
        "Generate chair design: ergonomic office chair",  
        "Generate chair design: comfortable armchair",  
        "Generate chair design: industrial bar stool",  
        "Create a simple wooden chair",  
        "Design a modern office chair",  
        "Make a comfortable reading chair",  
        "Build a dining room chair",  
        "Create a stylish accent chair"  
    ]  
    
    print(f"\n🎯 测试 {len(test_prompts)} 个椅子设计提示...")  
    
    results = []  
    
    for i, prompt in enumerate(test_prompts):  
        print(f"\n{'='*60}")  
        print(f"测试 {i+1}/{len(test_prompts)}: {prompt}")  
        
        try:  
            inputs = tokenizer(prompt, return_tensors="pt")  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
            
            with torch.no_grad():  
                outputs = model.generate(  
                    **inputs,  
                    max_length=len(inputs['input_ids'][0]) + 500,  
                    temperature=0.7,  
                    do_sample=True,  
                    top_p=0.9,  
                    pad_token_id=tokenizer.eos_token_id,  
                    repetition_penalty=1.1  
                )  
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
            generated = result[len(prompt):].strip()  
            
            print(f"📝 生成内容 ({len(generated)} 字符):")  
            print(f"{generated}")  
            
            # 分析生成内容  
            analysis = analyze_output(generated)  
            print(f"\n📊 内容分析:")  
            print(f"  ✅ 包含Blender代码: {analysis['has_blender_code']}")  
            print(f"  ✅ 包含椅子相关: {analysis['has_chair_content']}")  
            print(f"  ✅ 代码结构合理: {analysis['has_good_structure']}")  
            print(f"  📏 长度适中: {analysis['good_length']}")  
            print(f"  ⭐ 总体评分: {analysis['score']}/5")  
            
            results.append({  
                'prompt': prompt,  
                'generated': generated,  
                'analysis': analysis  
            })  
            
        except Exception as e:  
            print(f"❌ 生成失败: {e}")  
            results.append({  
                'prompt': prompt,  
                'generated': '',  
                'analysis': {'score': 0}  
            })  
    
    # 总结  
    print(f"\n{'='*60}")  
    print("📊 测试总结:")  
    
    total_score = sum(r['analysis'].get('score', 0) for r in results)  
    avg_score = total_score / len(results)  
    
    successful_generations = len([r for r in results if r['generated']])  
    
    print(f"  成功生成: {successful_generations}/{len(results)}")  
    print(f"  平均评分: {avg_score:.2f}/5")  
    
    # 显示最佳结果  
    best_result = max(results, key=lambda x: x['analysis'].get('score', 0))  
    print(f"\n🏆 最佳生成结果:")  
    print(f"  提示: {best_result['prompt']}")  
    print(f"  评分: {best_result['analysis'].get('score', 0)}/5")  
    print(f"  内容: {best_result['generated'][:300]}...")  
    
    return results  

def analyze_output(text):  
    """分析生成内容的质量"""  
    analysis = {  
        'has_blender_code': False,  
        'has_chair_content': False,  
        'has_good_structure': False,  
        'good_length': False,  
        'score': 0  
    }  
    
    text_lower = text.lower()  
    
    # 检查Blender代码  
    blender_keywords = ['bpy.', 'import bpy', 'mesh.', 'object.', 'add_object', 'ops.']  
    if any(keyword in text for keyword in blender_keywords):  
        analysis['has_blender_code'] = True  
        analysis['score'] += 2  
    
    # 检查椅子相关内容  
    chair_keywords = ['chair', 'seat', 'leg', 'backrest', 'armrest', 'cushion']  
    if any(keyword in text_lower for keyword in chair_keywords):  
        analysis['has_chair_content'] = True  
        analysis['score'] += 1  
    
    # 检查代码结构  
    if ('def ' in text or 'bpy.ops.' in text) and 'location' in text:  
        analysis['has_good_structure'] = True  
        analysis['score'] += 1  
    
    # 检查长度  
    if 100 < len(text) < 2000:  
        analysis['good_length'] = True  
        analysis['score'] += 1  
    
    return analysis  

if __name__ == "__main__":  
    test_original_model()  
