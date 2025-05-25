# inference_and_eval.py  
import argparse  
from fine_tune_blender_llm import BlenderLLMFineTuner  
from evaluation_metrics import ChairGenerationEvaluator  
import subprocess  
import os  
import sys  
from pathlib import Path  

def generate_chair_views(model, style_description, output_dir, chair_id):  
    """生成椅子三视图"""  
    # 1. 使用微调后的模型生成Blender脚本  
    blender_script = model.generate_blender_script(style_description)  
    
    # 提取Python代码部分  
    script_start = blender_script.find("```python") + len("```python")  
    script_end = blender_script.find("```", script_start)  
    if script_end == -1:  
        script_end = len(blender_script)  
    
    clean_script = blender_script[script_start:script_end].strip()  
    
    # 2. 保存Blender脚本  
    script_path = Path(output_dir) / f"{chair_id}_generated_script.py"  
    with open(script_path, 'w', encoding='utf-8') as f:  
        f.write(clean_script)  
    
    # 3. 使用Blender执行脚本生成3D模型  
    blender_executable = "/home/saisai/graph/BlenderModel/blender-4.4.3-linux-x64/blender"  
    blend_output = Path(output_dir) / f"{chair_id}_model.blend"  
    
    blender_cmd = [  
        blender_executable,  
        "--background",  
        "--python", str(script_path),  
        "--",   
        str(blend_output)  
    ]  
    
    try:  
        subprocess.run(blender_cmd, check=True, capture_output=True, text=True)  
        print(f"3D model generated: {blend_output}")  
    except subprocess.CalledProcessError as e:  
        print(f"Error generating 3D model: {e}")  
        return None  
    
    # 4. 使用data_transfer.py生成三视图  
    sys.path.append("/home/saisai/graph/Fudan-Graphics-GenAI/script_for_data")  
    from data_transfer import process_cad_script  
    
    svg_output_dir = Path(output_dir) / "views"  
    svg_output_dir.mkdir(exist_ok=True)  
    
    try:  
        process_cad_script(clean_script, chair_id, str(svg_output_dir), 0)  
        
        # 返回生成的三视图文件路径  
        svg_files = {  
            'front': svg_output_dir / f"{chair_id}_front.svg",  
            'side': svg_output_dir / f"{chair_id}_side.svg",   
            'top': svg_output_dir / f"{chair_id}_top.svg"  
        }  
        
        return svg_files  
    except Exception as e:  
        print(f"Error generating views: {e}")  
        return None  

def evaluate_single_chair(evaluator, generated_views, reference_views, style_description):  
    """评估单个椅子的生成结果"""  
    if not generated_views or not all(Path(v).exists() for v in generated_views.values()):  
        return {"error": "Generated views not found"}  
    
    if not reference_views or not all(Path(v).exists() for v in reference_views.values()):  
        return {"error": "Reference views not found"}  
    
    results = evaluator.comprehensive_evaluation(  
        generated_views, reference_views, style_description  
    )  
    
    return results  

def main():  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model_path', type=str, required=True,  
                       help='Path to fine-tuned model')  
    parser.add_argument('--test_data_path', type=str, required=True,  
                       help='Path to test data')  
    parser.add_argument('--output_dir', type=str, default='./inference_results',  
                       help='Output directory for generated results')  
    parser.add_argument('--eval_only', action='store_true',  
                       help='Only run evaluation on existing results')  
    
    args = parser.parse_args()  
    
    output_dir = Path(args.output_dir)  
    output_dir.mkdir(exist_ok=True)  
    
    # 初始化模型和评估器  
    if not args.eval_only:  
        model = BlenderLLMFineTuner(args.model_path, args.output_dir)  
        model.model.eval()  
    
    evaluator = ChairGenerationEvaluator()  
    
    # 加载测试数据  
    test_data_path = Path(args.test_data_path)  
    test_samples = []  
    
    for chair_dir in test_data_path.iterdir():  
        if chair_dir.is_dir():  
            tags_file = chair_dir / "tags.txt"  
            if tags_file.exists():  
                with open(tags_file, 'r', encoding='utf-8') as f:  
                    tags_content = f.read()  
                
                # 获取参考三视图  
                reference_views = {  
                    'front': chair_dir / f"{chair_dir.name}_front.svg",  
                    'side': chair_dir / f"{chair_dir.name}_side.svg",  
                    'top': chair_dir / f"{chair_dir.name}_top.svg"  
                }  
                
                if all(f.exists() for f in reference_views.values()):  
                    test_samples.append({  
                        'chair_id': chair_dir.name,  
                        'style_description': tags_content,  
                        'reference_views': reference_views  
                    })  
    
    print(f"Found {len(test_samples)} test samples")  
    
    # 处理每个测试样本  
    all_results = []  
    
    for i, sample in enumerate(test_samples):  
        print(f"\nProcessing sample {i+1}/{len(test_samples)}: {sample['chair_id']}")  
        
        sample_output_dir = output_dir / sample['chair_id']  
        sample_output_dir.mkdir(exist_ok=True)  
        
        if not args.eval_only:  
            # 生成新的三视图  
            generated_views = generate_chair_views(  
                model,   
                sample['style_description'],  
                sample_output_dir,  
                sample['chair_id']  
            )  
        else:  
            # 使用已存在的结果  
            views_dir = sample_output_dir / "views"  
            if views_dir.exists():  
                generated_views = {  
                    'front': views_dir / f"{sample['chair_id']}_front.svg",  
                    'side': views_dir / f"{sample['chair_id']}_side.svg",  
                    'top': views_dir / f"{sample['chair_id']}_top.svg"  
                }  
            else:  
                generated_views = None  
        
        # 评估结果  
        eval_results = evaluate_single_chair(  
            evaluator,  
            generated_views,  
            sample['reference_views'],  
            sample['style_description']  
        )  
        
        eval_results['chair_id'] = sample['chair_id']  
        all_results.append(eval_results)  
        
        print(f"Evaluation results for {sample['chair_id']}: {eval_results}")  
        
        # 保存单个样本的结果  
        import json  
        with open(sample_output_dir / "evaluation_results.json", 'w') as f:  
            json.dump(eval_results, f, indent=2, default=str)  
    
    # 计算总体统计  
    compute_overall_statistics(all_results, output_dir)  

def compute_overall_statistics(all_results, output_dir):  
    """计算总体统计结果"""  
    import json  
    import numpy as np  
    
    # 过滤掉错误的结果  
    valid_results = [r for r in all_results if 'error' not in r]  
    
    if not valid_results:  
        print("No valid results found!")  
        return  
    
    statistics = {  
        'total_samples': len(all_results),  
        'valid_samples': len(valid_results),  
        'success_rate': len(valid_results) / len(all_results)  
    }  
    
    # 计算各个指标的平均值和标准差  
    metrics = ['overall_score', 'style_consistency', 'geometric_accuracy']  
    
    for metric in metrics:  
        values = []  
        for result in valid_results:  
            if metric in result and isinstance(result[metric], (int, float)):  
                values.append(result[metric])  
        
        if values:  
            statistics[f'{metric}_mean'] = np.mean(values)  
            statistics[f'{metric}_std'] = np.std(values)  
            statistics[f'{metric}_min'] = np.min(values)  
            statistics[f'{metric}_max'] = np.max(values)  
    
    # 计算视觉相似性指标  
    view_metrics = ['front_visual_similarity', 'side_visual_similarity', 'top_visual_similarity']  
    for view_metric in view_metrics:  
        ssim_values = []  
        mse_values = []  
        psnr_values = []  
        
        for result in valid_results:  
            if view_metric in result and isinstance(result[view_metric], dict):  
                sim_data = result[view_metric]  
                if 'ssim' in sim_data:  
                    ssim_values.append(sim_data['ssim'])  
                if 'mse' in sim_data:  
                    mse_values.append(sim_data['mse'])  
                if 'psnr' in sim_data:  
                    psnr_values.append(sim_data['psnr'])  
        
        if ssim_values:  
            statistics[f'{view_metric}_ssim_mean'] = np.mean(ssim_values)  
            statistics[f'{view_metric}_ssim_std'] = np.std(ssim_values)  
        
        if mse_values:  
            statistics[f'{view_metric}_mse_mean'] = np.mean(mse_values)  
            statistics[f'{view_metric}_mse_std'] = np.std(mse_values)  
        
        if psnr_values:  
            statistics[f'{view_metric}_psnr_mean'] = np.mean(psnr_values)  
            statistics[f'{view_metric}_psnr_std'] = np.std(psnr_values)  
    
    # 保存统计结果  
    with open(output_dir / "overall_statistics.json", 'w') as f:  
        json.dump(statistics, f, indent=2, default=str)  
    
    # 保存所有详细结果  
    with open(output_dir / "detailed_results.json", 'w') as f:  
        json.dump(all_results, f, indent=2, default=str)  
    
    print("\n" + "="*50)  
    print("OVERALL EVALUATION STATISTICS")  
    print("="*50)  
    print(f"Total samples: {statistics['total_samples']}")  
    print(f"Valid samples: {statistics['valid_samples']}")  
    print(f"Success rate: {statistics['success_rate']:.2%}")  
    
    if 'overall_score_mean' in statistics:  
        print(f"\nOverall Score: {statistics['overall_score_mean']:.4f} ± {statistics['overall_score_std']:.4f}")  
    
    if 'style_consistency_mean' in statistics:  
        print(f"Style Consistency: {statistics['style_consistency_mean']:.4f} ± {statistics['style_consistency_std']:.4f}")  
    
    if 'geometric_accuracy_mean' in statistics:  
        print(f"Geometric Accuracy: {statistics['geometric_accuracy_mean']:.4f} ± {statistics['geometric_accuracy_std']:.4f}")  
    
    print("\nDetailed results saved to:", output_dir / "detailed_results.json")  
    print("Statistics saved to:", output_dir / "overall_statistics.json")  

if __name__ == "__main__":  
    main()