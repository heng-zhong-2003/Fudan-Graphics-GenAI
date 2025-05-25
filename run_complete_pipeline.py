# run_complete_pipeline.py  
#!/usr/bin/env python3  
"""  
椅子风格生成完整流水线  
"""  

import os  
import sys  
import argparse  
import subprocess  
from pathlib import Path  
import json  
import time  

def run_command(cmd, description):  
    """运行命令并处理错误"""  
    print(f"\n{'='*60}")  
    print(f"Running: {description}")  
    print(f"Command: {' '.join(cmd)}")  
    print('='*60)  
    
    try:  
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)  
        print("✓ Success!")  
        if result.stdout:  
            print("Output:", result.stdout)  
        return True  
    except subprocess.CalledProcessError as e:  
        print(f"✗ Failed with return code {e.returncode}")  
        if e.stdout:  
            print("Stdout:", e.stdout)  
        if e.stderr:  
            print("Stderr:", e.stderr)  
        return False  

def main():  
    parser = argparse.ArgumentParser(description='Complete Chair Style Generation Pipeline')  
    parser.add_argument('--data_path', type=str, required=True,  
                       help='Path to chair dataset')  
    parser.add_argument('--base_model', type=str, required=True,  
                       help='Path to base BlenderLLM model')  
    parser.add_argument('--styles_file', type=str, required=True,  
                       help='File containing style descriptions')  
    parser.add_argument('--output_dir', type=str, default='./pipeline_output',  
                       help='Output directory for all results')  
    parser.add_argument('--skip_training', action='store_true',  
                       help='Skip training and use existing model')  
    parser.add_argument('--model_path', type=str, default=None,  
                       help='Path to pre-trained model (if skipping training)')  
    parser.add_argument('--epochs', type=int, default=3,  
                       help='Training epochs')  
    parser.add_argument('--batch_size', type=int, default=2,  
                       help='Training batch size')  
    parser.add_argument('--max_workers', type=int, default=2,  
                       help='Max workers for batch processing')  
    parser.add_argument('--num_test_samples', type=int, default=10,  
                       help='Number of test samples')  
    
    args = parser.parse_args()  
    
    # 创建输出目录  
    output_dir = Path(args.output_dir)  
    output_dir.mkdir(parents=True, exist_ok=True)  
    
    # 记录运行配置  
    config = {  
        'start_time': time.time(),  
        'args': vars(args),  
        'steps': []  
    }  
    
    print("🚀 Starting Complete Chair Style Generation Pipeline")  
    print(f"Output directory: {output_dir}")  
    
    # Step 1: 预处理风格描述  
    print(f"\n📝 Step 1: Preprocessing style descriptions")  
    processed_styles_file = output_dir / "processed_styles.json"  
    
    preprocess_cmd = [  
        sys.executable, "preprocess_styles.py",  
        "--input_file", args.styles_file,  
        "--output_file", str(processed_styles_file)  
    ]  
    
    if run_command(preprocess_cmd, "Preprocessing style descriptions"):  
        config['steps'].append({'step': 1, 'status': 'success', 'output': str(processed_styles_file)})  
    else:  
        config['steps'].append({'step': 1, 'status': 'failed'})  
        print("❌ Pipeline failed at step 1")  
        return  
    
    # Step 2: 训练或加载模型  
    if args.skip_training and args.model_path:  
        print(f"\n🤖 Step 2: Using existing model")  
        model_path = args.model_path  
        config['steps'].append({'step': 2, 'status': 'skipped', 'model_path': model_path})  
    else:  
        print(f"\n🏋️ Step 2: Training model")  
        model_path = output_dir / "fine_tuned_model"  
        
        train_cmd = [  
            sys.executable, "train_chair_model.py",  
            "--data_path", args.data_path,  
            "--base_model", args.base_model,  
            "--output_dir", str(model_path),  
            "--epochs", str(args.epochs),  
            "--batch_size", str(args.batch_size)  
        ]  
        
        if run_command(train_cmd, "Training chair style generation model"):  
            config['steps'].append({'step': 2, 'status': 'success', 'model_path': str(model_path)})  
        else:  
            config['steps'].append({'step': 2, 'status': 'failed'})  
            print("❌ Pipeline failed at step 2")  
            return  
    
    # Step 3: 批量生成  
    print(f"\n🎨 Step 3: Batch generation")  
    batch_output_dir = output_dir / "batch_generation"  
    
    batch_cmd = [  
        sys.executable, "batch_process.py",  
        "--model_path", str(model_path),  
        "--styles_file", str(processed_styles_file),  
        "--output_dir", str(batch_output_dir),  
        "--max_workers", str(args.max_workers),  
        "--end_index", str(args.num_test_samples)  
    ]  
    
    if run_command(batch_cmd, "Batch generation"):  
        config['steps'].append({'step': 3, 'status': 'success', 'output': str(batch_output_dir)})  
    else:  
        config['steps'].append({'step': 3, 'status': 'failed'})  
        print("❌ Pipeline failed at step 3")  
        return  
    
    # Step 4: 评估结果  
    print(f"\n📊 Step 4: Evaluating results")  
    eval_output_dir = output_dir / "evaluation"  
    
    eval_cmd = [  
        sys.executable, "evaluate_results.py",  
        "--batch_results_dir", str(batch_output_dir),  
        "--output_dir", str(eval_output_dir)  
    ]  
    
    if run_command(eval_cmd, "Evaluating generation results"):  
        config['steps'].append({'step': 4, 'status': 'success', 'output': str(eval_output_dir)})  
    else:  
        config['steps'].append({'step': 4, 'status': 'failed'})  
        print("⚠️ Evaluation failed, but generation completed")  
    
    # 保存运行配置  
    config['end_time'] = time.time()  
    config['duration'] = config['end_time'] - config['start_time']  
    
    with open(output_dir / "pipeline_config.json", 'w') as f:  
        json.dump(config, f, indent=2, default=str)  
    
    # 生成报告  
    generate_final_report(output_dir, config)  
    
    print(f"\n🎉 Pipeline completed successfully!")  
    print(f"Total duration: {config['duration']:.2f} seconds")  
    print(f"Results saved to: {output_dir}")  

def generate_final_report(output_dir, config):  
    """生成最终报告"""  
    report_file = output_dir / "pipeline_report.md"  
    
    with open(report_file, 'w', encoding='utf-8') as f:  
        f.write("# Chair Style Generation Pipeline Report\n\n")  
        f.write(f"**Generated on:** {time.ctime()}\n")  
        f.write(f"**Duration:** {config['duration']:.2f} seconds\n\n")  
        
        f.write("## Configuration\n\n")  
        for key, value in config['args'].items():  
            f.write(f"- **{key}:** {value}\n")  
        
        f.write("\n## Pipeline Steps\n\n")  
        for step_info in config['steps']:  
            step_num = step_info['step']  
            status = step_info['status']  
            
            if status == 'success':  
                icon = "✅"  
            elif status == 'failed':  
                icon = "❌"  
            else:  
                icon = "⏭️"  
            
            f.write(f"{icon} **Step {step_num}:** {status.title()}\n")  
            
            if 'output' in step_info:  
                f.write(f"   - Output: `{step_info['output']}`\n")  
            if 'model_path' in step_info:  
                f.write(f"   - Model: `{step_info['model_path']}`\n")  
        
        f.write("\n## Output Structure\n\n")  
        f.write("```\n")  
        f.write(f"{output_dir}/\n")  
        f.write("├── processed_styles.json       # 预处理后的风格描述\n")  
        f.write("├── fine_tuned_model/          # 微调后的模型\n")  
        f.write("├── batch_generation/          # 批量生成结果\n")  
        f.write("├── evaluation/                # 评估结果\n")  
        f.write("├── pipeline_config.json       # 运行配置\n")  
        f.write("└── pipeline_report.md         # 本报告\n")  
        f.write("```\n")  
        
        # 尝试添加批量生成统计  
        try:  
            batch_summary_file = output_dir / "batch_generation" / "batch_summary.json"  
            if batch_summary_file.exists():  
                with open(batch_summary_file) as bf:  
                    batch_data = json.load(bf)  
                
                f.write("\n## Generation Statistics\n\n")  
                f.write(f"- **Total processed:** {batch_data.get('total_processed', 0)}\n")  
                f.write(f"- **Successful:** {batch_data.get('successful', 0)}\n")   
                f.write(f"- **Failed:** {batch_data.get('failed', 0)}\n")  
                f.write(f"- **Success rate:** {batch_data.get('success_rate', 0)*100:.1f}%\n")  
        except:  
            pass  
    
    print(f"📄 Report generated: {report_file}")  

if __name__ == "__main__":  
    main()