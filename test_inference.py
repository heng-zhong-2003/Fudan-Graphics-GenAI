# test_inference.py  
#!/usr/bin/env python3  
"""  
椅子风格生成模型推理测试脚本  
"""  

import os  
import sys  
import argparse  
from pathlib import Path  

# 添加项目路径  
sys.path.append(str(Path(__file__).parent))  

def main():  
    parser = argparse.ArgumentParser(description='Test Chair Style Generation Model Inference')  
    parser.add_argument('--model_path', type=str, required=True,  
                       help='Path to fine-tuned model')  
    parser.add_argument('--test_data_path', type=str,  
                       default='/home/saisai/graph/Fudan-Graphics-GenAI/data_grouped',  
                       help='Path to test data')  
    parser.add_argument('--output_dir', type=str, default='./inference_test_results',  
                       help='Output directory for test results')  
    parser.add_argument('--num_samples', type=int, default=5,  
                       help='Number of samples to test')  
    parser.add_argument('--eval_only', action='store_true',  
                       help='Only run evaluation on existing results')  
    parser.add_argument('--style_description', type=str, default=None,  
                       help='Custom style description for testing')  
    
    args = parser.parse_args()  
    
    # 验证路径  
    if not args.eval_only and not Path(args.model_path).exists():  
        print(f"Error: Model path does not exist: {args.model_path}")  
        sys.exit(1)  
    
    if not Path(args.test_data_path).exists():  
        print(f"Error: Test data path does not exist: {args.test_data_path}")  
        sys.exit(1)  
    
    # 创建输出目录  
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
    
    print("Starting Chair Style Generation Model Testing...")  
    print(f"Model path: {args.model_path}")  
    print(f"Test data path: {args.test_data_path}")  
    print(f"Output directory: {args.output_dir}")  
    print(f"Number of samples: {args.num_samples}")  
    
    try:  
        from inference_and_eval import main as run_inference  
        
        # 构建参数  
        sys.argv = [  
            'inference_and_eval.py',  
            '--model_path', args.model_path,  
            '--test_data_path', args.test_data_path,  
            '--output_dir', args.output_dir  
        ]  
        
        if args.eval_only:  
            sys.argv.append('--eval_only')  
        
        # 运行推理和评估  
        run_inference()  
        
        print("Testing completed successfully!")  
        print(f"Results saved to: {args.output_dir}")  
        
    except Exception as e:  
        print(f"Testing failed with error: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)  

if __name__ == "__main__":  
    main()