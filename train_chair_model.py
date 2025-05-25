# train_chair_model.py  
#!/usr/bin/env python3  
"""  
椅子风格生成模型训练脚本  
"""  

import os  
import sys  
import argparse  
from pathlib import Path  

# 添加项目路径  
sys.path.append(str(Path(__file__).parent))  

def main():  
    parser = argparse.ArgumentParser(description='Train Chair Style Generation Model')  
    parser.add_argument('--data_path', type=str,   
                       default='/home/saisai/graph/Fudan-Graphics-GenAI/data_grouped',  
                       help='Path to the chair dataset')  
    parser.add_argument('--base_model', type=str,   
                       default='/home/saisai/graph/models/BlenderLLM',  
                       help='Path to base BlenderLLM model')  
    parser.add_argument('--output_dir', type=str,   
                       default='./fine_tuned_chair_model',  
                       help='Output directory for fine-tuned model')  
    parser.add_argument('--epochs', type=int, default=5,  
                       help='Number of training epochs')  
    parser.add_argument('--batch_size', type=int, default=2,  
                       help='Batch size for training')  
    parser.add_argument('--learning_rate', type=float, default=1e-5,  
                       help='Learning rate')  
    parser.add_argument('--max_length', type=int, default=1024,  
                       help='Maximum sequence length')  
    parser.add_argument('--save_steps', type=int, default=100,  
                       help='Save checkpoint every N steps')  
    parser.add_argument('--eval_steps', type=int, default=50,  
                       help='Evaluate every N steps')  
    parser.add_argument('--warmup_steps', type=int, default=100,  
                       help='Number of warmup steps')  
    parser.add_argument('--logging_steps', type=int, default=10,  
                       help='Log every N steps')  
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,  
                       help='Gradient accumulation steps')  
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,  
                       help='Resume training from checkpoint')  
    
    args = parser.parse_args()  
    
    # 验证路径  
    if not Path(args.data_path).exists():  
        print(f"Error: Data path does not exist: {args.data_path}")  
        sys.exit(1)  
    
    if not Path(args.base_model).exists():  
        print(f"Error: Base model path does not exist: {args.base_model}")  
        sys.exit(1)  
    
    # 创建输出目录  
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
    
    print("Starting Chair Style Generation Model Training...")  
    print(f"Data path: {args.data_path}")  
    print(f"Base model: {args.base_model}")  
    print(f"Output directory: {args.output_dir}")  
    print(f"Training parameters:")  
    print(f"  - Epochs: {args.epochs}")  
    print(f"  - Batch size: {args.batch_size}")  
    print(f"  - Learning rate: {args.learning_rate}")  
    print(f"  - Max length: {args.max_length}")  
    
    try:  
        from fine_tune_blender_llm import BlenderLLMFineTuner  
        
        # 初始化微调器  
        fine_tuner = BlenderLLMFineTuner(args.base_model, args.output_dir)  
        
        # 准备数据  
        print("Loading training data...")  
        dataset = fine_tuner.prepare_data(args.data_path)  
        print(f"Loaded {len(dataset)} training samples")  
        
        if len(dataset) == 0:  
            print("Error: No training samples found!")  
            sys.exit(1)  
        
        # 开始训练  
        print("Starting training...")  
        fine_tuner.train(  
            dataset=dataset,  
            epochs=args.epochs,  
            batch_size=args.batch_size,  
            learning_rate=args.learning_rate,  
            max_length=args.max_length,  
            save_steps=args.save_steps,  
            eval_steps=args.eval_steps,  
            warmup_steps=args.warmup_steps,  
            logging_steps=args.logging_steps,  
            gradient_accumulation_steps=args.gradient_accumulation_steps,  
            resume_from_checkpoint=args.resume_from_checkpoint  
        )  
        
        print("Training completed successfully!")  
        print(f"Model saved to: {args.output_dir}")  
        
    except Exception as e:  
        print(f"Training failed with error: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)  

if __name__ == "__main__":  
    main()