# train.py  
import argparse  
from fine_tune_blender_llm import BlenderLLMFineTuner  
from evaluation_metrics import ChairGenerationEvaluator  

def main():  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--data_path', type=str, required=True,   
                       help='Path to the chair dataset')  
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2-7B',   
                       help='Base model name')  
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_model',  
                       help='Output directory for fine-tuned model')  
    parser.add_argument('--epochs', type=int, default=3,  
                       help='Number of training epochs')  
    parser.add_argument('--batch_size', type=int, default=4,  
                       help='Batch size for training')  
    parser.add_argument('--learning_rate', type=float, default=2e-5,  
                       help='Learning rate')  
    
    args = parser.parse_args()  
    
    # 初始化微调器  
    fine_tuner = BlenderLLMFineTuner(args.base_model, args.output_dir)  
    
    # 准备数据  
    dataset = fine_tuner.prepare_data(args.data_path)  
    print(f"Loaded {len(dataset)} training samples")  
    
    # 开始训练  
    fine_tuner.train(  
        dataset,   
        epochs=args.epochs,  
        batch_size=args.batch_size,  
        learning_rate=args.learning_rate  
    )  
    
    print("Training completed!")  

if __name__ == "__main__":  
    main()