#!/usr/bin/env python3  
"""  
简化的椅子风格生成模型训练脚本  
"""  

import argparse  
import json  
import os  
from pathlib import Path  
import torch  
from simple_fine_tune import SimpleBlenderLLMFineTuner  

def main():  
    parser = argparse.ArgumentParser(description='Simple Chair Style Generation Model Training')  
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')  
    parser.add_argument('--base_model', type=str, required=True, help='Path to base model')  
    parser.add_argument('--output_dir', type=str, default='./output/simple_model', help='Output directory')  
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')  
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')  
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')  
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')  
    
    args = parser.parse_args()  
    
    print("🪑 Starting Simple Chair Style Generation Model Training...")  
    print(f"📊 Data path: {args.data_path}")  
    print(f"🤖 Base model: {args.base_model}")  
    print(f"�� Output directory: {args.output_dir}")  
    
    # 强制使用GPU 0  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    print(f"🎯 Using device: {device}")  
    
    # 检查文件  
    if not Path(args.data_path).exists():  
        print(f"❌ Data file not found: {args.data_path}")  
        return  
    
    if not Path(args.base_model).exists():  
        print(f"❌ Base model path does not exist: {args.base_model}")  
        return  
    
    try:  
        # 创建fine-tuner  
        print("🔧 Creating fine-tuner...")  
        fine_tuner = SimpleBlenderLLMFineTuner(args.base_model, device=device)  
        
        # 加载模型  
        print("🔄 Loading model and tokenizer...")  
        fine_tuner.load_model()  
        
        # 创建数据集  
        print("📚 Creating dataset...")  
        dataset = fine_tuner.create_dataset(args.data_path, args.max_length)  
        
        if len(dataset) == 0:  
            print("❌ Dataset is empty!")  
            return  
        
        print(f"✅ Dataset loaded with {len(dataset)} samples")  
        
        # 显示样本数据  
        print("�� Sample data preview:")  
        with open(args.data_path, 'r', encoding='utf-8') as f:  
            sample_data = json.load(f)  
            if sample_data:  
                sample = sample_data[0]  
                print(f"   Input: {sample.get('input', '')[:100]}...")  
                print(f"   Output: {sample.get('output', '')[:100]}...")  
        
        # 开始训练  
        print("🏋️ Starting training...")  
        model_path = fine_tuner.train(  
            dataset=dataset,  
            output_dir=args.output_dir,  
            epochs=args.epochs,  
            batch_size=args.batch_size,  
            learning_rate=args.learning_rate,  
            warmup_steps=50,  
            logging_steps=20,  
            save_steps=200,  
            fp16=True,  
            gradient_accumulation_steps=1  
        )  
        
        print(f"🎉 Training completed successfully!")  
        print(f"💾 Model saved to: {model_path}")  
        
        # 简单测试  
        print("🧪 Testing model...")  
        test_prompt = "Generate chair design: modern minimalist"  
        
        try:  
            result = fine_tuner.generate(test_prompt, max_length=200)  
            print(f"📝 Test prompt: {test_prompt}")  
            print(f"   Result: {result[:150]}...")  
        except Exception as e:  
            print(f"⚠️  Test generation failed: {e}")  
        
    except Exception as e:  
        print(f"❌ Training failed: {e}")  
        import traceback  
        traceback.print_exc()  
        return  
    
    print("✅ Training completed successfully!")  

if __name__ == '__main__':  
    main()  
