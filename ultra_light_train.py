#!/usr/bin/env python3  
"""  
超轻量级椅子风格生成模型训练脚本  
"""  

import argparse  
import json  
import os  
from pathlib import Path  
import torch  
import gc  
from simple_fine_tune_v2 import MemoryOptimizedBlenderLLMFineTuner  

def clear_memory():  
    """清理GPU内存"""  
    gc.collect()  
    torch.cuda.empty_cache()  
    if torch.cuda.is_available():  
        free, total = torch.cuda.mem_get_info()  
        print(f"🧹 GPU memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")  

def main():  
    parser = argparse.ArgumentParser(description='Ultra Light Chair Training')  
    parser.add_argument('--data_path', type=str, required=True)  
    parser.add_argument('--base_model', type=str, required=True)  
    parser.add_argument('--output_dir', type=str, default='./output/ultra_light_model')  
    parser.add_argument('--epochs', type=int, default=1)  
    parser.add_argument('--batch_size', type=int, default=1)  
    parser.add_argument('--learning_rate', type=float, default=5e-6)  # 更小的学习率  
    parser.add_argument('--max_length', type=int, default=128)  # 大幅减少序列长度  
    
    args = parser.parse_args()  
    
    print("🪑 Starting Ultra Light Chair Training...")  
    print(f"📊 Data: {args.data_path}")  
    print(f"🤖 Model: {args.base_model}")  
    print(f"📁 Output: {args.output_dir}")  
    print(f"⚡ Max length: {args.max_length} (ultra light)")  
    print(f"📚 Learning rate: {args.learning_rate}")  
    
    # 设置内存优化环境变量  
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 更好的错误追踪  
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    print(f"🎯 Device: {device}")  
    
    clear_memory()  
    
    if not Path(args.data_path).exists():  
        print(f"❌ Data file not found: {args.data_path}")  
        return  
    
    if not Path(args.base_model).exists():  
        print(f"❌ Model path not found: {args.base_model}")  
        return  
    
    try:  
        print("�� Creating ultra light fine-tuner...")  
        fine_tuner = MemoryOptimizedBlenderLLMFineTuner(args.base_model, device=device)  
        
        print("🔄 Loading model...")  
        fine_tuner.load_model()  
        clear_memory()  
        
        print("📚 Creating mini dataset...")  
        dataset = fine_tuner.create_dataset(args.data_path, args.max_length)  
        
        if len(dataset) == 0:  
            print("❌ Dataset is empty!")  
            return  
        
        print(f"✅ Dataset: {len(dataset)} samples")  
        clear_memory()  
        
        print("🏋️ Starting ultra light training...")  
        model_path = fine_tuner.train(  
            dataset=dataset,  
            output_dir=args.output_dir,  
            epochs=args.epochs,  
            batch_size=args.batch_size,  
            learning_rate=args.learning_rate,  
            warmup_steps=10,  # 最少的warmup  
            logging_steps=100,  
            save_steps=500,  
            fp16=True,  
            gradient_accumulation_steps=8,  # 大梯度累积  
            max_grad_norm=0.5,  # 更严格的梯度裁剪  
        )  
        
        print(f"🎉 Training completed!")  
        print(f"💾 Model saved to: {model_path}")  
        
        clear_memory()  
        
        print("🧪 Quick test...")  
        try:  
            result = fine_tuner.generate("Design a simple chair:", max_length=100)  
            print(f"📝 Test result: {result[:80]}...")  
        except Exception as e:  
            print(f"⚠️  Test failed: {e}")  
        
    except Exception as e:  
        print(f"❌ Training failed: {e}")  
        clear_memory()  
        import traceback  
        traceback.print_exc()  
        return  
    
    clear_memory()  
    print("✅ All done!")  

if __name__ == '__main__':  
    main()  
