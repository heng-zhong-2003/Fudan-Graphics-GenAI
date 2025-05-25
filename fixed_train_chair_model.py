"""  
修复后的椅子风格生成模型训练脚本  
"""  

import argparse  
import json  
import os  
from pathlib import Path  
import torch  
from fixed_fine_tune_blender_llm import BlenderLLMFineTuner  

def load_config(config_path):  
    """加载配置文件"""  
    if not config_path or not Path(config_path).exists():  
        return {}  
    
    with open(config_path, 'r') as f:  
        return json.load(f)  

def check_data_structure(data_path):  
    """检查数据结构"""  
    data_path = Path(data_path)  
    print(f"🔍 Checking data structure at: {data_path}")  
    
    if not data_path.exists():  
        print(f"❌ Data path does not exist: {data_path}")  
        return False  
    
    # 查找数据文件  
    json_files = list(data_path.rglob('*.json'))  
    txt_files = list(data_path.rglob('*.txt'))  
    
    print(f"📁 Found {len(json_files)} JSON files and {len(txt_files)} TXT files")  
    
    if json_files:  
        print("📄 Sample JSON files:")  
        for f in json_files[:3]:  
            print(f"   - {f}")  
            
    if txt_files:  
        print("📄 Sample TXT files:")  
        for f in txt_files[:3]:  
            print(f"   - {f}")  
    
    return len(json_files) > 0 or len(txt_files) > 0  

def main():  
    parser = argparse.ArgumentParser(description='Train Chair Style Generation Model (Fixed)')  
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')  
    parser.add_argument('--base_model', type=str, required=True, help='Path to base model')  
    parser.add_argument('--config', type=str, help='Configuration file path')  
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')  
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')  
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')  
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')  
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')  
    
    args = parser.parse_args()  
    
    # 加载配置  
    config = load_config(args.config)  
    
    # 合并配置和命令行参数  
    data_path = args.data_path or config.get('data_path')  
    base_model = args.base_model or config.get('base_model')  
    output_dir = args.output_dir or config.get('output_dir', './output')  
    
    print("🪑 Starting Chair Style Generation Model Training (Fixed)...")  
    print(f"📊 Data path: {data_path}")  
    print(f"🤖 Base model: {base_model}")  
    print(f"📁 Output directory: {output_dir}")  
    print(f"⚙️  Training parameters:")  
    print(f"   - Epochs: {args.epochs}")  
    print(f"   - Batch size: {args.batch_size}")  
    print(f"   - Learning rate: {args.learning_rate}")  
    print(f"   - Max length: {args.max_length}")  
    
    # 检查数据结构  
    if not check_data_structure(data_path):  
        print("❌ No valid data files found. Please check your data directory.")  
        return  
    
    # 检查模型路径  
    if not Path(base_model).exists():  
        print(f"❌ Base model path does not exist: {base_model}")  
        return  
    
    # 检查CUDA可用性  
    if torch.cuda.is_available():  
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")  
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")  
    else:  
        print("⚠️  CUDA not available, using CPU")  
    
    try:  
        # 创建fine-tuner  
        fine_tuner = BlenderLLMFineTuner(base_model)  
        
        # 加载模型  
        print("🔄 Loading model and tokenizer...")  
        fine_tuner.load_model()  
        
        # 创建数据集  
        print("📚 Creating dataset...")  
        dataset = fine_tuner.create_dataset(data_path, args.max_length)  
        
        if len(dataset) == 0:  
            print("❌ Dataset is empty! Please check your data format.")  
            return  
        
        print(f"✅ Dataset loaded with {len(dataset)} samples")  
        
        # 开始训练  
        print("🏋️ Starting training...")  
        model_path = fine_tuner.train(  
            dataset=dataset,  
            output_dir=output_dir,  
            epochs=args.epochs,  
            batch_size=args.batch_size,  
            learning_rate=args.learning_rate,  
            warmup_steps=config.get('training', {}).get('warmup_steps', 100),  
            logging_steps=config.get('training', {}).get('logging_steps', 50),  
            save_steps=config.get('training', {}).get('save_steps', 500),  
            fp16=config.get('training', {}).get('fp16', True),  
            gradient_accumulation_steps=config.get('training', {}).get('gradient_accumulation_steps', 1)  
        )  
        
        print(f"🎉 Training completed successfully!")  
        print(f"💾 Model saved to: {model_path}")  
        
        # 简单测试  
        print("🧪 Testing model with sample prompt...")  
        test_prompt = "Design a modern minimalist chair with clean lines"  
        try:  
            result = fine_tuner.generate(test_prompt, max_length=256)  
            print(f"📝 Test result: {result[:200]}...")  
        except Exception as e:  
            print(f"⚠️  Test generation failed: {e}")  
        
    except Exception as e:  
        print(f"❌ Training failed with error: {e}")  
        import traceback  
        traceback.print_exc()  
        return  
    
    print("✅ All done!")  

if __name__ == '__main__':  
    main()  
