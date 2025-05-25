# quick_start.py  
#!/usr/bin/env python3  
"""  
椅子风格生成快速启动脚本  
"""  

import os  
import sys  
import argparse  
from pathlib import Path  

def main():  
    parser = argparse.ArgumentParser(description='Quick Start Chair Style Generation')  
    parser.add_argument('--mode', choices=['train', 'inference', 'full'], default='full',  
                       help='Running mode: train only, inference only, or full pipeline')  
    parser.add_argument('--config', type=str, default='config/default.json',  
                       help='Configuration file path')  
    
    args = parser.parse_args()  
    
    print("🪑 Chair Style Generation - Quick Start")  
    print(f"Mode: {args.mode}")  
    
    # 默认配置  
    default_config = {  
        "data_path": "/home/saisai/graph/Fudan-Graphics-GenAI/data_grouped",  
        "base_model": "/home/saisai/graph/models/BlenderLLM",   
        "styles_file": "examples/chair_styles.txt",  
        "output_dir": "./quick_start_output",  
        "epochs": 3,  
        "batch_size": 2,  
        "max_workers": 1,  
        "num_test_samples": 5  
    }  
    
    # 加载配置文件（如果存在）  
    config_file = Path(args.config)  
    if config_file.exists():  
        import json  
        with open(config_file) as f:  
            user_config = json.load(f)  
        default_config.update(user_config)  
        print(f"✓ Loaded config from: {config_file}")  
    else:  
        print(f"⚠ Config file not found: {config_file}, using defaults")
        # 验证必要的路径  
    missing_paths = []  
    if not Path(default_config["data_path"]).exists():  
        missing_paths.append(f"Data path: {default_config['data_path']}")  
    if not Path(default_config["base_model"]).exists():  
        missing_paths.append(f"Base model: {default_config['base_model']}")  
    if not Path(default_config["styles_file"]).exists():  
        missing_paths.append(f"Styles file: {default_config['styles_file']}")  
    
    if missing_paths:  
        print("❌ Missing required paths:")  
        for path in missing_paths:  
            print(f"   - {path}")  
        print("\nPlease update the configuration or create missing files.")  
        return  
    
    print("✓ All required paths verified")  
    
    # 根据模式运行相应的命令  
    if args.mode == 'train':  
        run_training_only(default_config)  
    elif args.mode == 'inference':  
        run_inference_only(default_config)  
    else:  
        run_full_pipeline(default_config)  

def run_training_only(config):  
    """只运行训练"""  
    print("\n🏋️ Running training only...")  
    
    cmd = [  
        sys.executable, "train_chair_model.py",  
        "--data_path", config["data_path"],  
        "--base_model", config["base_model"],  
        "--output_dir", str(Path(config["output_dir"]) / "trained_model"),  
        "--epochs", str(config["epochs"]),  
        "--batch_size", str(config["batch_size"])  
    ]  
    
    print(f"Command: {' '.join(cmd)}")  
    os.system(' '.join(cmd))  

def run_inference_only(config):  
    """只运行推理"""  
    print("\n🎨 Running inference only...")  
    
    # 查找已训练的模型  
    model_path = find_trained_model(config)  
    if not model_path:  
        print("❌ No trained model found. Please run training first.")  
        return  
    
    # 预处理风格描述  
    processed_styles = Path(config["output_dir"]) / "processed_styles.json"  
    preprocess_cmd = [  
        sys.executable, "preprocess_styles.py",  
        "--input_file", config["styles_file"],  
        "--output_file", str(processed_styles)  
    ]  
    
    print("📝 Preprocessing styles...")  
    os.system(' '.join(preprocess_cmd))  
    
    # 批量生成  
    batch_cmd = [  
        sys.executable, "batch_process.py",  
        "--model_path", str(model_path),  
        "--styles_file", str(processed_styles),  
        "--output_dir", str(Path(config["output_dir"]) / "generation_results"),  
        "--max_workers", str(config["max_workers"]),  
        "--end_index", str(config["num_test_samples"])  
    ]  
    
    print("🎨 Starting batch generation...")  
    os.system(' '.join(batch_cmd))  

def run_full_pipeline(config):  
    """运行完整流水线"""  
    print("\n🚀 Running full pipeline...")  
    
    cmd = [  
        sys.executable, "run_complete_pipeline.py",  
        "--data_path", config["data_path"],  
        "--base_model", config["base_model"],  
        "--styles_file", config["styles_file"],  
        "--output_dir", config["output_dir"],  
        "--epochs", str(config["epochs"]),  
        "--batch_size", str(config["batch_size"]),  
        "--max_workers", str(config["max_workers"]),  
        "--num_test_samples", str(config["num_test_samples"])  
    ]  
    
    print(f"Command: {' '.join(cmd)}")  
    os.system(' '.join(cmd))  

def find_trained_model(config):  
    """查找已训练的模型"""  
    possible_paths = [  
        Path(config["output_dir"]) / "trained_model",  
        Path(config["output_dir"]) / "fine_tuned_model",   
        Path("./trained_model"),  
        Path("./fine_tuned_model"),  
        Path("./models/chair_model")  
    ]  
    
    for path in possible_paths:  
        if path.exists() and (path / "config.json").exists():  
            print(f"✓ Found trained model: {path}")  
            return path  
    
    return None  

if __name__ == "__main__":  
    main() 