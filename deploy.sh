#!/bin/bash  
# deploy.sh - 一键部署脚本  

set -e  

echo "🪑 Chair Style Generation - One-Click Deployment"  
echo "=================================================="  

# 检查系统要求  
check_requirements() {  
    echo "🔍 Checking system requirements..."  
    
    # 检查Python  
    if ! command -v python3 &> /dev/null; then  
        echo "❌ Python 3 not found. Please install Python 3.8+"  
        exit 1  
    fi  
    
    # 检查GPU  
    if command -v nvidia-smi &> /dev/null; then  
        echo "✅ NVIDIA GPU detected"  
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader  
    else  
        echo "⚠️  No NVIDIA GPU detected. CPU-only mode will be used."  
    fi  
    
    # 检查磁盘空间  
    available_space=$(df . | awk 'NR==2{print $4}')  
    required_space=10485760  # 10GB in KB  
    
    if [ "$available_space" -lt "$required_space" ]; then  
        echo "❌ Insufficient disk space. At least 10GB required."  
        exit 1  
    fi  
    
    echo "✅ System requirements met"  
}  

# 安装依赖  
install_dependencies() {  
    echo "📦 Installing dependencies..."  
    
    # 创建虚拟环境  
    python3 -m venv chair_env  
    source chair_env/bin/activate  
    
    # 升级pip  
    pip install --upgrade pip  
    
    # 安装PyTorch (根据系统选择版本)  
    if command -v nvidia-smi &> /dev/null; then  
        echo "Installing PyTorch with CUDA support..."  
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
    else  
        echo "Installing PyTorch CPU version..."  
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
    fi  
    
    # 安装其他依赖  
    pip install transformers datasets accelerate  
    pip install matplotlib pandas numpy scipy  
    pip install tqdm rich psutil GPUtil  
    pip install pytest black pre-commit  
    
    echo "✅ Dependencies installed"  
}  

# 设置项目结构  
setup_project() {  
    echo "📁 Setting up project structure..."  
    
    # 创建目录  
    mkdir -p {config,examples,output,logs,tests,scripts}  
    mkdir -p output/{models,generation_results,evaluations,performance}  
    
    # 设置权限  
    chmod +x scripts/*.sh 2>/dev/null || true  
    
    echo "✅ Project structure created"  
}  

# 配置环境  
configure_environment() {  
    echo "⚙️  Configuring environment..."  
    
    # 创建环境变量文件  
    cat > .env << EOF  
# Chair Style Generation Environment Variables  
PYTHONPATH=\$(pwd)  
CUDA_VISIBLE_DEVICES=0  
TRANSFORMERS_CACHE=./cache/transformers  
HF_HOME=./cache/huggingface  
BLENDER_PATH=/usr/bin/blender  
EOF  

    # 创建激活脚本  
    cat > activate.sh << 'EOF'  
#!/bin/bash  
source chair_env/bin/activate  
source .env  
export PYTHONPATH=$(pwd)  
echo "🪑 Chair Style Generation environment activated"  
echo "Python: $(which python)"  
echo "GPU: $(python -c 'import torch; print("Available" if torch.cuda.is_available() else "Not available")')"  
EOF  
    
    chmod +x activate.sh  
    
    echo "✅ Environment configured"  
}  

# 下载或配置模型  
setup_models() {  
    echo "🤖 Setting up models..."  
    
    # 检查BlenderLLM模型路径  
    if [ ! -d "/home/saisai/graph/models/BlenderLLM" ]; then  
        echo "⚠️  BlenderLLM model not found at expected path"  
        echo "Please ensure the model is available or update the path in config/default.json"  
    else  
        echo "✅ BlenderLLM model found"  
    fi  
    
    # 创建模型缓存目录  
    mkdir -p cache/{transformers,huggingface}  
    
    echo "✅ Model setup completed"  
}  

# 验证安装  
validate_installation() {  
    echo "🔬 Validating installation..."  
    
    source chair_env/bin/activate  
    
    # 测试Python导入  
    python3 -c "  
import torch  
import transformers  
import matplotlib  
import pandas  
import numpy  
print('✅ All Python packages imported successfully')  
print(f'PyTorch version: {torch.__version__}')  
print(f'CUDA available: {torch.cuda.is_available()}')  
if torch.cuda.is_available():  
    print(f'GPU count: {torch.cuda.device_count()}')  
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')  
"  
    
    # 测试脚本语法  
    echo "🔍 Checking script syntax..."  
    python3 -m py_compile quick_start.py  
    python3 -m py_compile train_chair_model.py  
    python3 -m py_compile batch_process.py  
    
    # 运行简单测试  
    echo "🧪 Running basic tests..."  
    if [ -f "tests/test_style_preprocessing.py" ]; then  
        python3 -m pytest tests/test_style_preprocessing.py -v  
    fi  
    
    echo "✅ Installation validated"  
}  

# 创建示例配置  
create_sample_config() {  
    echo "📝 Creating sample configuration..."  
    
    # 检测系统路径  
    DATA_PATH="/home/saisai/graph/Fudan-Graphics-GenAI/data_grouped"  
    MODEL_PATH="/home/saisai/graph/models/BlenderLLM"  
    
    # 如果路径不存在，使用相对路径  
    if [ ! -d "$DATA_PATH" ]; then  
        DATA_PATH="./data"  
        echo "⚠️  Creating placeholder data directory: $DATA_PATH"  
        mkdir -p "$DATA_PATH"  
    fi  
    
    if [ ! -d "$MODEL_PATH" ]; then  
        MODEL_PATH="./models/BlenderLLM"  
        echo "⚠️  Model path not found, using: $MODEL_PATH"  
    fi  
    
    # 生成配置文件  
    cat > config/production.json << EOF  
{  
    "data_path": "$DATA_PATH",  
    "base_model": "$MODEL_PATH",  
    "styles_file": "examples/chair_styles.txt",  
    "output_dir": "./output",  
    "epochs": 3,  
    "batch_size": 2,  
    "max_workers": 2,  
    "num_test_samples": 10,  
    "training": {  
        "learning_rate": 2e-5,  
        "warmup_steps": 100,  
        "logging_steps": 50,  
        "save_steps": 500,  
        "eval_steps": 500,  
        "max_seq_length": 512,  
        "gradient_accumulation_steps": 1,  
        "fp16": true,  
        "dataloader_pin_memory": true  
    },  
    "generation": {  
        "max_new_tokens": 256,  
        "temperature": 0.7,  
        "top_p": 0.9,  
        "top_k": 50,  
        "do_sample": true,  
        "num_return_sequences": 1  
    },  
    "evaluation": {  
        "metrics": ["bleu", "rouge", "success_rate"],  
        "save_failed_cases": true,  
        "generate_visualizations": true  
    }  
}  
EOF  

    # 生成开发配置  
    cat > config/development.json << EOF  
{  
    "data_path": "$DATA_PATH",  
    "base_model": "$MODEL_PATH",   
    "styles_file": "examples/chair_styles.txt",  
    "output_dir": "./output/dev",  
    "epochs": 1,  
    "batch_size": 1,  
    "max_workers": 1,  
    "num_test_samples": 3,  
    "training": {  
        "learning_rate": 1e-4,  
        "warmup_steps": 10,  
        "logging_steps": 10,  
        "save_steps": 100,  
        "eval_steps": 100,  
        "max_seq_length": 256,  
        "gradient_accumulation_steps": 2,  
        "fp16": false  
    },  
    "generation": {  
        "max_new_tokens": 128,  
        "temperature": 0.8,  
        "top_p": 0.9,  
        "do_sample": true  
    }  
}  
EOF  

    echo "✅ Sample configurations created"  
}  

# 创建管理脚本  
create_management_scripts() {  
    echo "📜 Creating management scripts..."  
    
    # 训练脚本  
    cat > scripts/train.sh << 'EOF'  
#!/bin/bash  
source ./activate.sh  

echo "🏋️ Starting model training..."  
python quick_start.py --mode train --config config/production.json "$@"  
EOF  

    # 推理脚本  
    cat > scripts/inference.sh << 'EOF'  
#!/bin/bash  
source ./activate.sh  

echo "🎨 Starting inference..."  
python quick_start.py --mode inference --config config/production.json "$@"  
EOF  

    # 完整流水线脚本  
    cat > scripts/full_pipeline.sh << 'EOF'  
#!/bin/bash  
source ./activate.sh  

echo "🚀 Starting full pipeline..."  
python quick_start.py --mode full --config config/production.json "$@"  
EOF  

    # 开发模式脚本  
    cat > scripts/dev.sh << 'EOF'  
#!/bin/bash  
source ./activate.sh  

echo "🔧 Running in development mode..."  
python quick_start.py --mode full --config config/development.json "$@"  
EOF  

    # 性能监控脚本  
    cat > scripts/monitor.sh << 'EOF'  
#!/bin/bash  
source ./activate.sh  

echo "📊 Starting performance monitoring..."  
python monitor_performance.py --action start --log_file output/performance/monitor.json "$@"  
EOF  

    # 测试脚本  
    cat > scripts/test.sh << 'EOF'  
#!/bin/bash  
source ./activate.sh  

echo "🧪 Running tests..."  
python -m pytest tests/ -v --tb=short  
EOF  

    # 清理脚本  
    cat > scripts/clean.sh << 'EOF'  
#!/bin/bash  

echo "🧹 Cleaning up..."  
rm -rf output/dev/  
rm -rf output/generation_results/temp_*  
rm -rf output/models/temp_*  
rm -rf logs/*.tmp  
rm -rf __pycache__/  
find . -name "*.pyc" -delete  
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true  

echo "✅ Cleanup completed"  
EOF  

    # 设置执行权限  
    chmod +x scripts/*.sh  
    
    echo "✅ Management scripts created"  
}  

# 生成使用文档  
generate_usage_docs() {  
    echo "📚 Generating usage documentation..."  
    
    cat > QUICK_START.md << 'EOF'  
# 🪑 Chair Style Generation - Quick Start  

## 🚀 Fast Track  

```bash  
# 1. Activate environment  
source activate.sh  

# 2. Run development pipeline (fast test)  
./scripts/dev.sh  

# 3. Run full production pipeline  
./scripts/full_pipeline.sh