#!/bin/bash  
# install.sh  

echo "🪑 Chair Style Generation Setup"  
echo "==============================="  

# 检查Python版本  
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2)  
if [ -z "$python_version" ]; then  
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."  
    exit 1  
fi  

echo "✓ Python version: $python_version"  

# 创建虚拟环境  
if [ ! -d "venv" ]; then  
    echo "📦 Creating virtual environment..."  
    python3 -m venv venv  
fi  

# 激活虚拟环境  
source venv/bin/activate  
echo "✓ Virtual environment activated"  

# 升级pip  
pip install --upgrade pip  

# 安装依赖  
echo "📚 Installing dependencies..."  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install transformers datasets accelerate  
pip install matplotlib pandas numpy  
pip install tqdm rich  

# 创建必要的目录  
mkdir -p config examples output logs  

# 复制配置文件模板  
if [ ! -f "config/default.json" ]; then  
    echo "📝 Creating default configuration..."  
    cat > config/default.json << 'EOF'  
{  
    "data_path": "/path/to/your/data",  
    "base_model": "/path/to/BlenderLLM",  
    "styles_file": "examples/chair_styles.txt",  
    "output_dir": "./output",  
    "epochs": 3,  
    "batch_size": 2,  
    "max_workers": 2,  
    "num_test_samples": 10  
}  
EOF  
fi  

# 创建运行脚本  
cat > run.sh << 'EOF'  
#!/bin/bash  
source venv/bin/activate  
python quick_start.py "$@"  
EOF  

chmod +x run.sh  

echo ""  
echo "🎉 Setup completed successfully!"  
echo ""  
echo "📋 Next steps:"  
echo "1. Update config/default.json with your paths"  
echo "2. Add your style descriptions to examples/chair_styles.txt"  
echo "3. Run: ./run.sh --mode full"  
echo ""  
echo "📖 For more options: ./run.sh --help"