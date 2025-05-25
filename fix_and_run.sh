#!/bin/bash  

echo "🔧 Chair Style Generation - Fix and Run"  
echo "======================================"  

# 检查Python环境  
if ! command -v python &> /dev/null; then  
    echo "❌ Python not found!"  
    exit 1  
fi  

# 创建输出目录  
mkdir -p output logs  

# 1. 验证和修复数据  
echo "🔍 Step 1: Validating and fixing data..."  
python validate_data.py --data_path ./data_grouped --output_path ./output/cleaned_data  

# 2. 检查修复结果  
if [ -f "./output/cleaned_data/cleaned_data.json" ]; then  
    echo "✅ Data validation completed"  
    DATA_PATH="./output/cleaned_data/cleaned_data.json"  
else  
    echo "⚠️  Using original data path"  
    DATA_PATH="./data_grouped"  
fi  

# 3. 运行训练（小规模测试）  
echo "🏋️ Step 2: Starting training..."  
python fixed_train_chair_model.py \
    --data_path "$DATA_PATH" \
    --base_model ../models/BlenderLLM \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_length 512 \
    --output_dir ./output/test_model  

echo "🎉 Training completed!"  
echo "📁 Check results in ./output/"  
EOF  

chmod +x fix_and_run.sh  