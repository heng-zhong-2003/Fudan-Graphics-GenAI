#!/bin/bash  

echo "🔧 修复训练环境..."  

# 1. 修复临时目录  
echo "📁 修复临时目录..."  
sudo mkdir -p /tmp  
sudo chmod 1777 /tmp  

# 创建项目临时目录  
mkdir -p ./temp  
chmod 755 ./temp  

# 2. 清理GPU内存  
echo "🧹 清理GPU内存..."  
python3 -c "  
import torch  
if torch.cuda.is_available():  
    torch.cuda.empty_cache()  
    print('GPU内存已清理')  
else:  
    print('GPU不可用')  
"  

# 3. 检查磁盘空间  
echo "💾 检查磁盘空间..."  
df -h .  

# 4. 设置环境变量  
export TMPDIR=./temp  
export TEMP=./temp  
export TMP=./temp  
export CUDA_VISIBLE_DEVICES=0  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  

echo "✅ 环境修复完成"  
echo "🚀 开始运行紧急修复训练..."  

python3 emergency_fix_train.py  

chmod +x fix_environment.sh