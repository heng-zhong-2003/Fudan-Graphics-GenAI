#!/usr/bin/env python3  
"""  
测试所有修复是否生效  
"""  

import os  
import sys  
import tempfile  

def test_temp_directory():  
    """测试临时目录"""  
    print("🧪 测试临时目录...")  
    try:  
        with tempfile.NamedTemporaryFile() as tmp:  
            print(f"✅ 临时目录工作正常: {tmp.name}")  
        return True  
    except Exception as e:  
        print(f"❌ 临时目录问题: {e}")  
        return False  

def test_torch_import():  
    """测试PyTorch导入"""  
    print("🧪 测试PyTorch导入...")  
    try:  
        import torch  
        print(f"✅ PyTorch版本: {torch.__version__}")  
        
        if torch.cuda.is_available():  
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")  
            print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")  
        else:  
            print("⚠️ CUDA不可用")  
        
        return True  
    except Exception as e:  
        print(f"❌ PyTorch导入失败: {e}")  
        return False  

def test_transformers():  
    """测试Transformers"""  
    print("🧪 测试Transformers...")  
    try:  
        from transformers import AutoTokenizer  
        print("✅ Transformers可用")  
        return True  
    except Exception as e:  
        print(f"❌ Transformers问题: {e}")  
        return False  

def test_disk_space():  
    """测试磁盘空间"""  
    print("🧪 测试磁盘空间...")  
    import shutil  
    total, used, free = shutil.disk_usage("./")  
    free_gb = free / (1024**3)  
    print(f"💾 可用空间: {free_gb:.2f} GB")  
    
    if free_gb < 2:  
        print("❌ 磁盘空间不足")  
        return False  
    else:  
        print("✅ 磁盘空间充足")  
        return True  

def main():  
    print("🔍 环境测试报告")  
    print("=" * 40)  
    
    tests = [  
        test_temp_directory,  
        test_torch_import,  
        test_transformers,  
        test_disk_space  
    ]  
    
    results = []  
    for test in tests:  
        result = test()  
        results.append(result)  
        print()  
    
    print("📊 测试结果:")  
    if all(results):  
        print("✅ 所有测试通过，环境就绪")  
        return True  
    else:  
        print("❌ 部分测试失败，需要修复")  
        return False  

if __name__ == "__main__":  
    if main():  
        print("\n🚀 可以开始训练了！")  
        print("运行: python emergency_fix_train.py")  
    else:  
        print("\n🔧 请先运行: ./fix_environment.sh")  
