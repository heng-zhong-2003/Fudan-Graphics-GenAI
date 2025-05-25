#!/usr/bin/env python3  
"""  
GPU设备检查和修复脚本  
"""  

import torch  
import os  

def check_gpu_status():  
    """检查GPU状态"""  
    print("🔍 GPU Status Check")  
    print("=" * 40)  
    
    if not torch.cuda.is_available():  
        print("❌ CUDA not available")  
        return False  
    
    gpu_count = torch.cuda.device_count()  
    print(f"🚀 Found {gpu_count} GPU(s)")  
    
    for i in range(gpu_count):  
        props = torch.cuda.get_device_properties(i)  
        memory_gb = props.total_memory / 1e9  
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")  
    
    current_device = torch.cuda.current_device()  
    print(f"📍 Current device: cuda:{current_device}")  
    
    return True  

def set_single_gpu():  
    """设置使用单个GPU"""  
    if torch.cuda.is_available():  
        # 设置只使用第一个GPU  
        torch.cuda.set_device(0)  
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
        print("✅ Set to use only GPU 0")  
        return True  
    return False  

def test_tensor_operations():  
    """测试tensor操作"""  
    print("\n🧪 Testing tensor operations...")  
    
    if not torch.cuda.is_available():  
        print("⚠️  Using CPU")  
        device = torch.device("cpu")  
    else:  
        device = torch.device("cuda:0")  
        print(f"✅ Using {device}")  
    
    try:  
        # 创建测试tensor  
        a = torch.randn(10, 10).to(device)  
        b = torch.randn(10, 10).to(device)  
        c = torch.matmul(a, b)  
        
        print(f"   Tensor device: {a.device}")  
        print(f"   Operation successful: {c.shape}")  
        return True  
        
    except Exception as e:  
        print(f"❌ Tensor operation failed: {e}")  
        return False  

if __name__ == "__main__":  
    check_gpu_status()  
    set_single_gpu()  
    test_tensor_operations()  
