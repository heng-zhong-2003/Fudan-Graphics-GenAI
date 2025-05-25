#!/usr/bin/env python3  
"""  
详细检查所有模型参数的状态  
"""  

import torch  
import os  

def check_params_detailed(params_path, model_name):  
    """详细检查参数状态"""  
    print(f"\n🔍 检查 {model_name}")  
    print("-" * 40)  
    
    if not os.path.exists(params_path):  
        print(f"❌ 文件不存在: {params_path}")  
        return False  
    
    try:  
        params = torch.load(params_path, map_location='cpu')  
        
        print(f"📁 参数文件: {params_path}")  
        print(f"🔢 参数数量: {len(params)}")  
        
        all_good = True  
        
        for name, param in params.items():  
            has_nan = torch.isnan(param).any()  
            has_inf = torch.isinf(param).any()  
            
            # 计算一些统计信息  
            param_min = param.min().item()  
            param_max = param.max().item()  
            param_mean = param.mean().item()  
            param_std = param.std().item()  
            
            status = "✅" if not (has_nan or has_inf) else "❌"  
            print(f"{status} {name}:")  
            print(f"    形状: {param.shape}")  
            print(f"    NaN: {has_nan}, Inf: {has_inf}")  
            print(f"    范围: [{param_min:.6f}, {param_max:.6f}]")  
            print(f"    均值: {param_mean:.6f}, 标准差: {param_std:.6f}")  
            
            if has_nan or has_inf:  
                all_good = False  
            
            # 检查是否有异常大的值  
            if abs(param_max) > 1e10 or abs(param_min) > 1e10:  
                print(f"    ⚠️ 存在异常大的值")  
                all_good = False  
        
        return all_good  
        
    except Exception as e:  
        print(f"❌ 检查失败: {e}")  
        return False  

def main():  
    """主函数"""  
    print("🔬 模型参数全面检查")  
    print("=" * 60)  
    
    # 检查各种模型参数  
    models_to_check = [  
        ("损坏的原始微调", "./output/emergency_model/trainable_params.pt"),  
        ("修复后的模型", "./output/fixed_model/trainable_params.pt"),  
        ("安全模型", "./output/safe_model/trainable_params.pt"),  
    ]  
    
    results = {}  
    
    for model_name, params_path in models_to_check:  
        is_good = check_params_detailed(params_path, model_name)  
        results[model_name] = is_good  
    
    # 总结  
    print("\n" + "="*60)  
    print("📊 检查结果总结")  
    print("="*60)  
    
    for model_name, is_good in results.items():  
        status = "✅ 正常" if is_good else "❌ 有问题"  
        print(f"{model_name}: {status}")  
    
    # 额外检查：对比原始模型的某个参数  
    print("\n🔍 对比检查:")  
    try:  
        from transformers import AutoModelForCausalLM  
        print("加载原始模型进行对比...")  
        original_model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map="cpu"  
        )  
        
        # 检查lm_head.weight参数  
        original_lm_head = None  
        for name, param in original_model.named_parameters():  
            if name == "lm_head.weight":  
                original_lm_head = param.clone()  
                break  
        
        if original_lm_head is not None:  
            print(f"\n原始 lm_head.weight:")  
            print(f"  形状: {original_lm_head.shape}")  
            print(f"  范围: [{original_lm_head.min():.6f}, {original_lm_head.max():.6f}]")  
            print(f"  均值: {original_lm_head.mean():.6f}")  
            print(f"  NaN: {torch.isnan(original_lm_head).any()}")  
            print(f"  Inf: {torch.isinf(original_lm_head).any()}")  
        
        del original_model  
        
    except Exception as e:  
        print(f"⚠️ 无法加载原始模型进行对比: {e}")  

if __name__ == "__main__":  
    main()  
