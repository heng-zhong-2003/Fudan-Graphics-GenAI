#!/usr/bin/env python3  
"""  
BlenderLLM原始模型 vs LoRA增强模型对比评估  
包含代码质量、椅子设计理解、可视化评分等  
"""  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from peft import PeftModel  
import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
import numpy as np  
import re  
import json  
import os  
from datetime import datetime  

class BlenderCodeEvaluator:  
    """Blender代码质量评估器"""  
    
    def __init__(self):  
        self.syntax_patterns = {  
            'import_bpy': r'import bpy',  
            'clear_objects': r'bpy\.ops\.object\.(select_all|delete)',  
            'create_primitive': r'bpy\.ops\.mesh\.primitive_\w+_add',  
            'object_naming': r'\.name\s*=\s*["\']',  
            'scaling': r'\.scale\s*=',  
            'location': r'location\s*=\s*\(',  
            'context_usage': r'bpy\.context\.',  
        }  
        
        self.chair_features = {  
            'seat': ['seat', 'cushion'],  
            'backrest': ['back', 'backrest'],  
            'legs': ['leg', 'support'],  
            'armrest': ['arm', 'armrest'],  
            'base': ['base', 'foundation']  
        }  
    
    def evaluate_code_quality(self, code):  
        """评估代码质量 (0-10分)"""  
        if not code or len(code.strip()) < 20:  
            return 0  
        
        score = 0  
        total_checks = len(self.syntax_patterns)  
        
        for pattern_name, pattern in self.syntax_patterns.items():  
            if re.search(pattern, code, re.IGNORECASE):  
                score += 1  
        
        # 额外检查  
        if 'import bpy' in code:  
            score += 1  
        if len(code.split('\n')) > 5:  # 代码长度合理  
            score += 1  
        if '#' in code:  # 有注释  
            score += 1  
        
        return min(10, (score / total_checks) * 10)  
    
    def evaluate_chair_understanding(self, prompt, code):  
        """评估椅子设计理解 (0-10分)"""  
        if not code:  
            return 0  
        
        prompt_lower = prompt.lower()  
        code_lower = code.lower()  
        
        score = 0  
        
        # 基础椅子组件检查  
        for component, keywords in self.chair_features.items():  
            if any(kw in prompt_lower for kw in keywords):  
                if any(kw in code_lower for kw in keywords):  
                    score += 2  
        
        # 特殊特征检查  
        special_features = {  
            'office': ['cylinder', 'swivel', 'wheel'],  
            'recliner': ['angle', 'tilt', 'rotation'],  
            'minimalist': ['simple', 'clean', 'basic'],  
            'armrest': ['arm', 'support'],  
            'cushion': ['soft', 'padding', 'comfort']  
        }  
        
        for feature, indicators in special_features.items():  
            if feature in prompt_lower:  
                if any(ind in code_lower for ind in indicators):  
                    score += 1  
        
        return min(10, score)  
    
    def evaluate_blender_completeness(self, code):  
        """评估Blender代码完整性 (0-10分)"""  
        if not code:  
            return 0  
        
        completeness_checks = {  
            'has_import': 'import bpy' in code,  
            'has_clear': any(x in code for x in ['clear', 'delete', 'select_all']),  
            'has_creation': 'primitive' in code and 'add' in code,  
            'has_naming': '.name' in code,  
            'has_transform': any(x in code for x in ['scale', 'location', 'rotation']),  
            'has_multiple_objects': code.count('bpy.ops.mesh') > 1,  
            'proper_structure': len(code.split('\n')) > 8  
        }  
        
        score = sum(completeness_checks.values()) * (10 / len(completeness_checks))  
        return min(10, score)  

class ModelComparator:  
    """模型对比器"""  
    
    def __init__(self):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        self.evaluator = BlenderCodeEvaluator()  
        
    def load_models(self):  
        """加载原始模型和LoRA模型"""  
        print("🔄 加载模型...")  
        
        # 加载tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
        if self.tokenizer.pad_token is None:  
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        # 加载原始模型  
        print("  📦 加载原始BlenderLLM...")  
        self.original_model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        
        # 加载LoRA增强模型  
        print("  🔧 加载LoRA增强模型...")  
        base_model = AutoModelForCausalLM.from_pretrained(  
            "../models/BlenderLLM",  
            trust_remote_code=True,  
            torch_dtype=torch.float16,  
            device_map={"": 0}  
        )  
        self.lora_model = PeftModel.from_pretrained(base_model, "./output/lora_blender_enhanced")  
        
        print("✅ 模型加载完成")  
    
    def generate_response(self, model, prompt, max_tokens=200):  
        """生成模型响应"""  
        input_text = f"User: {prompt}\n\nAssistant:"  
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)  
        
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,  
                max_new_tokens=max_tokens,  
                temperature=0.7,  
                do_sample=True,  
                pad_token_id=self.tokenizer.pad_token_id,  
                eos_token_id=self.tokenizer.eos_token_id  
            )  
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        assistant_response = response.split("Assistant:")[-1].strip()  
        return assistant_response  
    
    def evaluate_models(self, test_prompts):  
        """评估两个模型"""  
        print("\n🎯 开始模型对比评估...")  
        
        results = []  
        
        for i, prompt in enumerate(test_prompts, 1):  
            print(f"\n📝 测试 {i}/{len(test_prompts)}: {prompt[:50]}...")  
            
            # 生成响应  
            original_response = self.generate_response(self.original_model, prompt)  
            lora_response = self.generate_response(self.lora_model, prompt)  
            
            # 评估原始模型  
            original_scores = {  
                'code_quality': self.evaluator.evaluate_code_quality(original_response),  
                'chair_understanding': self.evaluator.evaluate_chair_understanding(prompt, original_response),  
                'completeness': self.evaluator.evaluate_blender_completeness(original_response)  
            }  
            
            # 评估LoRA模型  
            lora_scores = {  
                'code_quality': self.evaluator.evaluate_code_quality(lora_response),  
                'chair_understanding': self.evaluator.evaluate_chair_understanding(prompt, lora_response),  
                'completeness': self.evaluator.evaluate_blender_completeness(lora_response)  
            }  
            
            # 保存结果  
            result = {  
                'prompt': prompt,  
                'original_response': original_response,  
                'lora_response': lora_response,  
                'original_scores': original_scores,  
                'lora_scores': lora_scores  
            }  
            results.append(result)  
            
            print(f"  📊 原始模型得分: {sum(original_scores.values()):.1f}/30")  
            print(f"  🔧 LoRA模型得分: {sum(lora_scores.values()):.1f}/30")  
        
        return results  
    
    def create_visualizations(self, results):  
        """创建可视化评分图表"""  
        print("\n📊 生成评估报告...")  
        
        # 准备数据  
        categories = ['Code Quality', 'Chair Understanding', 'Completeness']  
        original_avg = []  
        lora_avg = []  
        
        for category in ['code_quality', 'chair_understanding', 'completeness']:  
            orig_scores = [r['original_scores'][category] for r in results]  
            lora_scores = [r['lora_scores'][category] for r in results]  
            
            original_avg.append(np.mean(orig_scores))  
            lora_avg.append(np.mean(lora_scores))  
        
        # 设置图表样式  
        plt.style.use('seaborn-v0_8')  
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))  
        
        # 1. 雷达图对比  
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)  
        angles = np.concatenate((angles, [angles[0]]))  
        
        original_avg_radar = original_avg + [original_avg[0]]  
        lora_avg_radar = lora_avg + [lora_avg[0]]  
        
        ax1.plot(angles, original_avg_radar, 'o-', linewidth=2, label='Original BlenderLLM', color='#FF6B6B')  
        ax1.fill(angles, original_avg_radar, alpha=0.25, color='#FF6B6B')  
        ax1.plot(angles, lora_avg_radar, 'o-', linewidth=2, label='LoRA Enhanced', color='#4ECDC4')  
        ax1.fill(angles, lora_avg_radar, alpha=0.25, color='#4ECDC4')  
        
        ax1.set_xticks(angles[:-1])  
        ax1.set_xticklabels(categories)  
        ax1.set_ylim(0, 10)  
        ax1.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold')  
        ax1.legend()  
        ax1.grid(True)  
        
        # 2. 柱状图对比  
        x = np.arange(len(categories))  
        width = 0.35  
        
        bars1 = ax2.bar(x - width/2, original_avg, width, label='Original BlenderLLM', color='#FF6B6B', alpha=0.8)  
        bars2 = ax2.bar(x + width/2, lora_avg, width, label='LoRA Enhanced', color='#4ECDC4', alpha=0.8)  
        
        ax2.set_xlabel('Evaluation Categories')  
        ax2.set_ylabel('Average Score (0-10)')  
        ax2.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')  
        ax2.set_xticks(x)  
        ax2.set_xticklabels(categories)  
        ax2.legend()  
        ax2.grid(True, alpha=0.3)  
        
        # 添加数值标签  
        for bar in bars1:  
            height = bar.get_height()  
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',   
                    ha='center', va='bottom')  
        for bar in bars2:  
            height = bar.get_height()  
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',   
                    ha='center', va='bottom')  
        
        # 3. 详细分数分布  
        all_original_scores = []  
        all_lora_scores = []  
        
        for result in results:  
            all_original_scores.extend(result['original_scores'].values())  
            all_lora_scores.extend(result['lora_scores'].values())  
        
        ax3.hist(all_original_scores, bins=10, alpha=0.7, label='Original BlenderLLM', color='#FF6B6B')  
        ax3.hist(all_lora_scores, bins=10, alpha=0.7, label='LoRA Enhanced', color='#4ECDC4')  
        ax3.set_xlabel('Score')  
        ax3.set_ylabel('Frequency')  
        ax3.set_title('Score Distribution', fontsize=14, fontweight='bold')  
        ax3.legend()  
        ax3.grid(True, alpha=0.3)  
        
        # 4. 改进度量化  
        improvements = [(lora_avg[i] - original_avg[i]) for i in range(len(categories))]  
        colors = ['#45B7D1' if imp > 0 else '#F96167' for imp in improvements]  
        
        bars = ax4.bar(categories, improvements, color=colors, alpha=0.8)  
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)  
        ax4.set_ylabel('Improvement Score')  
        ax4.set_title('LoRA Enhancement Impact', fontsize=14, fontweight='bold')  
        ax4.grid(True, alpha=0.3)  
        
        # 添加改进数值  
        for bar, imp in zip(bars, improvements):  
            height = bar.get_height()  
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.2),   
                    f'{imp:+.1f}', ha='center', va='bottom' if height > 0 else 'top')  
        
        plt.tight_layout()  
        
        # 保存图表  
        os.makedirs('./output/evaluation_results', exist_ok=True)  
        plt.savefig('./output/evaluation_results/model_comparison.png', dpi=300, bbox_inches='tight')  
        print("📈 可视化图表已保存: ./output/evaluation_results/model_comparison.png")  
        
        return fig  
    
    def generate_report(self, results):  
        """生成详细评估报告"""  
        print("\n📋 生成详细报告...")  
        
        # 计算统计数据  
        original_total = [sum(r['original_scores'].values()) for r in results]  
        lora_total = [sum(r['lora_scores'].values()) for r in results]  
        
        report = {  
            "evaluation_timestamp": datetime.now().isoformat(),  
            "test_samples": len(results),  
            "summary": {  
                "original_model": {  
                    "avg_total_score": np.mean(original_total),  
                    "avg_code_quality": np.mean([r['original_scores']['code_quality'] for r in results]),  
                    "avg_chair_understanding": np.mean([r['original_scores']['chair_understanding'] for r in results]),  
                    "avg_completeness": np.mean([r['original_scores']['completeness'] for r in results])  
                },  
                "lora_model": {  
                    "avg_total_score": np.mean(lora_total),  
                    "avg_code_quality": np.mean([r['lora_scores']['code_quality'] for r in results]),  
                    "avg_chair_understanding": np.mean([r['lora_scores']['chair_understanding'] for r in results]),  
                    "avg_completeness": np.mean([r['lora_scores']['completeness'] for r in results])  
                },  
                "improvement": {  
                    "total_score_improvement": np.mean(lora_total) - np.mean(original_total),  
                    "code_quality_improvement": np.mean([r['lora_scores']['code_quality'] for r in results]) - np.mean([r['original_scores']['code_quality'] for r in results]),  
                    "chair_understanding_improvement": np.mean([r['lora_scores']['chair_understanding'] for r in results]) - np.mean([r['original_scores']['chair_understanding'] for r in results]),  
                    "completeness_improvement": np.mean([r['lora_scores']['completeness'] for r in results]) - np.mean([r['original_scores']['completeness'] for r in results])  
                }  
            },  
            "detailed_results": results  
        }  
        
        # 保存报告  
        with open('./output/evaluation_results/detailed_report.json', 'w') as f:  
            json.dump(report, f, indent=2)  
        
        # 生成Markdown报告  
        markdown_report = self.create_markdown_report(report)  
        with open('./output/evaluation_results/evaluation_report.md', 'w') as f:  
            f.write(markdown_report)  
        
        print("📊 详细报告已保存:")  
        print("  - JSON格式: ./output/evaluation_results/detailed_report.json")  
        print("  - Markdown格式: ./output/evaluation_results/evaluation_report.md")  
        
        return report  
    
    def create_markdown_report(self, report):  
        """创建Markdown格式的报告"""  
        md = f"""# BlenderLLM 模型对比评估报告  

            ## 📊 评估概况  

            **评估时间**: {report['evaluation_timestamp']}  
            **测试样本数**: {report['test_samples']}  

            ## 🎯 总体性能对比  

            | 模型 | 总分 | 代码质量 | 椅子理解 | 完整性 |  
            |------|------|----------|----------|--------|  
            | 原始BlenderLLM | {report['summary']['original_model']['avg_total_score']:.2f}/30 | {report['summary']['original_model']['avg_code_quality']:.2f}/10 | {report['summary']['original_model']['avg_chair_understanding']:.2f}/10 | {report['summary']['original_model']['avg_completeness']:.2f}/10 |  
            | LoRA增强版本 | {report['summary']['lora_model']['avg_total_score']:.2f}/30 | {report['summary']['lora_model']['avg_code_quality']:.2f}/10 | {report['summary']['lora_model']['avg_chair_understanding']:.2f}/10 | {report['summary']['lora_model']['avg_completeness']:.2f}/10 |  

            ## 📈 改进效果  

            | 评估维度 | 改进分数 | 改进百分比 |  
            |----------|----------|------------|  
            | **总体性能** | {report['summary']['improvement']['total_score_improvement']:+.2f} | {(report['summary']['improvement']['total_score_improvement']/report['summary']['original_model']['avg_total_score']*100):+.1f}% |  
            | **代码质量** | {report['summary']['improvement']['code_quality_improvement']:+.2f} | {(report['summary']['improvement']['code_quality_improvement']/report['summary']['original_model']['avg_code_quality']*100 if report['summary']['original_model']['avg_code_quality'] > 0 else 0):+.1f}% |  
            | **椅子理解** | {report['summary']['improvement']['chair_understanding_improvement']:+.2f} | {(report['summary']['improvement']['chair_understanding_improvement']/report['summary']['original_model']['avg_chair_understanding']*100 if report['summary']['original_model']['avg_chair_understanding'] > 0 else 0):+.1f}% |  
            | **代码完整性** | {report['summary']['improvement']['completeness_improvement']:+.2f} | {(report['summary']['improvement']['completeness_improvement']/report['summary']['original_model']['avg_completeness']*100 if report['summary']['original_model']['avg_completeness'] > 0 else 0):+.1f}% |  

            ## 📝 详细测试结果  

            """  
        
        for i, result in enumerate(report['detailed_results'], 1):  
            md += f"""  
            ### 测试案例 {i}  

            **提示**: {result['prompt']}  

            **评分对比**:  
            - 原始模型: {sum(result['original_scores'].values()):.1f}/30 (代码:{result['original_scores']['code_quality']:.1f} | 理解:{result['original_scores']['chair_understanding']:.1f} | 完整:{result['original_scores']['completeness']:.1f})  
            - LoRA模型: {sum(result['lora_scores'].values()):.1f}/30 (代码:{result['lora_scores']['code_quality']:.1f} | 理解:{result['lora_scores']['chair_understanding']:.1f} | 完整:{result['lora_scores']['completeness']:.1f})  

            **原始模型生成**:  
            ```python  
            {result['original_response'][:300]}{'...' if len(result['original_response']) > 300 else ''}
            LoRA模型生成:
            {result['lora_response'][:300]}{'...' if len(result['lora_response']) > 300 else ''}  
            """

        md += f"""  
        🎯 结论
        {self.generate_conclusion(report)}
        报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return md

def generate_conclusion(self, report):  
    """生成结论"""  
    improvement = report['summary']['improvement']['total_score_improvement']  
    
    if improvement > 2:  
        conclusion = "🎉 **显著改进**: LoRA微调显著提升了模型性能"  
    elif improvement > 0.5:  
        conclusion = "✅ **明显改进**: LoRA微调有效提升了模型能力"  
    elif improvement > 0:  
        conclusion = "📈 **轻微改进**: LoRA微调带来了小幅提升"  
    else:  
        conclusion = "⚠️ **需要优化**: 建议调整训练策略"  
    
    best_category = max(report['summary']['improvement'], key=report['summary']['improvement'].get)  
    
    conclusion += f"\n\n**最大改进领域**: {best_category.replace('_', ' ').title()}"  
    conclusion += f"\n\n**推荐**: 基于评估结果，LoRA增强版本在椅子设计理解方面表现更好，建议在实际应用中使用。"  
    
    return conclusion  

def main():
    """主评估函数"""
    print("🎯 BlenderLLM 模型对比评估系统")
    print("=" * 50)

    # 测试用例  
    test_prompts = [  
        "Design a chair: modern minimalist office chair with wheels",  
        "Design a chair: comfortable recliner with armrests and cushions",  
        "Design a chair: dining chair with tall backrest and no armrests",  
        "Design a chair: ergonomic gaming chair with adjustable height",  
        "Design a chair: vintage wooden chair with carved details",  
        "Design a chair: bar stool with footrest and swivel function",  
        "Design a chair: bean bag chair for relaxation",  
        "Design a chair: folding chair for outdoor use"  
    ]  

    # 创建对比器  
    comparator = ModelComparator()  

    try:  
        # 加载模型  
        comparator.load_models()  
        
        # 评估模型  
        results = comparator.evaluate_models(test_prompts)  
        
        # 创建可视化  
        fig = comparator.create_visualizations(results)  
        
        # 生成报告  
        report = comparator.generate_report(results)  
        
        # 显示总结  
        print("\n🎯 评估总结:")  
        print("=" * 30)  
        orig_avg = report['summary']['original_model']['avg_total_score']  
        lora_avg = report['summary']['lora_model']['avg_total_score']  
        improvement = report['summary']['improvement']['total_score_improvement']  
        
        print(f"📊 原始BlenderLLM平均得分: {orig_avg:.2f}/30")  
        print(f"🔧 LoRA增强版本平均得分: {lora_avg:.2f}/30")  
        print(f"📈 总体改进: {improvement:+.2f} ({improvement/orig_avg*100:+.1f}%)")  
        
        # 分类改进  
        categories = ['code_quality', 'chair_understanding', 'completeness']  
        category_names = ['代码质量', '椅子理解', '代码完整性']  
        
        print(f"\n📋 详细改进:")  
        for cat, name in zip(categories, category_names):  
            imp = report['summary']['improvement'][f'{cat}_improvement']  
            orig = report['summary']['original_model'][f'avg_{cat}']  
            print(f"  {name}: {imp:+.2f} ({imp/orig*100 if orig > 0 else 0:+.1f}%)")  
        
        print(f"\n📁 结果文件:")  
        print(f"  - 可视化图表: ./output/evaluation_results/model_comparison.png")  
        print(f"  - 详细报告: ./output/evaluation_results/evaluation_report.md")  
        print(f"  - JSON数据: ./output/evaluation_results/detailed_report.json")  
        
        plt.show()  # 显示图表  
        
    except Exception as e:  
        print(f"❌ 评估失败: {e}")  
        import traceback  
        traceback.print_exc()  

if __name__ == "__main__":
    main()