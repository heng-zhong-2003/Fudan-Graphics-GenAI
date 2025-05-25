# evaluate_results.py  
#!/usr/bin/env python3  
"""  
评估批量生成结果  
"""  

import os  
import sys  
import json  
import argparse  
from pathlib import Path  
import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np  

def main():  
    parser = argparse.ArgumentParser(description='Evaluate Batch Generation Results')  
    parser.add_argument('--batch_results_dir', type=str, required=True,  
                       help='Directory containing batch generation results')  
    parser.add_argument('--output_dir', type=str, required=True,  
                       help='Output directory for evaluation results')  
    
    args = parser.parse_args()  
    
    batch_dir = Path(args.batch_results_dir)  
    output_dir = Path(args.output_dir)  
    output_dir.mkdir(parents=True, exist_ok=True)  
    
    if not batch_dir.exists():  
        print(f"Error: Batch results directory does not exist: {batch_dir}")  
        return  
    
    print(f"Evaluating results from: {batch_dir}")  
    print(f"Output directory: {output_dir}")  
    
    # 加载批量处理摘要  
    summary_file = batch_dir / "batch_summary.json"  
    if summary_file.exists():  
        with open(summary_file) as f:  
            batch_summary = json.load(f)  
    else:  
        print("Warning: No batch_summary.json found")  
        batch_summary = {}  
    
    # 收集所有结果  
    results = collect_all_results(batch_dir)  
    
    if not results:  
        print("No results found to evaluate")  
        return  
    
    # 生成评估报告  
    evaluation_report = generate_evaluation_report(results, batch_summary)  
    
    # 保存评估结果  
    with open(output_dir / "evaluation_report.json", 'w') as f:  
        json.dump(evaluation_report, f, indent=2, default=str)  
    
    # 生成可视化图表  
    generate_visualizations(results, evaluation_report, output_dir)  
    
    # 生成Markdown报告  
    generate_markdown_report(evaluation_report, output_dir)  
    
    print("✅ Evaluation completed successfully!")  
    print(f"Report saved to: {output_dir}")  

def collect_all_results(batch_dir):  
    """收集所有生成结果"""  
    results = []  
    
    # 遍历所有style_*目录  
    for style_dir in batch_dir.glob("style_*"):  
        if style_dir.is_dir():  
            result_file = style_dir / "generation_result.json"  
            if result_file.exists():  
                try:  
                    with open(result_file) as f:  
                        result = json.load(f)  
                    
                    # 检查生成的文件是否存在  
                    result['file_exists'] = check_generated_files(style_dir, result)  
                    results.append(result)  
                    
                except Exception as e:  
                    print(f"Error loading result from {style_dir}: {e}")  
    
    return results  

def check_generated_files(style_dir, result):  
    """检查生成的文件是否存在"""  
    if result['status'] != 'success' or not result.get('generated_views'):  
        return False  
    
    for view_name, view_path in result['generated_views'].items():  
        if not Path(view_path).exists():  
            return False  
    
    return True  

def generate_evaluation_report(results, batch_summary):  
    """生成评估报告"""  
    total_results = len(results)  
    successful_results = [r for r in results if r['status'] == 'success']  
    failed_results = [r for r in results if r['status'] == 'failed']  
    
    # 基础统计  
    success_count = len(successful_results)  
    failure_count = len(failed_results)  
    success_rate = success_count / total_results if total_results > 0 else 0  
    
    # 文件存在性检查  
    files_exist_count = len([r for r in successful_results if r.get('file_exists', False)])  
    file_existence_rate = files_exist_count / success_count if success_count > 0 else 0  
    
    # 分析失败原因  
    failure_reasons = {}  
    for result in failed_results:  
        error = result.get('error', 'Unknown error')  
        # 简化错误信息  
        simplified_error = simplify_error_message(error)  
        failure_reasons[simplified_error] = failure_reasons.get(simplified_error, 0) + 1  
    
    # 分析风格类型分布  
    style_analysis = analyze_style_distribution(results)  
    
    # 性能分析  
    performance_analysis = analyze_performance(results)  
    
    evaluation_report = {  
        'summary': {  
            'total_processed': total_results,  
            'successful': success_count,  
            'failed': failure_count,  
            'success_rate': success_rate,  
            'files_exist_count': files_exist_count,  
            'file_existence_rate': file_existence_rate  
        },  
        'failure_analysis': {  
            'reasons': failure_reasons,  
            'most_common_failure': max(failure_reasons.items(), key=lambda x: x[1])[0] if failure_reasons else None  
        },  
        'style_analysis': style_analysis,  
        'performance_analysis': performance_analysis,  
        'batch_summary': batch_summary  
    }  
    
    return evaluation_report  

def simplify_error_message(error_msg):  
    """简化错误信息"""  
    error_msg = str(error_msg).lower()  
    
    if 'cuda' in error_msg or 'gpu' in error_msg:  
        return 'GPU/CUDA Error'  
    elif 'memory' in error_msg or 'oom' in error_msg:  
        return 'Memory Error'  
    elif 'model' in error_msg and ('load' in error_msg or 'not found' in error_msg):  
        return 'Model Loading Error'  
    elif 'timeout' in error_msg:  
        return 'Timeout Error'  
    elif 'blender' in error_msg:  
        return 'Blender Execution Error'  
    elif 'generation' in error_msg:  
        return 'Generation Error'  
    else:  
        return 'Other Error'  

def analyze_style_distribution(results):  
    """分析风格分布"""  
    style_stats = {  
        'by_status': {'success': {}, 'failed': {}},  
        'total_styles': {}  
    }  
    
    for result in results:  
        status = result['status']  
        style_desc = result.get('style_description', 'Unknown')  
        
        # 提取风格关键词  
        style_keywords = extract_style_keywords(style_desc)  
        
        for keyword in style_keywords:  
            # 按状态统计  
            if keyword not in style_stats['by_status'][status]:  
                style_stats['by_status'][status][keyword] = 0  
            style_stats['by_status'][status][keyword] += 1  
            
            # 总体统计  
            if keyword not in style_stats['total_styles']:  
                style_stats['total_styles'][keyword] = 0  
            style_stats['total_styles'][keyword] += 1  
    
    return style_stats  

def extract_style_keywords(style_description):  
    """从风格描述中提取关键词"""  
    keywords = []  
    desc = str(style_description).lower()  
    
    # 风格关键词映射  
    style_keywords = {  
        '现代': ['现代', 'modern', '极简', 'minimalist'],  
        '传统': ['传统', 'traditional', '古典', 'classical'],  
        '工业': ['工业', 'industrial', '金属'],  
        '办公': ['办公', 'office', '可调节'],  
        '实木': ['实木', 'wood', '木质'],  
        '人体工学': ['人体工学', 'ergonomic'],  
        '可折叠': ['折叠', 'foldable'],  
        '可调节': ['可调节', 'adjustable']  
    }  
    
    for category, terms in style_keywords.items():  
        if any(term in desc for term in terms):  
            keywords.append(category)  
    
    return keywords if keywords else ['其他']  

def analyze_performance(results):  
    """分析性能数据"""  
    successful_results = [r for r in results if r['status'] == 'success']  
    
    if not successful_results:  
        return {'message': 'No successful results for performance analysis'}  
    
    # 这里可以添加更多性能分析  
    # 比如生成时间、文件大小等（如果有这些数据）  
    
    performance_data = {  
        'total_successful': len(successful_results),  
        'average_success_rate': len(successful_results) / len(results) if results else 0  
    }  
    
    return performance_data  

def generate_visualizations(results, evaluation_report, output_dir):  
    """生成可视化图表"""  
    import matplotlib  
    matplotlib.use('Agg')  # 使用非交互式后端  
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  
    
    # 1. 成功率饼图  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
    
    # 成功/失败分布  
    success_data = [  
        evaluation_report['summary']['successful'],  
        evaluation_report['summary']['failed']  
    ]  
    labels = ['Successful', 'Failed']  
    colors = ['#28a745', '#dc3545']  
    
    ax1.pie(success_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)  
    ax1.set_title('Generation Success Rate')  
    
    # 失败原因分布  
    if evaluation_report['failure_analysis']['reasons']:  
        failure_reasons = evaluation_report['failure_analysis']['reasons']  
        ax2.bar(range(len(failure_reasons)), list(failure_reasons.values()))  
        ax2.set_xticks(range(len(failure_reasons)))  
        ax2.set_xticklabels(list(failure_reasons.keys()), rotation=45, ha='right')  
        ax2.set_title('Failure Reasons Distribution')  
        ax2.set_ylabel('Count')  
    else:  
        ax2.text(0.5, 0.5, 'No failures to analyze', ha='center', va='center', transform=ax2.transAxes)  
        ax2.set_title('Failure Reasons Distribution')  
    
    plt.tight_layout()  
    plt.savefig(output_dir / 'success_failure_analysis.png', dpi=300, bbox_inches='tight')  
    plt.close()  
    
    # 2. 风格分布图  
    if evaluation_report['style_analysis']['total_styles']:  
        fig, ax = plt.subplots(figsize=(12, 8))  
        
        style_data = evaluation_report['style_analysis']['total_styles']  
        styles = list(style_data.keys())  
        counts = list(style_data.values())  
        
        bars = ax.bar(styles, counts, color='skyblue')  
        ax.set_title('Style Distribution')  
        ax.set_ylabel('Count')  
        ax.set_xlabel('Style Categories')  
        
        # 添加数值标签  
        for bar in bars:  
            height = bar.get_height()  
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,  
                   f'{int(height)}', ha='center', va='bottom')  
        
        plt.xticks(rotation=45, ha='right')  
        plt.tight_layout()  
        plt.savefig(output_dir / 'style_distribution.png', dpi=300, bbox_inches='tight')  
        plt.close()  

def generate_markdown_report(evaluation_report, output_dir):  
    """生成Markdown格式的评估报告"""  
    report_file = output_dir / "evaluation_report.md"  
    
    with open(report_file, 'w', encoding='utf-8') as f:  
        f.write("# Chair Style Generation Evaluation Report\n\n")  
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")  
        
        # 概要统计  
        summary = evaluation_report['summary']  
        f.write("## Summary Statistics\n\n")  
        f.write(f"- **Total Processed:** {summary['total_processed']}\n")  
        f.write(f"- **Successful:** {summary['successful']}\n")  
        f.write(f"- **Failed:** {summary['failed']}\n")  
        f.write(f"- **Success Rate:** {summary['success_rate']:.1%}\n")  
        f.write(f"- **Files Generated Successfully:** {summary['files_exist_count']}\n")  
        f.write(f"- **File Generation Rate:** {summary['file_existence_rate']:.1%}\n\n")  
        
        # 失败分析  
        f.write("## Failure Analysis\n\n")  
        failure_analysis = evaluation_report['failure_analysis']  
        
        if failure_analysis['reasons']:  
            f.write("### Failure Reasons Distribution\n\n")  
            for reason, count in sorted(failure_analysis['reasons'].items(), key=lambda x: x[1], reverse=True):  
                f.write(f"- **{reason}:** {count} cases\n")  
            
            f.write(f"\n**Most Common Failure:** {failure_analysis['most_common_failure']}\n\n")  
        else:  
            f.write("No failures detected! 🎉\n\n")  
        
        # 风格分析  
        f.write("## Style Analysis\n\n")  
        style_analysis = evaluation_report['style_analysis']  
        
        if style_analysis['total_styles']:  
            f.write("### Style Distribution\n\n")  
            f.write("| Style Category | Total Count | Success Rate |\n")  
            f.write("|----------------|-------------|-------------|\n")  
            
            for style in style_analysis['total_styles']:  
                total = style_analysis['total_styles'][style]  
                success = style_analysis['by_status']['success'].get(style, 0)  
                success_rate = success / total if total > 0 else 0  
                f.write(f"| {style} | {total} | {success_rate:.1%} |\n")  
        
        # 性能分析  
        f.write("\n## Performance Analysis\n\n")  
        perf = evaluation_report['performance_analysis']  
        
        if isinstance(perf, dict) and 'message' not in perf:  
            f.write(f"- **Successful Generations:** {perf.get('total_successful', 0)}\n")  
            f.write(f"- **Overall Success Rate:** {perf.get('average_success_rate', 0):.1%}\n")  
        else:  
            f.write("Performance data not available.\n")  
        
        # 可视化图表  
        f.write("\n## Visualizations\n\n")  
        f.write("![Success/Failure Analysis](success_failure_analysis.png)\n\n")  
        f.write("![Style Distribution](style_distribution.png)\n\n")  
        
        # 建议和结论  
        f.write("## Recommendations\n\n")  
        
        if summary['success_rate'] < 0.8:  
            f.write("- 📈 **Improve Success Rate:** Current success rate is below 80%. Consider:\n")  
            f.write("  - Checking model configuration\n")  
            f.write("  - Optimizing input preprocessing\n")  
            f.write("  - Reviewing error patterns\n\n")  
        
        if summary['file_existence_rate'] < summary['success_rate']:  
            f.write("- 💾 **File Generation Issues:** Some successful generations didn't produce output files.\n")  
            f.write("  - Check file writing permissions\n")  
            f.write("  - Verify Blender script execution\n\n")  
        
        if failure_analysis.get('most_common_failure'):  
            f.write(f"- 🔧 **Address Common Failures:** Focus on resolving '{failure_analysis['most_common_failure']}' errors.\n\n")  
        
        f.write("## Conclusion\n\n")  
        
        if summary['success_rate'] >= 0.9:  
            f.write("🎉 **Excellent Performance!** The model is generating high-quality results with minimal failures.\n")  
        elif summary['success_rate'] >= 0.7:  
            f.write("✅ **Good Performance:** The model is working well with room for minor improvements.\n")  
        elif summary['success_rate'] >= 0.5:  
            f.write("⚠️ **Moderate Performance:** The model shows potential but needs optimization.\n")  
        else:  
            f.write("❌ **Needs Improvement:** Significant issues detected that require attention.\n")  
    
    print(f"📄 Markdown report generated: {report_file}")  

if __name__ == "__main__":  
    main()