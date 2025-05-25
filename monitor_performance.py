# monitor_performance.py  
#!/usr/bin/env python3  
"""  
性能监控脚本  
"""  

import psutil  
import GPUtil  
import time  
import json  
import argparse  
from pathlib import Path  
from datetime import datetime  
import threading  
import queue  

class PerformanceMonitor:  
    def __init__(self, output_file="performance_log.json", interval=5):  
        self.output_file = Path(output_file)  
        self.interval = interval  
        self.monitoring = False  
        self.data_queue = queue.Queue()  
        
    def start_monitoring(self):  
        """开始监控"""  
        self.monitoring = True  
        monitor_thread = threading.Thread(target=self._monitor_loop)  
        monitor_thread.daemon = True  
        monitor_thread.start()  
        
        writer_thread = threading.Thread(target=self._write_loop)  
        writer_thread.daemon = True  
        writer_thread.start()  
        
        print(f"📊 Performance monitoring started (interval: {self.interval}s)")  
        print(f"📝 Logging to: {self.output_file}")  
    
    def stop_monitoring(self):  
        """停止监控"""  
        self.monitoring = False  
        print("📊 Performance monitoring stopped")  
    
    def _monitor_loop(self):  
        """监控循环"""  
        while self.monitoring:  
            try:  
                # CPU信息  
                cpu_percent = psutil.cpu_percent(interval=1)  
                cpu_count = psutil.cpu_count()  
                
                # 内存信息  
                memory = psutil.virtual_memory()  
                memory_percent = memory.percent  
                memory_used_gb = memory.used / (1024**3)  
                memory_total_gb = memory.total / (1024**3)  
                
                # GPU信息  
                gpu_info = []  
                try:  
                    gpus = GPUtil.getGPUs()  
                    for gpu in gpus:  
                        gpu_info.append({  
                            'id': gpu.id,  
                            'name': gpu.name,  
                            'load': gpu.load * 100,  
                            'memory_used': gpu.memoryUsed,  
                            'memory_total': gpu.memoryTotal,  
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,  
                            'temperature': gpu.temperature  
                        })  
                except:  
                    gpu_info = []  
                
                # 磁盘信息  
                disk = psutil.disk_usage('/')  
                disk_percent = disk.percent  
                
                # 网络信息  
                network = psutil.net_io_counters()  
                
                data_point = {  
                    'timestamp': datetime.now().isoformat(),  
                    'cpu': {  
                        'percent': cpu_percent,  
                        'count': cpu_count  
                    },  
                    'memory': {  
                        'percent': memory_percent,  
                        'used_gb': memory_used_gb,  
                        'total_gb': memory_total_gb  
                    },  
                    'gpu': gpu_info,  
                    'disk': {  
                        'percent': disk_percent  
                    },  
                    'network': {  
                        'bytes_sent': network.bytes_sent,  
                        'bytes_recv': network.bytes_recv  
                    }  
                }  
                
                self.data_queue.put(data_point)  
                
            except Exception as e:  
                print(f"Error in monitoring: {e}")  
            
            time.sleep(self.interval)  
    
        def _write_loop(self):  
        """写入循环"""  
            data_buffer = []  
            
            while self.monitoring:  
                try:  
                    # 收集数据点  
                    while not self.data_queue.empty():  
                        data_buffer.append(self.data_queue.get())  
                    
                    # 每10个数据点或每分钟写入一次  
                    if len(data_buffer) >= 10:  
                        self._write_data(data_buffer)  
                        data_buffer = []  
                    
                    time.sleep(10)  # 每10秒检查一次  
                    
                except Exception as e:  
                    print(f"Error in writing: {e}")  
            
            # 监控结束时写入剩余数据  
            if data_buffer:  
                self._write_data(data_buffer)  
        
        def _write_data(self, data_buffer):  
            """写入数据到文件"""  
            try:  
                # 如果文件存在，读取现有数据  
                if self.output_file.exists():  
                    with open(self.output_file, 'r') as f:  
                        existing_data = json.load(f)  
                else:  
                    existing_data = []  
                
                # 添加新数据  
                existing_data.extend(data_buffer)  
                
                # 写回文件  
                with open(self.output_file, 'w') as f:  
                    json.dump(existing_data, f, indent=2)  
                    
            except Exception as e:  
                print(f"Error writing data: {e}")  
        
        def get_current_stats(self):  
            """获取当前系统状态"""  
            cpu_percent = psutil.cpu_percent(interval=1)  
            memory = psutil.virtual_memory()  
            
            gpu_info = []  
            try:  
                gpus = GPUtil.getGPUs()  
                for gpu in gpus:  
                    gpu_info.append({  
                        'name': gpu.name,  
                        'load': f"{gpu.load * 100:.1f}%",  
                        'memory': f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({(gpu.memoryUsed/gpu.memoryTotal)*100:.1f}%)",  
                        'temperature': f"{gpu.temperature}°C"  
                    })  
            except:  
                gpu_info = [{'name': 'No GPU detected'}]  
            
            return {  
                'cpu': f"{cpu_percent:.1f}%",  
                'memory': f"{memory.percent:.1f}% ({memory.used/(1024**3):.1f}GB/{memory.total/(1024**3):.1f}GB)",  
                'gpu': gpu_info  
            }  

def generate_performance_report(log_file, output_dir):  
    """生成性能报告"""  
    import matplotlib.pyplot as plt  
    import pandas as pd  
    
    log_path = Path(log_file)  
    if not log_path.exists():  
        print(f"❌ Log file not found: {log_file}")  
        return  
    
    with open(log_path) as f:  
        data = json.load(f)  
    
    if not data:  
        print("❌ No data in log file")  
        return  
    
    output_path = Path(output_dir)  
    output_path.mkdir(exist_ok=True)  
    
    # 转换为DataFrame  
    timestamps = [item['timestamp'] for item in data]  
    cpu_usage = [item['cpu']['percent'] for item in data]  
    memory_usage = [item['memory']['percent'] for item in data]  
    
    df = pd.DataFrame({  
        'timestamp': pd.to_datetime(timestamps),  
        'cpu_percent': cpu_usage,  
        'memory_percent': memory_usage  
    })  
    
    # 生成图表  
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  
    fig.suptitle('System Performance Report', fontsize=16)  
    
    # CPU使用率  
    axes[0, 0].plot(df['timestamp'], df['cpu_percent'], color='blue', alpha=0.7)  
    axes[0, 0].set_title('CPU Usage Over Time')  
    axes[0, 0].set_ylabel('CPU Usage (%)')  
    axes[0, 0].grid(True, alpha=0.3)  
    axes[0, 0].tick_params(axis='x', rotation=45)  
    
    # 内存使用率  
    axes[0, 1].plot(df['timestamp'], df['memory_percent'], color='green', alpha=0.7)  
    axes[0, 1].set_title('Memory Usage Over Time')  
    axes[0, 1].set_ylabel('Memory Usage (%)')  
    axes[0, 1].grid(True, alpha=0.3)  
    axes[0, 1].tick_params(axis='x', rotation=45)  
    
    # GPU使用率（如果有数据）  
    gpu_loads = []  
    gpu_memory = []  
    for item in data:  
        if item['gpu']:  
            gpu_loads.append(item['gpu'][0].get('load', 0))  
            gpu_memory.append(item['gpu'][0].get('memory_percent', 0))  
        else:  
            gpu_loads.append(0)  
            gpu_memory.append(0)  
    
    axes[1, 0].plot(df['timestamp'], gpu_loads, color='red', alpha=0.7)  
    axes[1, 0].set_title('GPU Load Over Time')  
    axes[1, 0].set_ylabel('GPU Load (%)')  
    axes[1, 0].grid(True, alpha=0.3)  
    axes[1, 0].tick_params(axis='x', rotation=45)  
    
    axes[1, 1].plot(df['timestamp'], gpu_memory, color='orange', alpha=0.7)  
    axes[1, 1].set_title('GPU Memory Over Time')  
    axes[1, 1].set_ylabel('GPU Memory (%)')  
    axes[1, 1].grid(True, alpha=0.3)  
    axes[1, 1].tick_params(axis='x', rotation=45)  
    
    plt.tight_layout()  
    plt.savefig(output_path / 'performance_charts.png', dpi=300, bbox_inches='tight')  
    plt.close()  
    
    # 生成统计报告  
    stats = {  
        'duration': str(df['timestamp'].max() - df['timestamp'].min()),  
        'cpu_stats': {  
            'mean': float(df['cpu_percent'].mean()),  
            'max': float(df['cpu_percent'].max()),  
            'min': float(df['cpu_percent'].min()),  
            'std': float(df['cpu_percent'].std())  
        },  
        'memory_stats': {  
            'mean': float(df['memory_percent'].mean()),  
            'max': float(df['memory_percent'].max()),  
            'min': float(df['memory_percent'].min()),  
            'std': float(df['memory_percent'].std())  
        }  
    }  
    
    if gpu_loads and any(gpu_loads):  
        stats['gpu_stats'] = {  
            'load_mean': float(pd.Series(gpu_loads).mean()),  
            'load_max': float(pd.Series(gpu_loads).max()),  
            'memory_mean': float(pd.Series(gpu_memory).mean()),  
            'memory_max': float(pd.Series(gpu_memory).max())  
        }  
    
    with open(output_path / 'performance_stats.json', 'w') as f:  
        json.dump(stats, f, indent=2)  
    
    # 生成Markdown报告  
    with open(output_path / 'performance_report.md', 'w') as f:  
        f.write("# Performance Report\n\n")  
        f.write(f"**Duration:** {stats['duration']}\n\n")  
        
        f.write("## CPU Usage\n")  
        f.write(f"- **Average:** {stats['cpu_stats']['mean']:.1f}%\n")  
        f.write(f"- **Peak:** {stats['cpu_stats']['max']:.1f}%\n")  
        f.write(f"- **Minimum:** {stats['cpu_stats']['min']:.1f}%\n\n")  
        
        f.write("## Memory Usage\n")  
        f.write(f"- **Average:** {stats['memory_stats']['mean']:.1f}%\n")  
        f.write(f"- **Peak:** {stats['memory_stats']['max']:.1f}%\n")  
        f.write(f"- **Minimum:** {stats['memory_stats']['min']:.1f}%\n\n")  
        
        if 'gpu_stats' in stats:  
            f.write("## GPU Usage\n")  
            f.write(f"- **Average Load:** {stats['gpu_stats']['load_mean']:.1f}%\n")  
            f.write(f"- **Peak Load:** {stats['gpu_stats']['load_max']:.1f}%\n")  
            f.write(f"- **Average Memory:** {stats['gpu_stats']['memory_mean']:.1f}%\n")  
            f.write(f"- **Peak Memory:** {stats['gpu_stats']['memory_max']:.1f}%\n\n")  
        
        f.write("## Charts\n\n")  
        f.write("![Performance Charts](performance_charts.png)\n")  
    
    print(f"📊 Performance report generated: {output_path}")  

def main():  
    parser = argparse.ArgumentParser(description='System Performance Monitor')  
    parser.add_argument('--action', choices=['start', 'report', 'status'], default='start',  
                    help='Action to perform')  
    parser.add_argument('--log_file', default='performance_log.json',  
                    help='Performance log file path')  
    parser.add_argument('--output_dir', default='./performance_report',  
                    help='Output directory for reports')  
    parser.add_argument('--interval', type=int, default=5,  
                    help='Monitoring interval in seconds')  
    
    args = parser.parse_args()  
    
    if args.action == 'start':  
        monitor = PerformanceMonitor(args.log_file, args.interval)  
        monitor.start_monitoring()  
        
        try:  
            print("Press Ctrl+C to stop monitoring...")  
            while True:  
                time.sleep(1)  
        except KeyboardInterrupt:  
            monitor.stop_monitoring()  
            print("\n✅ Monitoring stopped")  
            
    elif args.action == 'report':  
        generate_performance_report(args.log_file, args.output_dir)  
        
    elif args.action == 'status':  
        monitor = PerformanceMonitor()  
        stats = monitor.get_current_stats()  
        
        print("📊 Current System Status:")  
        print(f"🖥️  CPU: {stats['cpu']}")  
        print(f"💾 Memory: {stats['memory']}")  
        for i, gpu in enumerate(stats['gpu']):  
            print(f"🎮 GPU {i}: {gpu['name']}")  
            if 'load' in gpu:  
                print(f"   Load: {gpu['load']}")  
                print(f"   Memory: {gpu['memory']}")  
                print(f"   Temperature: {gpu['temperature']}")  

if __name__ == "__main__":  
    main()