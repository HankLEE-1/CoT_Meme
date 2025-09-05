import time
import re
import os
from datetime import datetime

def monitor_training_progress(log_file='training.log'):
    """监控训练进度"""
    print("=== MemeCLIP 训练监控 ===")
    print("按 Ctrl+C 退出监控\n")
    
    if not os.path.exists(log_file):
        print(f"❌ 日志文件 {log_file} 不存在")
        print("\n💡 请先运行训练脚本生成日志文件：")
        print("   python main.py")
        print("\n⏳ 等待日志文件生成...")
        
        # 等待日志文件生成
        while not os.path.exists(log_file):
            print(".", end="", flush=True)
            time.sleep(2)
        
        print("\n✅ 日志文件已生成，开始监控...\n")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            # 移到文件末尾
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # 过滤重要信息
                    if any(keyword in line for keyword in ['准确率', 'AUROC', 'F1分数', '损失', '轮数', '完成', '开始', '样本数量']):
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                else:
                    time.sleep(1)
    except KeyboardInterrupt:
        print("\n监控已停止")
    except Exception as e:
        print(f"监控出错: {e}")

# ... existing code ...