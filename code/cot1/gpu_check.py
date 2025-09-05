#!/usr/bin/env python3
"""
GPU检测和配置脚本
检查GPU可用性并配置训练环境
"""

import torch
import os
import logging
import subprocess
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_gpu_availability():
    """检查GPU可用性"""
    logging.info("=== GPU可用性检查 ===")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        logging.info(f"GPU数量: {gpu_count}")
        
        # 获取当前GPU
        current_device = torch.cuda.current_device()
        logging.info(f"当前GPU设备: {current_device}")
        
        # 获取GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            logging.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 检查当前GPU内存使用情况
        if gpu_count > 0:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logging.info(f"当前GPU内存使用: {memory_allocated:.2f} GB (已分配) / {memory_reserved:.2f} GB (已保留)")
        
        return True, gpu_count
    else:
        logging.warning("CUDA不可用，将使用CPU训练")
        return False, 0

def check_nvidia_smi():
    """使用nvidia-smi检查GPU状态"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("nvidia-smi输出:")
            print(result.stdout)
        else:
            logging.error("nvidia-smi命令失败")
    except FileNotFoundError:
        logging.warning("nvidia-smi命令不可用")

def configure_gpu_environment():
    """配置GPU环境"""
    logging.info("=== GPU环境配置 ===")
    
    # 设置CUDA环境变量
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # 检查GPU可用性
    cuda_available, gpu_count = check_gpu_availability()
    
    if cuda_available and gpu_count > 0:
        # 设置默认GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logging.info("设置CUDA_VISIBLE_DEVICES=0")
        
        # 设置CUDA性能优化
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        logging.info("设置CUBLAS_WORKSPACE_CONFIG=:4096:8")
        
        # 启用TF32（如果支持）
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("启用TF32优化")
        
        return True
    else:
        logging.warning("未检测到可用GPU，将使用CPU训练")
        return False

def test_gpu_training():
    """测试GPU训练"""
    logging.info("=== GPU训练测试 ===")
    
    try:
        # 创建测试张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")
        
        # 创建测试模型
        test_model = torch.nn.Linear(100, 10).to(device)
        test_input = torch.randn(32, 100).to(device)
        test_target = torch.randint(0, 10, (32,)).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            output = test_model(test_input)
            loss = torch.nn.functional.cross_entropy(output, test_target)
        
        logging.info(f"测试损失: {loss.item():.4f}")
        logging.info("GPU训练测试成功!")
        
        # 检查内存使用
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            logging.info(f"GPU内存使用: {memory_used:.2f} MB")
        
        return True
        
    except Exception as e:
        logging.error(f"GPU训练测试失败: {e}")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    logging.info("=== PyTorch CUDA支持检查 ===")
    
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"CUDA版本: {torch.version.cuda}")
    logging.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 检查CUDA功能
    if torch.cuda.is_available():
        logging.info("CUDA功能检查:")
        logging.info(f"  - 当前设备: {torch.cuda.current_device()}")
        logging.info(f"  - 设备数量: {torch.cuda.device_count()}")
        logging.info(f"  - 设备名称: {torch.cuda.get_device_name()}")
        logging.info(f"  - 计算能力: {torch.cuda.get_device_capability()}")
        
        # 测试基本CUDA操作
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            logging.info("  - 矩阵乘法测试: 通过")
        except Exception as e:
            logging.error(f"  - 矩阵乘法测试: 失败 - {e}")
    else:
        logging.warning("CUDA不可用")

def main():
    """主函数"""
    logging.info("🚀 开始GPU检查和配置...")
    
    # 1. 检查PyTorch CUDA支持
    check_pytorch_cuda()
    
    # 2. 检查nvidia-smi
    check_nvidia_smi()
    
    # 3. 配置GPU环境
    gpu_available = configure_gpu_environment()
    
    # 4. 测试GPU训练
    training_success = test_gpu_training()
    
    # 总结
    logging.info("=== 检查结果总结 ===")
    if gpu_available and training_success:
        logging.info("✅ GPU配置成功，可以开始训练!")
        logging.info("💡 建议的训练配置:")
        logging.info("   - 使用GPU训练")
        logging.info("   - 启用混合精度训练")
        logging.info("   - 使用适当的批次大小")
    else:
        logging.warning("⚠ GPU配置有问题，将使用CPU训练")
        logging.info("💡 CPU训练建议:")
        logging.info("   - 减小批次大小")
        logging.info("   - 减少模型复杂度")
        logging.info("   - 考虑使用更小的数据集")
    
    return gpu_available and training_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 