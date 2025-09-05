#!/usr/bin/env python3
"""
GPU优化的MemeCLIP with CoT训练脚本
包含GPU检测、内存优化、混合精度训练等功能
"""

import argparse
import random
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from enhanced_datasets import Enhanced_Custom_Collator, load_enhanced_dataset
from MemeCLIP_with_CoT import create_model_with_cot
from configs import cfg
import os
import logging
import time
from datetime import datetime
import psutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_optimized_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def check_system_resources():
    """检查系统资源"""
    logging.info("=== 系统资源检查 ===")
    
    # CPU信息
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logging.info(f"CPU核心数: {cpu_count}")
    logging.info(f"CPU使用率: {cpu_percent}%")
    logging.info(f"内存总量: {memory.total / 1024**3:.1f} GB")
    logging.info(f"内存使用率: {memory.percent}%")
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logging.info(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logging.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 检查GPU内存使用
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"  GPU {i} 内存使用: {memory_allocated:.2f} GB (已分配) / {memory_reserved:.2f} GB (已保留)")
    
    return torch.cuda.is_available()

def optimize_gpu_settings():
    """优化GPU设置"""
    logging.info("=== GPU优化设置 ===")
    
    if torch.cuda.is_available():
        # 启用TF32（如果支持）
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("启用TF32优化")
        
        # 设置cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logging.info("启用cuDNN benchmark")
        
        # 设置CUDA环境变量
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        logging.info("设置CUDA环境变量")
        
        return True
    else:
        logging.warning("GPU不可用，将使用CPU训练")
        return False

def create_optimized_trainer(cfg, gpu_available):
    """创建优化的trainer"""
    logging.info("=== 创建优化Trainer ===")
    
    # 检查点回调
    monitor = "验证/AUROC"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_path, 
        filename='gpu_optimized_model',
        monitor=monitor, 
        mode='max', 
        verbose=True, 
        save_weights_only=True,
        save_top_k=1, 
        save_last=True
    )
    
    # 训练器配置
    trainer_kwargs = {
        'max_epochs': cfg.max_epochs,
        'callbacks': [checkpoint_callback],
        'deterministic': False,
        'enable_progress_bar': True,
        'log_every_n_steps': 10,
    }
    
    if gpu_available:
        # GPU优化配置
        trainer_kwargs.update({
            'accelerator': 'gpu',
            'devices': cfg.gpus,
            'precision': 16,  # 混合精度训练
            'strategy': 'auto',  # 自动选择策略
            'accumulate_grad_batches': 2,  # 梯度累积
            'gradient_clip_val': 1.0,  # 梯度裁剪
        })
        logging.info("使用GPU优化配置")
    else:
        # CPU配置
        trainer_kwargs.update({
            'accelerator': 'cpu',
            'devices': None,
            'precision': 32,
        })
        logging.info("使用CPU配置")
    
    trainer = Trainer(**trainer_kwargs)
    return trainer

def gpu_optimized_training(cfg):
    """GPU优化的训练函数"""
    logging.info("=== 开始GPU优化训练 ===")
    logging.info(f"训练配置: 批次大小={cfg.batch_size}, 学习率={cfg.lr}, 最大轮数={cfg.max_epochs}")
    
    # 设置随机种子
    seed_everything(cfg.seed, workers=True)
    
    try:
        # 检查系统资源
        gpu_available = check_system_resources()
        
        # 优化GPU设置
        optimize_gpu_settings()
        
        # 加载数据集
        logging.info("加载数据集...")
        dataset_train = load_enhanced_dataset(cfg=cfg, split='train')
        dataset_val = load_enhanced_dataset(cfg=cfg, split='val')
        dataset_test = load_enhanced_dataset(cfg=cfg, split='test')

        logging.info(f"训练样本数量: {len(dataset_train)}")
        logging.info(f"验证样本数量: {len(dataset_val)}")
        logging.info(f"测试样本数量: {len(dataset_test)}")

        # 创建数据加载器
        collator = Enhanced_Custom_Collator(cfg)
        
        # 根据GPU可用性调整worker数量
        num_workers = 4 if gpu_available else 0
        
        train_loader = DataLoader(
            dataset_train, 
            batch_size=cfg.batch_size, 
            shuffle=True,
            collate_fn=collator, 
            num_workers=num_workers,
            pin_memory=gpu_available
        )
        val_loader = DataLoader(
            dataset_val, 
            batch_size=cfg.batch_size, 
            collate_fn=collator, 
            num_workers=num_workers,
            pin_memory=gpu_available
        )
        test_loader = DataLoader(
            dataset_test, 
            batch_size=cfg.batch_size,
            collate_fn=collator, 
            num_workers=num_workers,
            pin_memory=gpu_available
        )
        
        # 创建模型
        logging.info("创建模型...")
        model = create_model_with_cot(cfg)
        
        # 创建优化的trainer
        trainer = create_optimized_trainer(cfg, gpu_available)
        
        # 开始训练
        logging.info("开始训练...")
        start_time = time.time()
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        training_time = time.time() - start_time
        logging.info(f"训练完成！总耗时: {training_time/3600:.2f} 小时")
        
        # 测试
        logging.info("开始测试...")
        trainer.test(model, dataloaders=test_loader)
        
        logging.info("=== GPU优化训练完成 ===")
        
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPU优化的MemeCLIP with CoT训练')
    parser.add_argument('--check_gpu', action='store_true', help='仅检查GPU状态')
    parser.add_argument('--optimize_only', action='store_true', help='仅进行GPU优化')
    args = parser.parse_args()
    
    if args.check_gpu:
        # 仅检查GPU
        check_system_resources()
        optimize_gpu_settings()
        return
    
    if args.optimize_only:
        # 仅优化GPU设置
        optimize_gpu_settings()
        return
    
    # 完整训练
    gpu_optimized_training(cfg)

if __name__ == '__main__':
    main() 