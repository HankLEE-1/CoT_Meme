# optimized_training.py
#!/usr/bin/env python3
"""
优化的训练脚本，充分利用80GB显存
"""

import torch
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

def create_optimized_trainer(cfg):
    """创建优化的trainer"""
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_path,
        filename='optimized_model',
        monitor="验证/AUROC",
        mode='max',
        save_top_k=1,
        save_last=True
    )
    
    # 优化的训练器配置
    trainer_kwargs = {
        'max_epochs': cfg.max_epochs,
        'callbacks': [checkpoint_callback],
        'accelerator': 'gpu',
        'devices': cfg.gpus,
        'precision': 16,  # 混合精度训练
        'strategy': 'auto',
        'accumulate_grad_batches': 2,  # 梯度累积
        'gradient_clip_val': 1.0,
        'log_every_n_steps': 10,
        'deterministic': False,
        'enable_progress_bar': True,
    }
    
    # 根据显存大小调整批次大小
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        if total_memory >= 80:  # 80GB显存
            cfg.batch_size = 128
            trainer_kwargs['accumulate_grad_batches'] = 1
            logging.info("检测到80GB+显存，使用大批次训练")
        elif total_memory >= 40:  # 40GB显存
            cfg.batch_size = 64
            trainer_kwargs['accumulate_grad_batches'] = 2
            logging.info("检测到40GB+显存，使用中等批次训练")
        else:  # 较小显存
            cfg.batch_size = 32
            trainer_kwargs['accumulate_grad_batches'] = 4
            logging.info("检测到较小显存，使用小批次+梯度累积")
    
    trainer = Trainer(**trainer_kwargs)
    return trainer

def optimized_training(cfg):
    """优化的训练函数"""
    logging.info("=== 开始优化训练 ===")
    logging.info(f"批次大小: {cfg.batch_size}")
    logging.info(f"学习率: {cfg.lr}")
    logging.info(f"最大轮数: {cfg.max_epochs}")
    
    # 创建优化的trainer
    trainer = create_optimized_trainer(cfg)
    
    # 加载数据和模型
    # ... (使用现有的数据加载和模型创建代码)
    
    # 开始训练
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    logging.info("=== 优化训练完成 ===")