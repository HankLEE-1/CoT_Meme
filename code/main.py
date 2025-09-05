import argparse
import random
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import Custom_Collator, load_dataset
from MemeCLIP import create_model, MemeCLIP
from configs import cfg
import os
import torchmetrics
from tqdm import tqdm
from chinese_utils import ChineseProgressBar, print_model_summary_chinese
import logging
import time
from datetime import datetime

torch.use_deterministic_algorithms(False)

import logging
import time
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def main(cfg):
    logging.info("=== 开始MemeCLIP训练 ===")
    logging.info(f"训练配置: 批次大小={cfg.batch_size}, 学习率={cfg.lr}, 最大轮数={cfg.max_epochs}")
    
    seed_everything(cfg.seed, workers=True)

    dataset_train = load_dataset(cfg=cfg, split='train')
    dataset_val = load_dataset(cfg=cfg, split='val')
    dataset_test = load_dataset(cfg=cfg, split='test')

    logging.info(f"训练样本数量: {len(dataset_train)}")
    logging.info(f"验证样本数量: {len(dataset_val)}")
    logging.info(f"测试样本数量: {len(dataset_test)}")

    collator = Custom_Collator(cfg)

    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0)
    val_loader = DataLoader(dataset_test, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size,
                             collate_fn=collator, num_workers=0)
    
    model = create_model(cfg)
    
    num_params = {f'参数_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    
    monitor = "验证/AUROC"
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_path, filename='model',
                                          monitor=monitor, mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)
    
    # 在文件顶部添加导入
    from chinese_utils import ChineseProgressBar, print_model_summary_chinese
    
    # 在创建trainer时添加中文进度条
    trainer = Trainer(
        accelerator='gpu', 
        devices=cfg.gpus, 
        max_epochs=cfg.max_epochs, 
        callbacks=[checkpoint_callback, ChineseProgressBar()], 
        deterministic=False
    )
    
    # 在创建模型后添加中文摘要
    model = create_model(cfg)
    print_model_summary_chinese(model)
    
    logging.info("开始训练模型...")
    start_time = time.time()
    
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        training_time = time.time() - start_time
        logging.info(f"训练完成！总耗时: {training_time/3600:.2f} 小时")
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        raise
    
    logging.info("加载最佳模型进行测试...")
    model = MemeCLIP.load_from_checkpoint(checkpoint_path=cfg.checkpoint_file, cfg=cfg)
    logging.info("开始测试模型...")
    trainer.test(model, dataloaders=test_loader)
    logging.info("测试完成！")
    
    logging.info("=== 训练流程全部完成 ===")
    
    print("测试完成！")

if __name__ == '__main__':
      main(cfg)

