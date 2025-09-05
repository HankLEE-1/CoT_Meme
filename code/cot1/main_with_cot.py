import argparse
import random
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from enhanced_datasets import Enhanced_Custom_Collator, load_enhanced_dataset
from MemeCLIP_with_CoT import create_model_with_cot
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
        logging.FileHandler('training_with_cot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def main_with_cot(cfg):
    logging.info("=== 开始MemeCLIP with CoT训练 ===")
    logging.info(f"训练配置: 批次大小={cfg.batch_size}, 学习率={cfg.lr}, 最大轮数={cfg.max_epochs}")
    logging.info("使用Chain of Thought推理增强")
    
    seed_everything(cfg.seed, workers=True)

    # 使用增强的数据集
    dataset_train = load_enhanced_dataset(cfg=cfg, split='train')
    dataset_val = load_enhanced_dataset(cfg=cfg, split='val')
    dataset_test = load_enhanced_dataset(cfg=cfg, split='test')

    logging.info(f"训练样本数量: {len(dataset_train)}")
    logging.info(f"验证样本数量: {len(dataset_val)}")
    logging.info(f"测试样本数量: {len(dataset_test)}")

    # 使用增强的collator
    collator = Enhanced_Custom_Collator(cfg)

    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size,
                             collate_fn=collator, num_workers=0)
    
    # 创建CoT增强的模型
    model = create_model_with_cot(cfg)
    
    num_params = {f'参数_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    
    monitor = "验证/AUROC"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_path, 
        filename='model_with_cot',
        monitor=monitor, 
        mode='max', 
        verbose=True, 
        save_weights_only=True,
        save_top_k=1, 
        save_last=False
    )
    
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
    model = create_model_with_cot(cfg)
    print_model_summary_chinese(model)
    
    logging.info("开始训练CoT增强的模型...")
    start_time = time.time()
    
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        training_time = time.time() - start_time
        logging.info(f"训练完成！总耗时: {training_time/3600:.2f} 小时")
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        raise
    
    logging.info("加载最佳模型进行测试...")
    model = MemeCLIPWithCoT.load_from_checkpoint(checkpoint_path=cfg.checkpoint_file, cfg=cfg)
    logging.info("开始测试CoT增强的模型...")
    trainer.test(model, dataloaders=test_loader)
    logging.info("测试完成！")
    
    logging.info("=== CoT增强训练流程全部完成 ===")
    
    print("CoT增强测试完成！")

def compare_models(cfg):
    """比较原始MemeCLIP和CoT增强版本的性能"""
    logging.info("=== 开始模型性能比较 ===")
    
    # 加载原始模型
    from MemeCLIP import create_model
    original_model = create_model(cfg)
    
    # 加载CoT增强模型
    cot_model = create_model_with_cot(cfg)
    
    # 加载测试数据
    dataset_test = load_enhanced_dataset(cfg=cfg, split='test')
    collator = Enhanced_Custom_Collator(cfg)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size,
                             collate_fn=collator, num_workers=0)
    
    trainer = Trainer(accelerator='gpu', devices=cfg.gpus, deterministic=False)
    
    # 测试原始模型
    logging.info("测试原始MemeCLIP模型...")
    original_results = trainer.test(original_model, dataloaders=test_loader)
    
    # 测试CoT增强模型
    logging.info("测试CoT增强MemeCLIP模型...")
    cot_results = trainer.test(cot_model, dataloaders=test_loader)
    
    # 比较结果
    logging.info("=== 性能比较结果 ===")
    logging.info(f"原始模型 - 准确率: {original_results[0]['test/准确率']:.4f}")
    logging.info(f"CoT增强模型 - 准确率: {cot_results[0]['test/准确率']:.4f}")
    logging.info(f"性能提升: {cot_results[0]['test/准确率'] - original_results[0]['test/准确率']:.4f}")

def analyze_reasoning_outputs(cfg):
    """分析CoT推理输出"""
    logging.info("=== 分析CoT推理输出 ===")
    
    # 加载模型和数据
    model = create_model_with_cot(cfg)
    dataset_test = load_enhanced_dataset(cfg=cfg, split='test')
    collator = Enhanced_Custom_Collator(cfg)
    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collator, num_workers=0)
    
    # 分析前几个样本的推理过程
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:  # 只分析前5个样本
                break
                
            output = model.common_step(batch)
            reasoning_texts = output.get('reasoning_texts', [])
            
            logging.info(f"样本 {i+1} 的推理过程:")
            for j, reasoning_text in enumerate(reasoning_texts):
                logging.info(f"  推理步骤 {j+1}: {reasoning_text[:200]}...")
            
            logging.info(f"  预测标签: {output['accuracy']}")
            logging.info("---")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'compare', 'analyze'],
                       help='运行模式: train(训练), compare(比较), analyze(分析推理)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        main_with_cot(cfg)
    elif args.mode == 'compare':
        compare_models(cfg)
    elif args.mode == 'analyze':
        analyze_reasoning_outputs(cfg) 