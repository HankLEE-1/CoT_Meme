import argparse
import random
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from enhanced_datasets import Enhanced_Custom_Collator, load_enhanced_dataset
from MemeCLIP_with_CoT import create_model_with_cot
from configs import cfg
import os
import logging
import time
from datetime import datetime

torch.use_deterministic_algorithms(False)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_cot_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def simple_cot_training(cfg):
    """简化的CoT训练函数"""
    logging.info("=== 开始简化CoT训练 ===")
    logging.info(f"训练配置: 批次大小={cfg.batch_size}, 学习率={cfg.lr}, 最大轮数={cfg.max_epochs}")
    
    seed_everything(cfg.seed, workers=True)

    try:
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
        
        monitor = "验证/AUROC"
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.checkpoint_path, 
            filename='simple_cot_model',
            monitor=monitor, 
            mode='max', 
            verbose=True, 
            save_weights_only=True,
            save_top_k=1, 
            save_last=False
        )
        
        # 检查GPU可用性
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logging.info(f"使用GPU训练，设备数量: {torch.cuda.device_count()}")
            logging.info(f"当前GPU: {torch.cuda.get_device_name()}")
        else:
            logging.warning("GPU不可用，将使用CPU训练")
        
        # 创建trainer
        trainer = Trainer(
            accelerator='gpu' if gpu_available else 'cpu', 
            devices=cfg.gpus if gpu_available else None, 
            max_epochs=cfg.max_epochs, 
            callbacks=[checkpoint_callback], 
            deterministic=False,
            precision=16 if gpu_available else 32  # 使用混合精度训练
        )
        
        logging.info("开始训练简化CoT模型...")
        start_time = time.time()
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        training_time = time.time() - start_time
        logging.info(f"训练完成！总耗时: {training_time/3600:.2f} 小时")
        
        logging.info("加载最佳模型进行测试...")
        model = MemeCLIPWithCoT.load_from_checkpoint(checkpoint_path=cfg.checkpoint_file, cfg=cfg)
        logging.info("开始测试简化CoT模型...")
        trainer.test(model, dataloaders=test_loader)
        logging.info("测试完成！")
        
        logging.info("=== 简化CoT训练流程全部完成 ===")
        
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_cot_module_only(cfg):
    """仅测试CoT模块"""
    logging.info("=== 测试CoT模块 ===")
    
    try:
        from stable_cot_module import LightweightCoTModule
        
        # 创建CoT模块
        cot_module = LightweightCoTModule(cfg)
        
        # 创建测试数据
        test_image_features = torch.randn(2, 512)
        test_text_features = torch.randn(2, 512)
        test_image_desc = ["A person in an image", "Another person in image"]
        test_text_content = ["Sample text 1", "Sample text 2"]
        test_task_type = "hate"
        
        # 测试前向传播
        with torch.no_grad():
            reasoning_features, reasoning_text = cot_module(
                test_image_features, 
                test_text_features,
                test_image_desc[0], 
                test_text_content[0], 
                test_task_type
            )
        
        logging.info(f"CoT模块测试成功!")
        logging.info(f"推理特征形状: {reasoning_features.shape}")
        logging.info(f"推理文本: {reasoning_text}")
        
    except Exception as e:
        logging.error(f"CoT模块测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_only(cfg):
    """仅测试数据集"""
    logging.info("=== 测试数据集 ===")
    
    try:
        # 测试数据集加载
        dataset_train = load_enhanced_dataset(cfg=cfg, split='train')
        logging.info(f"数据集加载成功，样本数量: {len(dataset_train)}")
        
        # 测试单个样本
        sample = dataset_train[0]
        logging.info(f"样本键: {list(sample.keys())}")
        logging.info(f"图像描述: {sample.get('image_description', 'N/A')}")
        logging.info(f"文本内容: {sample.get('text', 'N/A')}")
        
        # 测试collator
        collator = Enhanced_Custom_Collator(cfg)
        batch = collator([sample])
        logging.info(f"批处理键: {list(batch.keys())}")
        logging.info(f"图像特征形状: {batch['image_features'].shape}")
        logging.info(f"文本特征形状: {batch['text_features'].shape}")
        
        logging.info("数据集测试成功!")
        
    except Exception as e:
        logging.error(f"数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test_cot', 'test_dataset'],
                       help='运行模式: train(训练), test_cot(测试CoT模块), test_dataset(测试数据集)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        simple_cot_training(cfg)
    elif args.mode == 'test_cot':
        test_cot_module_only(cfg)
    elif args.mode == 'test_dataset':
        test_dataset_only(cfg) 