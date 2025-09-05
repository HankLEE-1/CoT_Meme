import argparse
import random
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import Custom_Collator, load_dataset
from MemeCLIP_CoT import create_model, MemeCLIP_CoT
from configs import cfg
import os
import torchmetrics
from tqdm import tqdm
from chinese_utils import ChineseProgressBar, print_model_summary_chinese
import logging
import time
from datetime import datetime
import numpy as np

torch.use_deterministic_algorithms(False)

import logging
import time
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_cot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 自定义回调类：每个epoch结束后执行测试
class EpochTestCallback(Callback):
    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader
        self.epoch_results = {}
        
    def on_train_epoch_end(self, trainer, pl_module):
        """每个训练epoch结束后执行测试"""
        current_epoch = trainer.current_epoch
        
        logging.info(f"=== Epoch {current_epoch} 训练完成，开始测试 ===")
        
        # 直接使用模型进行测试，避免调用trainer.test()
        pl_module.eval()
        
        # 重置指标
        pl_module.acc.reset()
        pl_module.auroc.reset()
        pl_module.f1.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # 将数据移动到正确的设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(pl_module.device)
                
                # 前向传播
                output = pl_module.common_step(batch)
                
                # 累积损失
                total_loss += output['loss'].item()
                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 获取累积的指标
        test_metrics = {
            '测试/损失': avg_loss,
            '测试/准确率': pl_module.acc.compute().item(),
            '测试/AUROC': pl_module.auroc.compute().item(),
            '测试/F1分数': pl_module.f1.compute().item(),
            '测试/CoT准确率': pl_module.acc.compute().item()  # CoT准确率与普通准确率相同
        }
        
        # 保存结果
        self.epoch_results[current_epoch] = test_metrics
        
        # 记录结果
        logging.info(f"Epoch {current_epoch} 测试结果:")
        for metric_name, metric_value in test_metrics.items():
            logging.info(f"  {metric_name}: {metric_value:.4f}")
        
        # 记录到tensorboard或其他日志系统
        if hasattr(trainer, 'logger') and trainer.logger is not None:
            try:
                for metric_name, metric_value in test_metrics.items():
                    trainer.logger.experiment.add_scalar(metric_name, metric_value, current_epoch)
            except Exception as e:
                logging.warning(f"记录到日志系统时出现错误: {e}")
        
        logging.info(f"=== Epoch {current_epoch} 测试完成 ===")
        
        # 恢复训练模式
        pl_module.train()

# 辅助：根据配置决定使用固定种子或时间种子
def setup_seed(seed_val):
    if seed_val is None or (isinstance(seed_val, int) and seed_val < 0):
        dynamic_seed = int(time.time()) % 2_147_483_647
        seed_everything(dynamic_seed, workers=True)
        return dynamic_seed
    else:
        seed_everything(seed_val, workers=True)
        return seed_val

def main(cfg):
    logging.info("=== 开始MemeCLIP-CoT训练 ===")
    logging.info(f"训练配置: 批次大小={cfg.batch_size}, 学习率={cfg.lr}, 最大轮数={cfg.max_epochs}")
    logging.info(f"CoT配置: 使用CoT={cfg.use_cot}, 推理步数={cfg.num_hops}, 融合维度={cfg.mfb_output_dim}")
    
    used_seed = setup_seed(getattr(cfg, 'seed', None))
    logging.info(f"本次使用种子: {used_seed}")

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
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_path, filename='model_cot',
                                          monitor=monitor, mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)
    
    # 创建自定义回调：每个epoch后测试
    epoch_test_callback = EpochTestCallback(test_loader)
    
    # 在文件顶部添加导入
    from chinese_utils import ChineseProgressBar, print_model_summary_chinese
    
    # 在创建trainer时添加中文进度条和epoch测试回调
    trainer = Trainer(
        accelerator='gpu', 
        devices=cfg.gpus, 
        max_epochs=cfg.max_epochs, 
        callbacks=[checkpoint_callback, ChineseProgressBar(), epoch_test_callback], 
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
        
        # 输出所有epoch的测试结果汇总
        logging.info("=== 所有Epoch测试结果汇总 ===")
        for epoch, results in epoch_test_callback.epoch_results.items():
            logging.info(f"Epoch {epoch}:")
            for metric_name, metric_value in results.items():
                logging.info(f"  {metric_name}: {metric_value:.4f}")
        
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        raise
    
    # 优先使用本次训练产生的最优检查点
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        ckpt_path_to_load = best_ckpt
    else:
        ckpt_path_to_load = cfg.checkpoint_file
    logging.info(f"加载用于最终测试的检查点: {ckpt_path_to_load}")
    
    model = MemeCLIP_CoT.load_from_checkpoint(checkpoint_path=ckpt_path_to_load, cfg=cfg)
    logging.info("开始最终测试模型...")
    final_test_results = trainer.test(model, dataloaders=test_loader)
    logging.info("最终测试完成！")
    
    # 记录最终测试结果
    if final_test_results:
        logging.info("最终测试结果:")
        for metric_name, metric_value in final_test_results[0].items():
            logging.info(f"  {metric_name}: {metric_value:.4f}")
    
    logging.info("=== 训练流程全部完成 ===")
    
    print("测试完成！")

if __name__ == '__main__':
      main(cfg) 