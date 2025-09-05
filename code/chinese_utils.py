from tqdm import tqdm
import pytorch_lightning as pl

class ChineseProgressBar(pl.callbacks.TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("训练中")
        return bar
    
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("验证中")
        return bar
    
    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.set_description("测试中")
        return bar

def print_model_summary_chinese(model):
    """打印中文模型摘要"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 模型摘要 ===")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print(f"模型大小估计: {total_params * 4 / 1024 / 1024:.1f} MB")
    print("=" * 50)