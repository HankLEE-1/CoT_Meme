#!/usr/bin/env python3
"""
设备安全性测试脚本
测试设备管理器的各种功能
"""

import torch
import logging
from safe_device_utils import DeviceManager, safe_to_device, ensure_model_on_device
from stable_cot_module import LightweightCoTModule
from configs import cfg

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_device_manager():
    """测试设备管理器"""
    logging.info("=== 测试设备管理器 ===")
    
    try:
        # 创建设备管理器
        device_manager = DeviceManager()
        logging.info(f"设备管理器创建成功，设备: {device_manager.get_device()}")
        
        # 测试模型移动
        test_model = torch.nn.Linear(100, 10)
        moved_model = device_manager.ensure_model_on_device(test_model)
        logging.info(f"模型移动成功: {next(moved_model.parameters()).device}")
        
        # 测试张量移动
        test_tensor = torch.randn(10, 10)
        moved_tensor = device_manager.move_to_device(test_tensor)
        logging.info(f"张量移动成功: {moved_tensor.device}")
        
        return True
        
    except Exception as e:
        logging.error(f"设备管理器测试失败: {e}")
        return False

def test_cot_module_device():
    """测试CoT模块设备管理"""
    logging.info("=== 测试CoT模块设备管理 ===")
    
    try:
        # 创建CoT模块
        cot_module = LightweightCoTModule(cfg)
        logging.info("CoT模块创建成功")
        
        # 测试设备管理器
        device_manager = DeviceManager()
        cot_module = device_manager.ensure_model_on_device(cot_module)
        logging.info(f"CoT模块设备: {next(cot_module.parameters()).device}")
        
        # 测试前向传播
        test_image_features = torch.randn(2, 512)
        test_text_features = torch.randn(2, 512)
        
        # 移动到正确设备
        device = device_manager.get_device()
        test_image_features = test_image_features.to(device)
        test_text_features = test_text_features.to(device)
        
        with torch.no_grad():
            reasoning_features, reasoning_text = cot_module(
                test_image_features, 
                test_text_features,
                "Test image description", 
                "Test text content", 
                "hate"
            )
        
        logging.info(f"CoT模块前向传播成功，推理特征设备: {reasoning_features.device}")
        
        return True
        
    except Exception as e:
        logging.error(f"CoT模块设备管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_safe_device_functions():
    """测试安全设备函数"""
    logging.info("=== 测试安全设备函数 ===")
    
    try:
        # 测试safe_to_device
        test_tensor = torch.randn(5, 5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        moved_tensor = safe_to_device(test_tensor, device)
        logging.info(f"safe_to_device测试成功: {moved_tensor.device}")
        
        # 测试ensure_model_on_device
        test_model = torch.nn.Linear(50, 10)
        moved_model = ensure_model_on_device(test_model, device)
        logging.info(f"ensure_model_on_device测试成功: {next(moved_model.parameters()).device}")
        
        # 测试非张量对象
        non_tensor = "test_string"
        result = safe_to_device(non_tensor, device)
        logging.info(f"非张量对象测试成功: {type(result)}")
        
        return True
        
    except Exception as e:
        logging.error(f"安全设备函数测试失败: {e}")
        return False

def test_device_consistency():
    """测试设备一致性"""
    logging.info("=== 测试设备一致性 ===")
    
    try:
        device_manager = DeviceManager()
        
        # 创建测试批次
        batch = {
            'image_features': torch.randn(2, 512),
            'text_features': torch.randn(2, 512),
            'labels': torch.randint(0, 2, (2,)),
            'image_descriptions': ['desc1', 'desc2'],
            'text_contents': ['text1', 'text2']
        }
        
        # 创建测试模型
        model = torch.nn.Linear(512, 10)
        
        # 检查设备一致性
        is_consistent = device_manager.check_batch_device(batch, model)
        logging.info(f"设备一致性检查: {is_consistent}")
        
        # 修复设备一致性
        fixed_batch = device_manager.fix_batch_device(batch, model)
        logging.info("设备一致性修复完成")
        
        # 再次检查
        is_consistent_after = device_manager.check_batch_device(fixed_batch, model)
        logging.info(f"修复后设备一致性: {is_consistent_after}")
        
        return True
        
    except Exception as e:
        logging.error(f"设备一致性测试失败: {e}")
        return False

def test_pytorch_lightning_compatibility():
    """测试PyTorch Lightning兼容性"""
    logging.info("=== 测试PyTorch Lightning兼容性 ===")
    
    try:
        import pytorch_lightning as pl
        
        # 创建一个简单的Lightning模块
        class TestModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
                # 不要直接设置device属性
                # self.device = torch.device('cuda')  # 这会导致错误
            
            def forward(self, x):
                return self.linear(x)
            
            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = torch.nn.functional.mse_loss(y_hat, y)
                return loss
            
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters())
        
        # 创建模型
        model = TestModule()
        logging.info("PyTorch Lightning模块创建成功")
        
        # 测试设备管理器
        device_manager = DeviceManager()
        model = device_manager.ensure_model_on_device(model)
        logging.info(f"模型设备: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        logging.error(f"PyTorch Lightning兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    logging.info("🚀 开始设备安全性测试...")
    
    # 运行所有测试
    tests = [
        test_device_manager,
        test_cot_module_device,
        test_safe_device_functions,
        test_device_consistency,
        test_pytorch_lightning_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logging.error(f"测试 {test.__name__} 失败: {e}")
            results.append(False)
    
    # 总结结果
    passed = sum(results)
    total = len(results)
    logging.info(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logging.info("🎉 所有设备安全性测试通过!")
        logging.info("💡 现在可以安全地使用设备管理器进行训练")
    else:
        logging.error("❌ 部分设备安全性测试失败!")
        logging.info("💡 请检查错误信息并修复问题")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    import sys
    sys.exit(0 if success else 1) 