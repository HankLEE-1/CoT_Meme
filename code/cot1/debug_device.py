import torch
import logging
from stable_cot_module import LightweightCoTModule
from device_utils import debug_device_info, check_batch_device_consistency
from configs import cfg

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_device_consistency():
    """测试设备一致性"""
    logging.info("=== 测试设备一致性 ===")
    
    try:
        # 创建CoT模块
        cot_module = LightweightCoTModule(cfg)
        
        # 检查模块设备
        device = next(cot_module.parameters()).device
        logging.info(f"CoT模块设备: {device}")
        
        # 创建测试数据
        batch_size = 2
        test_image_features = torch.randn(batch_size, 512)
        test_text_features = torch.randn(batch_size, 512)
        
        # 确保测试数据在正确的设备上
        test_image_features = test_image_features.to(device)
        test_text_features = test_text_features.to(device)
        
        logging.info(f"图像特征设备: {test_image_features.device}")
        logging.info(f"文本特征设备: {test_text_features.device}")
        
        # 测试前向传播
        with torch.no_grad():
            reasoning_features, reasoning_text = cot_module(
                test_image_features, 
                test_text_features,
                "Test image description", 
                "Test text content", 
                "hate"
            )
        
        logging.info(f"推理特征设备: {reasoning_features.device}")
        logging.info(f"推理特征形状: {reasoning_features.shape}")
        logging.info(f"推理文本: {reasoning_text}")
        
        # 检查批次一致性
        batch = {
            'image_features': test_image_features,
            'text_features': test_text_features,
            'labels': torch.randint(0, 2, (batch_size,))
        }
        
        if check_batch_device_consistency(batch):
            logging.info("批次设备一致性检查通过!")
        else:
            logging.error("批次设备一致性检查失败!")
        
        return True
        
    except Exception as e:
        logging.error(f"设备一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_device_assignment():
    """测试模型设备分配"""
    logging.info("=== 测试模型设备分配 ===")
    
    try:
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA可用: {cuda_available}")
        
        if cuda_available:
            logging.info(f"CUDA设备数量: {torch.cuda.device_count()}")
            logging.info(f"当前CUDA设备: {torch.cuda.current_device()}")
            logging.info(f"设备名称: {torch.cuda.get_device_name()}")
        
        # 创建模块并移动到设备
        cot_module = LightweightCoTModule(cfg)
        
        # 检查默认设备
        default_device = next(cot_module.parameters()).device
        logging.info(f"默认设备: {default_device}")
        
        # 测试CPU设备
        cpu_module = cot_module.cpu()
        cpu_device = next(cpu_module.parameters()).device
        logging.info(f"CPU设备: {cpu_device}")
        
        # 测试CUDA设备（如果可用）
        if cuda_available:
            cuda_module = cot_module.cuda()
            cuda_device = next(cuda_module.parameters()).device
            logging.info(f"CUDA设备: {cuda_device}")
        
        return True
        
    except Exception as e:
        logging.error(f"模型设备分配测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_operations():
    """测试张量操作"""
    logging.info("=== 测试张量操作 ===")
    
    try:
        # 创建不同设备上的张量
        cpu_tensor = torch.randn(2, 3)
        logging.info(f"CPU张量设备: {cpu_tensor.device}")
        
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(2, 3).cuda()
            logging.info(f"CUDA张量设备: {cuda_tensor.device}")
            
            # 测试设备移动
            moved_tensor = cpu_tensor.to(cuda_tensor.device)
            logging.info(f"移动后张量设备: {moved_tensor.device}")
            
            # 测试张量连接
            try:
                combined = torch.cat([cuda_tensor, moved_tensor], dim=1)
                logging.info(f"连接成功，设备: {combined.device}")
            except RuntimeError as e:
                logging.error(f"张量连接失败: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"张量操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logging.info("开始设备调试测试...")
    
    # 运行所有测试
    tests = [
        test_device_consistency,
        test_model_device_assignment,
        test_tensor_operations
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
    logging.info(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logging.info("所有设备测试通过!")
    else:
        logging.error("部分设备测试失败!")

if __name__ == '__main__':
    main() 