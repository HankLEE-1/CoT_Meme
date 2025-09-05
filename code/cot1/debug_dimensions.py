import torch
import logging
from stable_cot_module import LightweightCoTModule
from dimension_utils import check_tensor_dimensions, ensure_tensor_dimensions, safe_tensor_cat
from configs import cfg

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_tensor_dimensions():
    """测试张量维度"""
    logging.info("=== 测试张量维度 ===")
    
    try:
        # 创建CoT模块
        cot_module = LightweightCoTModule(cfg)
        
        # 创建测试数据
        batch_size = 2
        test_image_features = torch.randn(batch_size, 512)
        test_text_features = torch.randn(batch_size, 512)
        
        # 检查输入维度
        logging.info(f"图像特征维度: {test_image_features.shape}")
        logging.info(f"文本特征维度: {test_text_features.shape}")
        
        # 测试前向传播
        with torch.no_grad():
            reasoning_features, reasoning_text = cot_module(
                test_image_features, 
                test_text_features,
                "Test image description", 
                "Test text content", 
                "hate"
            )
        
        logging.info(f"推理特征维度: {reasoning_features.shape}")
        logging.info(f"推理文本: {reasoning_text}")
        
        # 测试维度修复
        test_tensor = torch.randn(3)  # 1D张量
        logging.info(f"原始张量维度: {test_tensor.shape}")
        
        fixed_tensor = ensure_tensor_dimensions(test_tensor, 2, "test_tensor")
        logging.info(f"修复后张量维度: {fixed_tensor.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"张量维度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_operations():
    """测试张量操作"""
    logging.info("=== 测试张量操作 ===")
    
    try:
        # 创建不同维度的张量
        tensor1 = torch.randn(2, 3)  # 2D
        tensor2 = torch.randn(3)     # 1D
        tensor3 = torch.randn(1, 2, 3)  # 3D
        
        logging.info(f"张量1维度: {tensor1.shape}")
        logging.info(f"张量2维度: {tensor2.shape}")
        logging.info(f"张量3维度: {tensor3.shape}")
        
        # 测试安全的张量连接
        tensors = [tensor1, tensor2.unsqueeze(0)]  # 统一为2D
        try:
            result = safe_tensor_cat(tensors, dim=0)
            logging.info(f"连接结果维度: {result.shape}")
        except Exception as e:
            logging.error(f"张量连接失败: {e}")
        
        # 测试维度修复
        tensors_fixed = []
        for i, tensor in enumerate(tensors):
            fixed = ensure_tensor_dimensions(tensor, 2, f"tensor_{i}")
            tensors_fixed.append(fixed)
            logging.info(f"修复后张量{i}维度: {fixed.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"张量操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cot_module_dimensions():
    """测试CoT模块的维度处理"""
    logging.info("=== 测试CoT模块维度 ===")
    
    try:
        cot_module = LightweightCoTModule(cfg)
        
        # 测试不同输入维度
        test_cases = [
            (torch.randn(1, 512), torch.randn(1, 512)),  # 正常情况
            (torch.randn(512), torch.randn(512)),         # 1D输入
            (torch.randn(1, 1, 512), torch.randn(1, 1, 512)),  # 3D输入
        ]
        
        for i, (img_feat, txt_feat) in enumerate(test_cases):
            logging.info(f"测试用例 {i+1}:")
            logging.info(f"  图像特征维度: {img_feat.shape}")
            logging.info(f"  文本特征维度: {txt_feat.shape}")
            
            try:
                with torch.no_grad():
                    reasoning_feat, reasoning_text = cot_module(
                        img_feat, txt_feat,
                        "Test description", "Test content", "hate"
                    )
                logging.info(f"  推理特征维度: {reasoning_feat.shape}")
                logging.info(f"  推理成功!")
            except Exception as e:
                logging.error(f"  推理失败: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"CoT模块维度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """测试批次处理"""
    logging.info("=== 测试批次处理 ===")
    
    try:
        # 创建模拟批次
        batch = {
            'image_features': torch.randn(2, 512),
            'text_features': torch.randn(2, 512),
            'labels': torch.randint(0, 2, (2,)),
            'image_descriptions': ['desc1', 'desc2'],
            'text_contents': ['text1', 'text2']
        }
        
        logging.info("原始批次:")
        check_tensor_dimensions(batch, "原始批次")
        
        # 测试维度修复
        from dimension_utils import fix_batch_dimensions
        fixed_batch = fix_batch_dimensions(batch)
        
        logging.info("修复后批次:")
        check_tensor_dimensions(fixed_batch, "修复后批次")
        
        return True
        
    except Exception as e:
        logging.error(f"批次处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logging.info("开始维度调试测试...")
    
    # 运行所有测试
    tests = [
        test_tensor_dimensions,
        test_tensor_operations,
        test_cot_module_dimensions,
        test_batch_processing
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
        logging.info("所有维度测试通过!")
    else:
        logging.error("部分维度测试失败!")

if __name__ == '__main__':
    main() 