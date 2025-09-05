import torch
import logging
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_cot_module import LightweightCoTModule
from dimension_utils import check_tensor_dimensions, fix_batch_dimensions, safe_tensor_cat
from device_utils import check_batch_device_consistency, fix_batch_device_consistency
from configs import cfg

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_complete_pipeline():
    """测试完整的训练流程"""
    logging.info("=== 测试完整训练流程 ===")
    
    try:
        # 1. 创建CoT模块
        cot_module = LightweightCoTModule(cfg)
        logging.info("✓ CoT模块创建成功")
        
        # 2. 创建模拟批次数据
        batch = {
            'image_features': torch.randn(2, 512),
            'text_features': torch.randn(2, 512),
            'labels': torch.randint(0, 2, (2,)),
            'image_descriptions': ['A person in an image', 'Another person in image'],
            'text_contents': ['Sample text 1', 'Sample text 2']
        }
        logging.info("✓ 模拟批次数据创建成功")
        
        # 3. 检查批次维度
        logging.info("原始批次:")
        check_tensor_dimensions(batch, "原始批次")
        
        # 4. 修复批次维度
        fixed_batch = fix_batch_dimensions(batch)
        logging.info("修复后批次:")
        check_tensor_dimensions(fixed_batch, "修复后批次")
        
        # 5. 检查设备一致性
        if check_batch_device_consistency(fixed_batch):
            logging.info("✓ 批次设备一致性检查通过")
        else:
            logging.warning("⚠ 批次设备不一致，正在修复...")
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            fixed_batch = fix_batch_device_consistency(fixed_batch, target_device)
        
        # 6. 测试CoT推理
        reasoning_features_list = []
        reasoning_texts = []
        
        for i in range(len(fixed_batch['image_features'])):
            img_feat = fixed_batch['image_features'][i:i+1]
            txt_feat = fixed_batch['text_features'][i:i+1]
            img_desc = fixed_batch['image_descriptions'][i]
            txt_content = fixed_batch['text_contents'][i]
            
            with torch.no_grad():
                reasoning_feat, reasoning_text = cot_module(
                    img_feat, txt_feat, img_desc, txt_content, "hate"
                )
            
            reasoning_features_list.append(reasoning_feat)
            reasoning_texts.append(reasoning_text)
        
        # 7. 连接推理特征
        reasoning_features = safe_tensor_cat(reasoning_features_list, dim=0)
        logging.info(f"✓ 推理特征连接成功: {reasoning_features.shape}")
        
        # 8. 模拟分类器输出
        original_features = torch.randn(2, 1024)  # 模拟原始特征
        combined_features = torch.cat([original_features, reasoning_features], dim=1)
        logging.info(f"✓ 特征融合成功: {combined_features.shape}")
        
        # 9. 模拟分类
        logits = torch.randn(2, 2)  # 模拟分类器输出
        predictions = torch.argmax(logits, dim=1)
        logging.info(f"✓ 分类完成: {predictions}")
        
        logging.info("🎉 完整训练流程测试通过!")
        return True
        
    except Exception as e:
        logging.error(f"❌ 完整训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    logging.info("=== 测试错误处理 ===")
    
    try:
        # 测试空张量列表
        try:
            result = safe_tensor_cat([], dim=0)
            logging.error("❌ 应该抛出异常但没有")
            return False
        except ValueError:
            logging.info("✓ 空张量列表正确处理")
        
        # 测试设备不匹配
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(2, 3)
        if torch.cuda.is_available():
            tensor2 = tensor2.cuda()
        
        try:
            result = safe_tensor_cat([tensor1, tensor2], dim=0)
            logging.info("✓ 设备不匹配自动修复")
        except Exception as e:
            logging.error(f"❌ 设备不匹配处理失败: {e}")
            return False
        
        # 测试维度不匹配
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(3)  # 1D张量
        
        try:
            result = safe_tensor_cat([tensor1, tensor2.unsqueeze(0)], dim=0)
            logging.info("✓ 维度不匹配自动修复")
        except Exception as e:
            logging.error(f"❌ 维度不匹配处理失败: {e}")
            return False
        
        logging.info("🎉 错误处理测试通过!")
        return True
        
    except Exception as e:
        logging.error(f"❌ 错误处理测试失败: {e}")
        return False

def test_performance():
    """测试性能"""
    logging.info("=== 测试性能 ===")
    
    try:
        import time
        
        # 创建CoT模块
        cot_module = LightweightCoTModule(cfg)
        
        # 创建测试数据
        batch_size = 10
        test_image_features = torch.randn(batch_size, 512)
        test_text_features = torch.randn(batch_size, 512)
        test_descriptions = [f"Description {i}" for i in range(batch_size)]
        test_contents = [f"Content {i}" for i in range(batch_size)]
        
        # 测试推理速度
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(batch_size):
                reasoning_feat, reasoning_text = cot_module(
                    test_image_features[i:i+1],
                    test_text_features[i:i+1],
                    test_descriptions[i],
                    test_contents[i],
                    "hate"
                )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        logging.info(f"✓ 推理速度: {inference_time:.4f}秒 ({batch_size}个样本)")
        logging.info(f"✓ 平均推理时间: {inference_time/batch_size:.4f}秒/样本")
        
        # 测试内存使用
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        logging.info(f"✓ 内存使用: {memory_usage:.2f} MB")
        
        logging.info("🎉 性能测试通过!")
        return True
        
    except Exception as e:
        logging.error(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    logging.info("🚀 开始集成测试...")
    
    # 运行所有测试
    tests = [
        test_complete_pipeline,
        test_error_handling,
        test_performance
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
        logging.info("🎉 所有集成测试通过! 可以开始训练了!")
        return True
    else:
        logging.error("❌ 部分集成测试失败! 请检查错误信息")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 