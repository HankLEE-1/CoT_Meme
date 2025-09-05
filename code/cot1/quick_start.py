#!/usr/bin/env python3
"""
MemeCLIP with CoT 快速启动脚本
按顺序运行所有测试，确保系统正常工作
"""

import subprocess
import sys
import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_test(test_name: str, command: str) -> bool:
    """运行测试并返回结果"""
    logging.info(f"🧪 运行测试: {test_name}")
    logging.info(f"命令: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            logging.info(f"✅ {test_name} 通过")
            return True
        else:
            logging.error(f"❌ {test_name} 失败")
            logging.error(f"错误输出: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"❌ {test_name} 执行失败: {e}")
        return False

def main():
    """主函数"""
    logging.info("🚀 开始MemeCLIP with CoT 快速启动测试...")
    
    # 测试列表
    tests = [
        ("维度调试测试", "python debug_dimensions.py"),
        ("设备调试测试", "python debug_device.py"),
        ("集成测试", "python test_integration.py"),
        ("CoT模块测试", "python simple_cot_training.py --mode test_cot"),
        ("数据集测试", "python simple_cot_training.py --mode test_dataset")
    ]
    
    results = []
    for test_name, command in tests:
        success = run_test(test_name, command)
        results.append(success)
        
        if not success:
            logging.error(f"❌ {test_name} 失败，停止测试")
            break
    
    # 总结结果
    passed = sum(results)
    total = len(results)
    
    logging.info(f"📊 测试总结: {passed}/{total} 通过")
    
    if passed == total:
        logging.info("🎉 所有测试通过! 系统准备就绪!")
        logging.info("💡 现在可以开始训练:")
        logging.info("   python simple_cot_training.py --mode train")
        return True
    else:
        logging.error("❌ 部分测试失败! 请检查错误信息")
        logging.info("💡 故障排除建议:")
        logging.info("   1. 检查依赖包是否正确安装")
        logging.info("   2. 检查配置文件路径是否正确")
        logging.info("   3. 查看详细的错误日志")
        logging.info("   4. 参考 troubleshooting_guide.md")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 