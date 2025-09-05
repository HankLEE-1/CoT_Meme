import torch
import logging
from typing import Any, Optional

def get_device() -> torch.device:
    """安全获取设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def safe_to_device(obj: Any, device: torch.device) -> Any:
    """安全地将对象移动到设备"""
    try:
        if hasattr(obj, 'to') and callable(getattr(obj, 'to')):
            return obj.to(device)
        else:
            return obj
    except Exception as e:
        logging.warning(f"无法将对象移动到设备 {device}: {e}")
        return obj

def ensure_model_on_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """确保模型在指定设备上"""
    try:
        current_device = next(model.parameters()).device
        if current_device != device:
            logging.info(f"将模型从 {current_device} 移动到 {device}")
            model = model.to(device)
        return model
    except Exception as e:
        logging.warning(f"无法移动模型到设备 {device}: {e}")
        return model

def check_tensor_device(tensor: torch.Tensor, expected_device: torch.device) -> bool:
    """检查张量是否在预期设备上"""
    return tensor.device == expected_device

def move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """安全地将张量移动到设备"""
    if tensor.device != device:
        return tensor.to(device)
    return tensor

def get_current_device() -> str:
    """获取当前设备信息"""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"

def log_device_info(model: torch.nn.Module, name: str = "模型"):
    """记录设备信息"""
    try:
        device = next(model.parameters()).device
        logging.info(f"{name} 当前设备: {device}")
    except Exception as e:
        logging.warning(f"无法获取 {name} 设备信息: {e}")

def safe_device_consistency_check(batch: dict, model: torch.nn.Module) -> bool:
    """安全地检查批次和模型的设备一致性"""
    try:
        model_device = next(model.parameters()).device
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.device != model_device:
                    logging.warning(f"批次张量 {key} 设备不匹配: {value.device} vs {model_device}")
                    return False
        
        return True
    except Exception as e:
        logging.warning(f"设备一致性检查失败: {e}")
        return True  # 如果检查失败，假设一致

def fix_device_consistency(batch: dict, model: torch.nn.Module) -> dict:
    """修复设备一致性"""
    try:
        model_device = next(model.parameters()).device
        fixed_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.device != model_device:
                    logging.info(f"修复 {key} 设备: {value.device} -> {model_device}")
                    fixed_batch[key] = value.to(model_device)
                else:
                    fixed_batch[key] = value
            else:
                fixed_batch[key] = value
        
        return fixed_batch
    except Exception as e:
        logging.warning(f"设备一致性修复失败: {e}")
        return batch

class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.device = get_device()
        logging.info(f"设备管理器初始化，使用设备: {self.device}")
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        return self.device
    
    def move_to_device(self, obj: Any) -> Any:
        """将对象移动到当前设备"""
        return safe_to_device(obj, self.device)
    
    def ensure_model_on_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """确保模型在当前设备上"""
        return ensure_model_on_device(model, self.device)
    
    def check_batch_device(self, batch: dict, model: torch.nn.Module) -> bool:
        """检查批次设备一致性"""
        return safe_device_consistency_check(batch, model)
    
    def fix_batch_device(self, batch: dict, model: torch.nn.Module) -> dict:
        """修复批次设备一致性"""
        return fix_device_consistency(batch, model)
    
    def log_device_status(self, model: torch.nn.Module, name: str = "模型"):
        """记录设备状态"""
        log_device_info(model, name)
        logging.info(f"当前设备: {self.device}")
        if torch.cuda.is_available():
            logging.info(f"GPU数量: {torch.cuda.device_count()}")
            logging.info(f"当前GPU: {torch.cuda.current_device()}") 