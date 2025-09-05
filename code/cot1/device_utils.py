import torch
import logging
from typing import Dict, Any

def check_tensor_devices(tensor_dict: Dict[str, torch.Tensor], name: str = "tensors") -> None:
    """检查张量的设备分布"""
    devices = {}
    for key, tensor in tensor_dict.items():
        device = str(tensor.device)
        if device not in devices:
            devices[device] = []
        devices[device].append(key)
    
    logging.info(f"{name} 设备分布:")
    for device, keys in devices.items():
        logging.info(f"  {device}: {keys}")

def ensure_tensors_on_device(tensor_dict: Dict[str, torch.Tensor], target_device: torch.device) -> Dict[str, torch.Tensor]:
    """确保所有张量都在目标设备上"""
    result = {}
    for key, tensor in tensor_dict.items():
        if tensor.device != target_device:
            logging.warning(f"移动张量 {key} 从 {tensor.device} 到 {target_device}")
            result[key] = tensor.to(target_device)
        else:
            result[key] = tensor
    return result

def get_model_device(model: torch.nn.Module) -> torch.device:
    """获取模型所在的设备"""
    return next(model.parameters()).device

def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """将模型移动到指定设备"""
    if get_model_device(model) != device:
        logging.info(f"移动模型到设备: {device}")
        model = model.to(device)
    return model

def check_batch_device_consistency(batch: Dict[str, Any]) -> bool:
    """检查批次中所有张量的设备一致性"""
    devices = set()
    tensor_keys = []
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            devices.add(str(value.device))
            tensor_keys.append(key)
    
    if len(devices) > 1:
        logging.error(f"批次中存在多个设备: {devices}")
        logging.error(f"张量键: {tensor_keys}")
        return False
    
    logging.info(f"批次设备一致性检查通过: {list(devices)[0] if devices else 'No tensors'}")
    return True

def fix_batch_device_consistency(batch: Dict[str, Any], target_device: torch.device) -> Dict[str, Any]:
    """修复批次中的设备不一致问题"""
    fixed_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.device != target_device:
                logging.warning(f"修复张量 {key}: {value.device} -> {target_device}")
                fixed_batch[key] = value.to(target_device)
            else:
                fixed_batch[key] = value
        else:
            fixed_batch[key] = value
    
    return fixed_batch

def debug_device_info(model: torch.nn.Module, batch: Dict[str, Any]) -> None:
    """调试设备和张量信息"""
    model_device = get_model_device(model)
    logging.info(f"模型设备: {model_device}")
    
    # 检查批次中的张量设备
    tensor_devices = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_str = str(value.device)
            if device_str not in tensor_devices:
                tensor_devices[device_str] = []
            tensor_devices[device_str].append(key)
    
    logging.info("批次张量设备分布:")
    for device, keys in tensor_devices.items():
        logging.info(f"  {device}: {keys}")
    
    # 检查是否有设备不匹配
    if len(tensor_devices) > 1:
        logging.warning("检测到设备不匹配!")
        return False
    
    return True 