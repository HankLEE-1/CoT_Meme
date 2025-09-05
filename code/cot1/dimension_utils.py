import torch
import logging
from typing import List, Tuple, Dict, Any

def check_tensor_dimensions(tensor_dict: Dict[str, Any], name: str = "tensors") -> None:
    """检查张量的维度"""
    logging.info(f"{name} 维度信息:")
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            logging.info(f"  {key}: {value.shape} (设备: {value.device})")
        else:
            logging.info(f"  {key}: {type(value).__name__} (非张量)")

def ensure_tensor_dimensions(tensor: torch.Tensor, expected_dims: int, name: str = "tensor") -> torch.Tensor:
    """确保张量具有正确的维度"""
    current_dims = tensor.dim()
    
    if current_dims < expected_dims:
        # 添加缺失的维度
        for _ in range(expected_dims - current_dims):
            tensor = tensor.unsqueeze(0)
        logging.info(f"为 {name} 添加了 {expected_dims - current_dims} 个维度: {tensor.shape}")
    elif current_dims > expected_dims:
        # 移除多余的维度
        for _ in range(current_dims - expected_dims):
            tensor = tensor.squeeze(0)
        logging.info(f"为 {name} 移除了 {current_dims - expected_dims} 个维度: {tensor.shape}")
    
    return tensor

def fix_batch_dimensions(batch: Dict[str, Any]) -> Dict[str, Any]:
    """修复批次中的维度问题"""
    fixed_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # 检查张量维度
            if value.dim() == 0:  # 标量张量
                value = value.unsqueeze(0)
                logging.warning(f"为 {key} 添加了batch维度")
            elif value.dim() == 1 and key in ['image_features', 'text_features']:
                # 这些应该是2D张量 [batch_size, features]
                value = value.unsqueeze(0)
                logging.warning(f"为 {key} 添加了batch维度")
            
            fixed_batch[key] = value
        elif isinstance(value, (list, tuple)):
            # 对于列表或元组，保持原样
            fixed_batch[key] = value
        else:
            # 对于其他类型，保持原样
            fixed_batch[key] = value
    
    return fixed_batch

def validate_tensor_shapes(tensors: Dict[str, torch.Tensor], expected_shapes: Dict[str, Tuple]) -> bool:
    """验证张量形状是否符合预期"""
    for name, expected_shape in expected_shapes.items():
        if name in tensors:
            actual_shape = tensors[name].shape
            if actual_shape != expected_shape:
                logging.error(f"张量 {name} 形状不匹配: 期望 {expected_shape}, 实际 {actual_shape}")
                return False
        else:
            logging.error(f"缺少张量: {name}")
            return False
    
    return True

def debug_tensor_operations(tensor1: torch.Tensor, tensor2: torch.Tensor, operation: str = "operation") -> None:
    """调试张量操作"""
    logging.info(f"调试 {operation}:")
    logging.info(f"  张量1: {tensor1.shape} (设备: {tensor1.device})")
    logging.info(f"  张量2: {tensor2.shape} (设备: {tensor2.device})")
    
    # 检查设备一致性
    if tensor1.device != tensor2.device:
        logging.warning(f"设备不匹配: {tensor1.device} vs {tensor2.device}")
    
    # 检查维度兼容性
    if tensor1.dim() != tensor2.dim():
        logging.warning(f"维度不匹配: {tensor1.dim()} vs {tensor2.dim()}")

def safe_tensor_cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """安全的张量连接操作"""
    if not tensors:
        raise ValueError("张量列表为空")
    
    # 检查所有张量的设备
    devices = set(t.device for t in tensors)
    if len(devices) > 1:
        logging.warning(f"检测到多个设备: {devices}")
        # 将所有张量移动到第一个张量的设备
        target_device = tensors[0].device
        tensors = [t.to(target_device) for t in tensors]
    
    # 检查维度兼容性
    shapes = [t.shape for t in tensors]
    logging.info(f"连接张量形状: {shapes}")
    
    try:
        result = torch.cat(tensors, dim=dim)
        logging.info(f"连接成功: {result.shape}")
        return result
    except RuntimeError as e:
        logging.error(f"张量连接失败: {e}")
        # 尝试修复维度问题
        fixed_tensors = []
        for i, tensor in enumerate(tensors):
            if tensor.dim() != tensors[0].dim():
                logging.warning(f"修复张量 {i} 的维度")
                if tensor.dim() < tensors[0].dim():
                    tensor = tensor.unsqueeze(0)
                else:
                    tensor = tensor.squeeze(0)
            fixed_tensors.append(tensor)
        
        result = torch.cat(fixed_tensors, dim=dim)
        logging.info(f"修复后连接成功: {result.shape}")
        return result

def safe_tensor_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """安全的张量堆叠操作"""
    if not tensors:
        raise ValueError("张量列表为空")
    
    # 检查所有张量的形状
    shapes = [t.shape for t in tensors]
    if len(set(shapes)) > 1:
        logging.error(f"张量形状不一致: {shapes}")
        # 尝试统一形状
        target_shape = tensors[0].shape
        fixed_tensors = []
        for i, tensor in enumerate(tensors):
            if tensor.shape != target_shape:
                logging.warning(f"修复张量 {i} 的形状: {tensor.shape} -> {target_shape}")
                # 这里需要根据具体情况调整形状
                if tensor.dim() < len(target_shape):
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() > len(target_shape):
                    tensor = tensor.squeeze(0)
            fixed_tensors.append(tensor)
        tensors = fixed_tensors
    
    try:
        result = torch.stack(tensors, dim=dim)
        logging.info(f"堆叠成功: {result.shape}")
        return result
    except RuntimeError as e:
        logging.error(f"张量堆叠失败: {e}")
        raise 