import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class MultiModalFusionBlock(nn.Module):
    """
    多模态融合块 - 用于融合图像和文本特征
    这是CoT推理中的核心组件
    """
    def __init__(self, input_dim: int, output_dim: int, factor: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factor = factor
        
        # MFB (Multi-modal Factorized Bilinear pooling) 组件
        self.fc1 = nn.Linear(input_dim, output_dim * factor)
        self.fc2 = nn.Linear(input_dim, output_dim * factor)
        self.fc3 = nn.Linear(output_dim * factor, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        融合图像和文本特征
        Args:
            image_feat: 图像特征 [batch_size, input_dim]
            text_feat: 文本特征 [batch_size, input_dim]
        Returns:
            融合后的特征 [batch_size, output_dim]
        """
        # 双线性池化
        image_proj = self.fc1(image_feat)  # [batch_size, output_dim * factor]
        text_proj = self.fc2(text_feat)    # [batch_size, output_dim * factor]
        
        # 元素级乘法
        fused = image_proj * text_proj     # [batch_size, output_dim * factor]
        
        # 降维
        output = self.fc3(fused)           # [batch_size, output_dim]
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output

class ReasoningStep(nn.Module):
    """
    单个推理步骤模块
    模拟人类推理过程中的一个思考步骤
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 注意力机制 - 关注当前推理状态
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor, reasoning_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行一个推理步骤
        Args:
            x: 输入特征 [batch_size, feature_dim]
            reasoning_context: 推理上下文 [batch_size, seq_len, feature_dim]
        Returns:
            更新后的特征 [batch_size, feature_dim]
        """
        # 自注意力
        if reasoning_context is not None:
            # 如果有上下文，使用交叉注意力
            x_reshaped = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
            attn_output, _ = self.attention(
                query=x_reshaped,
                key=reasoning_context,
                value=reasoning_context
            )
            x = x + self.norm1(attn_output.squeeze(1))
        else:
            # 自注意力
            x_reshaped = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
            attn_output, _ = self.attention(
                query=x_reshaped,
                key=x_reshaped,
                value=x_reshaped
            )
            x = x + self.norm1(attn_output.squeeze(1))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = x + self.norm2(ffn_output)
        
        return x

class ChainOfThought(nn.Module):
    """
    Chain of Thought 推理模块
    模拟人类逐步推理的过程
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 512,
                 num_steps: int = 3,
                 use_consistency: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.use_consistency = use_consistency
        
        # 多模态融合块
        self.fusion_block = MultiModalFusionBlock(
            input_dim=input_dim,
            output_dim=hidden_dim
        )
        
        # 推理步骤
        self.reasoning_steps = nn.ModuleList([
            ReasoningStep(feature_dim=hidden_dim, hidden_dim=hidden_dim)
            for _ in range(num_steps)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # 一致性损失权重
        self.consistency_weight = 0.1
        
    def forward(self, 
                image_feat: torch.Tensor, 
                text_feat: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        执行Chain of Thought推理
        Args:
            image_feat: 图像特征 [batch_size, input_dim]
            text_feat: 文本特征 [batch_size, input_dim]
            return_intermediate: 是否返回中间结果
        Returns:
            包含最终输出和中间结果的字典
        """
        batch_size = image_feat.size(0)
        
        # 初始融合
        fused_feat = self.fusion_block(image_feat, text_feat)
        
        # 存储中间推理结果
        intermediate_states = [fused_feat]
        reasoning_context = fused_feat.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 逐步推理
        for i, reasoning_step in enumerate(self.reasoning_steps):
            # 执行推理步骤
            current_state = reasoning_step(fused_feat, reasoning_context)
            
            # 更新推理上下文
            reasoning_context = torch.cat([
                reasoning_context, 
                current_state.unsqueeze(1)
            ], dim=1)  # [batch_size, i+2, hidden_dim]
            
            # 更新融合特征
            fused_feat = current_state
            intermediate_states.append(current_state)
        
        # 输出投影
        final_output = self.output_projection(fused_feat)
        
        result = {
            'final_output': final_output,
            'reasoning_context': reasoning_context
        }
        
        if return_intermediate:
            result['intermediate_states'] = intermediate_states
        
        return result
    
    def compute_consistency_loss(self, intermediate_states: List[torch.Tensor]) -> torch.Tensor:
        """
        计算推理步骤间的一致性损失
        """
        if not self.use_consistency or len(intermediate_states) < 2:
            return torch.tensor(0.0, device=intermediate_states[0].device)
        
        consistency_loss = 0.0
        for i in range(1, len(intermediate_states)):
            # 计算相邻步骤间的余弦相似度损失
            cos_sim = F.cosine_similarity(
                intermediate_states[i-1], 
                intermediate_states[i], 
                dim=1
            )
            consistency_loss += (1 - cos_sim).mean()
        
        return consistency_loss * self.consistency_weight

class CoTClassifier(nn.Module):
    """
    基于CoT的分类器
    """
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 512,
                 num_reasoning_steps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps
        
        # CoT模块
        self.cot_module = ChainOfThought(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_steps=num_reasoning_steps
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, 
                image_feat: torch.Tensor, 
                text_feat: torch.Tensor,
                return_reasoning: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            image_feat: 图像特征
            text_feat: 文本特征
            return_reasoning: 是否返回推理过程
        Returns:
            预测结果和推理信息
        """
        # CoT推理
        cot_output = self.cot_module(
            image_feat, 
            text_feat, 
            return_intermediate=return_reasoning
        )
        
        # 分类
        logits = self.classifier(cot_output['final_output'])
        
        result = {
            'logits': logits,
            'reasoning_context': cot_output['reasoning_context']
        }
        
        if return_reasoning:
            result['intermediate_states'] = cot_output['intermediate_states']
            result['consistency_loss'] = self.cot_module.compute_consistency_loss(
                cot_output['intermediate_states']
            )
        
        return result