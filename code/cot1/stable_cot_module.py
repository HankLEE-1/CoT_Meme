import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
from typing import Tuple, List, Dict

class StableCoTReasoningModule(nn.Module):
    """更稳定的Chain of Thought推理模块"""
    
    def __init__(self, cfg, reasoning_steps: int = 3):
        super().__init__()
        self.cfg = cfg
        self.reasoning_steps = reasoning_steps
        
        # 使用更稳定的tokenizer
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.llm_model = AutoModel.from_pretrained("bert-base-uncased")
            # 将模型移动到正确的设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.llm_model = self.llm_model.to(device)
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
            # 使用简单的线性层作为fallback
            self.llm_tokenizer = None
            self.llm_model = None
            self.fallback_encoder = nn.Linear(512, 256)  # 简单的特征编码器
        
        # 推理提示模板
        self.reasoning_prompts = {
            'hate': [
                "Analyze visual content",
                "Analyze text content", 
                "Analyze interaction between visual and text",
                "Identify harmful elements",
                "Make final classification"
            ],
            'target': [
                "Identify mentioned groups or individuals",
                "Analyze targeting specificity",
                "Determine targeting scope",
                "Classify target type"
            ],
            'stance': [
                "Analyze overall tone",
                "Identify supporting or opposing elements",
                "Evaluate stance strength",
                "Make final stance classification"
            ],
            'humor': [
                "Identify humor attempts",
                "Analyze humor type and style",
                "Evaluate appropriateness",
                "Make final humor classification"
            ]
        }
        
        # 推理特征提取
        self.reasoning_encoder = nn.Sequential(
            nn.Linear(768 if self.llm_model else 256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 注意力机制
        self.reasoning_attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # 推理聚合
        self.reasoning_aggregator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def generate_reasoning_prompt(self, task_type: str, image_desc: str, text_content: str) -> str:
        """生成推理提示"""
        prompts = self.reasoning_prompts.get(task_type, self.reasoning_prompts['hate'])
        
        base_prompt = f"Task: Analyze meme for {task_type} classification. Image: {image_desc}. Text: {text_content}."
        reasoning_prompt = base_prompt + " Steps: " + " ".join(prompts)
        return reasoning_prompt
    
    def extract_reasoning_features_simple(self, reasoning_text: str) -> torch.Tensor:
        """使用简单方法提取推理特征"""
        # 简单的文本特征提取
        text_length = len(reasoning_text)
        # 创建基于文本长度的简单特征
        features = torch.zeros(1, 256)
        features[0, :min(text_length, 256)] = 1.0  # 简单的one-hot编码
        
        # 确保特征在正确的设备上
        if hasattr(self, 'llm_model') and self.llm_model is not None:
            device = next(self.llm_model.parameters()).device
            features = features.to(device)
        else:
            # 如果没有模型，使用默认设备
            features = features.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        return features
    
    def extract_reasoning_features_bert(self, reasoning_text: str) -> torch.Tensor:
        """使用BERT提取推理特征"""
        try:
            inputs = self.llm_tokenizer(
                reasoning_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # 确保输入在正确的设备上
            device = next(self.llm_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
            
            return features
        except Exception as e:
            print(f"Error in BERT feature extraction: {e}")
            return self.extract_reasoning_features_simple(reasoning_text)
    
    def extract_reasoning_features(self, reasoning_text: str) -> torch.Tensor:
        """提取推理特征"""
        if self.llm_model is not None:
            return self.extract_reasoning_features_bert(reasoning_text)
        else:
            return self.extract_reasoning_features_simple(reasoning_text)
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
                image_desc: str, text_content: str, task_type: str) -> Tuple[torch.Tensor, str]:
        """
        CoT推理模块的前向传播
        
        Args:
            image_features: CLIP图像特征
            text_features: CLIP文本特征  
            image_desc: 图像描述
            text_content: 文本内容
            task_type: 分类任务类型
            
        Returns:
            reasoning_features: 聚合的推理特征
            reasoning_text: 生成的推理文本
        """
        
        # 生成推理提示
        reasoning_prompt = self.generate_reasoning_prompt(task_type, image_desc, text_content)
        
        # 提取推理特征
        reasoning_features = self.extract_reasoning_features(reasoning_prompt)
        
        # 编码推理特征
        encoded_reasoning = self.reasoning_encoder(reasoning_features)
        
        # 应用注意力机制
        encoded_reasoning = encoded_reasoning.unsqueeze(1)  # [batch_size, 1, 256]
        attended_reasoning, _ = self.reasoning_attention(
            encoded_reasoning, encoded_reasoning, encoded_reasoning
        )
        
        # 聚合推理特征
        aggregated_reasoning = self.reasoning_aggregator(attended_reasoning.squeeze(1))
        
        return aggregated_reasoning, reasoning_prompt

class LightweightCoTModule(nn.Module):
    """轻量级CoT模块，不依赖外部LLM"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 简单的文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(512, 256),  # CLIP特征维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 推理步骤编码器
        self.step_encoder = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 推理聚合器
        self.reasoning_aggregator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64)
        )
        
        # 推理模板
        self.reasoning_templates = {
            'hate': ['visual', 'text', 'interaction', 'harm', 'classification'],
            'target': ['groups', 'specificity', 'scope', 'type'],
            'stance': ['tone', 'elements', 'strength', 'classification'],
            'humor': ['attempts', 'style', 'appropriateness', 'classification']
        }
    
    def encode_text_content(self, text: str) -> torch.Tensor:
        """编码文本内容"""
        # 简单的字符级编码
        text_tensor = torch.zeros(1, 512)
        for i, char in enumerate(text[:512]):
            text_tensor[0, i] = ord(char) % 256
        
        # 确保张量在正确的设备上
        device = next(self.parameters()).device
        text_tensor = text_tensor.to(device)
        
        return text_tensor
    
    def generate_reasoning_steps(self, task_type: str, image_desc: str, text_content: str) -> List[str]:
        """生成推理步骤"""
        template = self.reasoning_templates.get(task_type, self.reasoning_templates['hate'])
        steps = []
        
        for step in template:
            if step == 'visual':
                steps.append(f"Analyze visual content: {image_desc}")
            elif step == 'text':
                steps.append(f"Analyze text content: {text_content}")
            elif step == 'interaction':
                steps.append(f"Analyze interaction between visual and text")
            elif step == 'harm':
                steps.append("Identify potential harmful elements")
            elif step == 'classification':
                steps.append("Make final classification")
            else:
                steps.append(f"Analyze {step}")
        
        return steps
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor,
                image_desc: str, text_content: str, task_type: str) -> Tuple[torch.Tensor, str]:
        """轻量级CoT前向传播"""
        
        # 编码文本内容
        text_encoded = self.encode_text_content(text_content)
        
        # 生成推理步骤
        reasoning_steps = self.generate_reasoning_steps(task_type, image_desc, text_content)
        
        # 编码每个推理步骤
        step_features = []
        for step in reasoning_steps:
            step_encoded = self.encode_text_content(step)
            step_processed = self.text_encoder(step_encoded)
            step_features.append(step_processed)
        
        # 使用LSTM处理推理步骤序列
        if step_features:
            # 确保所有步骤特征具有相同的形状
            step_features = [f.squeeze(0) if f.dim() > 1 else f for f in step_features]
            steps_tensor = torch.stack(step_features, dim=0)  # [num_steps, 128]
            
            # 添加batch维度
            steps_tensor = steps_tensor.unsqueeze(0)  # [1, num_steps, 128]
            
            lstm_out, (hidden, cell) = self.step_encoder(steps_tensor)
            
            # 使用最后一个隐藏状态
            reasoning_features = self.reasoning_aggregator(hidden[-1])
        else:
            # Fallback
            device = next(self.parameters()).device
            reasoning_features = torch.zeros(1, 64, device=device)
        
        reasoning_text = " | ".join(reasoning_steps)
        
        return reasoning_features, reasoning_text 