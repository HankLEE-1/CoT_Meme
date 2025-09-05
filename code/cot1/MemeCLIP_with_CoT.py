import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import clip
from tqdm import tqdm
import os
from functools import partial
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import re
from typing import Dict, List, Tuple, Optional
torch.set_default_dtype(torch.float32)
from models import LinearClassifier, CosineClassifier, LinearProjection, CLIP_Text, Adapter
from stable_cot_module import LightweightCoTModule
from device_utils import check_batch_device_consistency, fix_batch_device_consistency, debug_device_info
from dimension_utils import check_tensor_dimensions, fix_batch_dimensions, safe_tensor_cat
from safe_device_utils import DeviceManager

class CoTReasoningModule(nn.Module):
    """Chain of Thought reasoning module for multimodal meme classification"""
    
    def __init__(self, cfg, reasoning_steps: int = 3):
        super().__init__()
        self.cfg = cfg
        self.reasoning_steps = reasoning_steps
        
        # LLM for reasoning (using a smaller model for efficiency)
        self.llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.llm_model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        
        # Fix padding token issue for DialoGPT
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        # Reasoning prompt templates
        self.reasoning_prompts = {
            'hate': [
                "Let's analyze this meme step by step:",
                "1. What is the visual content?",
                "2. What is the text content?",
                "3. How do they interact?",
                "4. Is there any harmful content?",
                "5. Final classification:"
            ],
            'target': [
                "Let's identify the target step by step:",
                "1. What groups are mentioned or implied?",
                "2. Are they specific individuals or communities?",
                "3. What type of targeting is this?",
                "4. Final target classification:"
            ],
            'stance': [
                "Let's analyze the stance step by step:",
                "1. What is the overall tone?",
                "2. Is it supportive, neutral, or opposing?",
                "3. What evidence supports this stance?",
                "4. Final stance classification:"
            ],
            'humor': [
                "Let's analyze the humor step by step:",
                "1. Is there an attempt at humor?",
                "2. What type of humor is used?",
                "3. Is it appropriate or inappropriate?",
                "4. Final humor classification:"
            ]
        }
        
        # Reasoning feature extraction
        self.reasoning_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for reasoning steps
        self.reasoning_attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Reasoning aggregation
        self.reasoning_aggregator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def generate_reasoning_prompt(self, task_type: str, image_desc: str, text_content: str) -> str:
        """Generate a reasoning prompt for the given task"""
        prompts = self.reasoning_prompts[task_type]
        
        base_prompt = f"""
        Task: Analyze this meme for {task_type} classification.
        
        Image description: {image_desc}
        Text content: {text_content}
        
        """
        
        reasoning_prompt = base_prompt + "\n".join(prompts)
        return reasoning_prompt
    
    def extract_reasoning_features(self, reasoning_text: str) -> torch.Tensor:
        """Extract features from reasoning text using LLM"""
        inputs = self.llm_tokenizer(
            reasoning_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.llm_model(**inputs)
            # Use the last hidden state as features
            features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
        
        return features
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
                image_desc: str, text_content: str, task_type: str) -> Tuple[torch.Tensor, str]:
        """
        Forward pass through the CoT reasoning module
        
        Args:
            image_features: CLIP image features
            text_features: CLIP text features  
            image_desc: Description of the image
            text_content: Text content of the meme
            task_type: Type of classification task
            
        Returns:
            reasoning_features: Aggregated reasoning features
            reasoning_text: Generated reasoning text
        """
        
        # Generate reasoning prompt
        reasoning_prompt = self.generate_reasoning_prompt(task_type, image_desc, text_content)
        
        # Extract reasoning features from LLM
        reasoning_features = self.extract_reasoning_features(reasoning_prompt)
        
        # Encode reasoning features
        encoded_reasoning = self.reasoning_encoder(reasoning_features)
        
        # Apply attention mechanism
        # Reshape for attention: [batch_size, 1, 256]
        encoded_reasoning = encoded_reasoning.unsqueeze(1)
        attended_reasoning, _ = self.reasoning_attention(
            encoded_reasoning, encoded_reasoning, encoded_reasoning
        )
        
        # Aggregate reasoning features
        aggregated_reasoning = self.reasoning_aggregator(attended_reasoning.squeeze(1))
        
        return aggregated_reasoning, reasoning_prompt

class MemeCLIPWithCoT(pl.LightningModule):
    """MemeCLIP model enhanced with Chain of Thought reasoning"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes=cfg.num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=cfg.num_classes, average='macro')

        # 初始化设备管理器
        self.device_manager = DeviceManager()
        
        # Load CLIP model
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(self.cfg.clip_variant, device=clip_device, jit=False)
        self.clip_model.float()

        # Initialize CoT reasoning module (use lightweight version for stability)
        self.cot_reasoning = LightweightCoTModule(cfg)
        
        # Original MemeCLIP components
        pre_output_input_dim = self.cfg.map_dim
        pre_output_layers = [nn.Dropout(p=cfg.drop_probs[1])]
        output_input_dim = pre_output_input_dim

        self.classifier = CosineClassifier(feat_dim=output_input_dim, num_classes=cfg.num_classes, dtype=self.clip_model.dtype)
        self.init_head_text_feat()
        self.text_encoder = CLIP_Text(self.clip_model)
        self.img_adapter = Adapter(self.cfg.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(self.cfg.map_dim, 4).to(self.clip_model.dtype)
        self.clip_model.visual.proj = None

        # Freeze CLIP parameters
        for _, p in self.clip_model.named_parameters():
            p.requires_grad_(False)
        
        for name, param in self.classifier.named_parameters():
            param.requires_grad_(True)

        # Mapping layers
        self.image_map = LinearProjection(self.cfg.unmapped_dim, self.cfg.map_dim,
                                          self.cfg.num_mapping_layers, self.cfg.drop_probs)
        self.text_map = LinearProjection(self.cfg.unmapped_dim, self.cfg.map_dim,
                                         self.cfg.num_mapping_layers, self.cfg.drop_probs)
        
        self.soft = nn.Softmax(dim=1)
            
        if self.cfg.num_pre_output_layers >= 1:
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.cfg.map_dim), nn.ReLU(), nn.Dropout(p=cfg.drop_probs[2])])
            output_input_dim = self.cfg.map_dim

        for _ in range(1, self.cfg.num_pre_output_layers):
            pre_output_layers.extend(
                [nn.Linear(self.cfg.map_dim, self.cfg.map_dim), nn.ReLU(), nn.Dropout(p=cfg.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # Reasoning feature fusion
        self.reasoning_fusion = nn.Sequential(
            nn.Linear(64 + self.cfg.map_dim, self.cfg.map_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.map_dim, self.cfg.map_dim)
        )

    def forward(self, batch):
        pass
    
    def init_head_text_feat(self):
        print("使用文本特征初始化分类头")
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names]
        prompts = clip.tokenize([p for p in prompts], context_length=77, truncate=True).to(self.cfg.device)
        text_features = self.clip_model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features @ self.clip_model.visual.proj.t()
        text_features = F.normalize(text_features, dim=-1)
        self.classifier.apply_weight(text_features)

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)
        total_loss = output['loss']
        
        self.log('训练/总损失', total_loss)
        self.log('训练/损失', output['loss'])
        self.log('训练/准确率', output['accuracy'])
        self.log(f'训练/AUROC', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        total_loss = output['loss']
        
        self.log(f'验证/总损失', total_loss)
        self.log(f'验证/损失', output['loss'])
        self.log(f'验证/准确率', output['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'验证/AUROC', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'验证/F1分数', output['f1'], on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log(f'测试/准确率', output['accuracy'])
        self.log(f'测试/AUROC', output['auroc'])
        self.log(f'测试/F1分数', output['f1'])
        return output

    def common_step(self, batch):
        # 检查批次设备一致性
        if not check_batch_device_consistency(batch):
            logging.warning("检测到批次设备不一致，正在修复...")
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = fix_batch_device_consistency(batch, target_device)
        
        # 修复批次维度问题
        batch = fix_batch_dimensions(batch)
        
        # 调试设备信息和维度
        debug_device_info(self, batch)
        check_tensor_dimensions(batch, "批次张量")
        
        image_embeds = batch['image_features']
        text_embeds = batch['text_features']
        
        # Get image descriptions and text content for reasoning
        image_descriptions = batch.get('image_descriptions', [''] * len(image_embeds))
        text_contents = batch.get('text_contents', [''] * len(text_embeds))

        image_projection = self.image_map(image_embeds)
        txt_projection = self.text_map(text_embeds)

        image_features = self.img_adapter(image_projection)
        text_features = self.text_adapter(txt_projection)

        text_features = self.cfg.ratio * text_features + (1 - self.cfg.ratio) * txt_projection
        image_features = self.cfg.ratio * image_features + (1 - self.cfg.ratio) * image_projection

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        features = torch.mul(image_features, text_features)

        # Apply CoT reasoning
        reasoning_features_list = []
        reasoning_texts = []
        
        # 使用设备管理器确保CoT模块在正确的设备上
        self.cot_reasoning = self.device_manager.ensure_model_on_device(self.cot_reasoning)
        
        for i in range(len(features)):
            reasoning_feat, reasoning_text = self.cot_reasoning(
                image_features[i:i+1], 
                text_features[i:i+1],
                image_descriptions[i], 
                text_contents[i], 
                self.cfg.label
            )
            # 确保推理特征是正确的形状
            if reasoning_feat.dim() == 1:
                reasoning_feat = reasoning_feat.unsqueeze(0)  # 添加batch维度
            reasoning_features_list.append(reasoning_feat)
            reasoning_texts.append(reasoning_text)
        
        # 使用安全的张量连接
        reasoning_features = safe_tensor_cat(reasoning_features_list, dim=0)
        
        # 确保推理特征在正确的设备上
        if reasoning_features.device != features.device:
            reasoning_features = reasoning_features.to(features.device)
        
        # Fuse reasoning features with original features
        combined_features = torch.cat([features, reasoning_features], dim=1)
        enhanced_features = self.reasoning_fusion(combined_features)

        features_pre_output = self.pre_output(enhanced_features)
        logits = self.classifier(features_pre_output).squeeze(dim=1) 
        preds_proxy = torch.sigmoid(logits)
        _, preds = logits.data.max(1)

        output = {}
        output['loss'] = self.cross_entropy_loss(logits, batch['labels'])
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])
        output['f1'] = self.f1(preds, batch['labels'])
        output['reasoning_texts'] = reasoning_texts

        return output

    def on_train_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()
        
    def on_validation_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return optimizer

def create_model_with_cot(cfg):
    model = MemeCLIPWithCoT(cfg)
    return model 