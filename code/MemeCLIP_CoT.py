import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import clip
from tqdm import tqdm
import os
from functools import partial
import torch.nn.functional as F
from transformers import AutoTokenizer
torch.set_default_dtype(torch.float32)
from models import LinearClassifier, CosineClassifier, LinearProjection, CLIP_Text, Adapter
from cot_modules import CoTClassifier, ChainOfThought

class MemeCLIP_CoT(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes = cfg.num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes = cfg.num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes = cfg.num_classes, average='macro')

        self.clip_model, _ = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        self.clip_model.float()

        pre_output_input_dim = self.cfg.map_dim
        pre_output_layers = [nn.Dropout(p=cfg.drop_probs[1])]
        output_input_dim = pre_output_input_dim

        # 原有的分类器（作为baseline）
        self.classifier = CosineClassifier(feat_dim = output_input_dim, num_classes=cfg.num_classes, dtype=self.clip_model.dtype)
        self.init_head_text_feat()
        self.text_encoder =  CLIP_Text(self.clip_model)
        self.img_adapter = Adapter(self.cfg.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(self.cfg.map_dim, 4).to(self.clip_model.dtype)
        self.clip_model.visual.proj = None

        # 新增：CoT分类器（仅使用CoT）
        self.cot_classifier = CoTClassifier(
            input_dim=self.cfg.map_dim,
            num_classes=cfg.num_classes,
            hidden_dim=self.cfg.mfb_output_dim,
            num_reasoning_steps=self.cfg.num_hops
        )

        for _, p in self.clip_model.named_parameters():
            p.requires_grad_(False)
        
        # 禁用基线分类器参数训练，仅训练CoT相关参数
        for name, param in self.classifier.named_parameters():
            param.requires_grad_(False)
        for name, param in self.cot_classifier.named_parameters():
            param.requires_grad_(True)

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
        
        # 记录CoT相关指标
        if self.cfg.use_cot:
            self.log('训练/CoT一致性损失', output.get('consistency_loss', 0.0))
            self.log('训练/CoT准确率', output.get('cot_accuracy', 0.0))
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        total_loss = output['loss']
        
        self.log(f'验证/总损失', total_loss)
        self.log(f'验证/损失', output['loss'])
        self.log(f'验证/准确率', output['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'验证/AUROC', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'验证/F1分数', output['f1'], on_step=False, on_epoch=True, prog_bar=True)
        
        # 记录CoT相关指标
        if self.cfg.use_cot:
            self.log('验证/CoT一致性损失', output.get('consistency_loss', 0.0))
            self.log('验证/CoT准确率', output.get('cot_accuracy', 0.0))
        
        return total_loss

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log(f'测试/准确率', output['accuracy'])
        self.log(f'测试/AUROC', output['auroc'])
        self.log(f'测试/F1分数', output['f1'])
        
        # 记录CoT相关指标
        if self.cfg.use_cot:
            self.log('测试/CoT准确率', output.get('cot_accuracy', 0.0))
        
        return output

    def common_step(self, batch):
        image_embeds = batch['image_features']
        text_embeds = batch['text_features']

        image_projection = self.image_map(image_embeds)
        txt_projection = self.text_map(text_embeds)

        image_features = self.img_adapter(image_projection)
        text_features = self.text_adapter(txt_projection)

        text_features = self.cfg.ratio * text_features + (1 - self.cfg.ratio) * txt_projection
        image_features = self.cfg.ratio * image_features + (1 - self.cfg.ratio) * image_projection

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        output = {}
        
        # 仅使用CoT分支进行推理与训练
        cot_output = self.cot_classifier(
            image_features,
            text_features,
            return_reasoning=True
        )
        
        cot_logits = cot_output['logits']
        cot_probs = torch.softmax(cot_logits, dim=1)
        cot_preds = cot_logits.argmax(dim=1)

        cot_loss = self.cross_entropy_loss(cot_logits, batch['labels'])
        consistency_loss = cot_output.get('consistency_loss', torch.tensor(0.0, device=cot_logits.device))
        total_loss = cot_loss + consistency_loss

        # 统一用CoT结果作为主指标
        output['loss'] = total_loss
        output['accuracy'] = self.acc(cot_preds, batch['labels'])
        output['auroc'] = self.auroc(cot_probs, batch['labels'])
        output['f1'] = self.f1(cot_preds, batch['labels'])
        output['cot_accuracy'] = output['accuracy']
        output['consistency_loss'] = consistency_loss
        output['cot_loss'] = cot_loss

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

def create_model(cfg):
    model = MemeCLIP_CoT(cfg)
    return model 