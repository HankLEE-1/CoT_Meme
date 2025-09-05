# MemeCLIP CoT 故障排除指南

## 常见错误及解决方案

### 1. Tokenizer Padding Token 错误

**错误信息**:
```
ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token`
```

**解决方案**:
```python
# 在tokenizer初始化后添加
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**已修复的文件**:
- `MemeCLIP_with_CoT.py` - 已添加padding token修复
- `enhanced_datasets.py` - 已添加padding token修复

### 2. 模型加载失败

**错误信息**:
```
OSError: We couldn't connect to 'https://huggingface.co/...' to load this file
```

**解决方案**:
```bash
# 使用离线模式或镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或者使用本地缓存
export TRANSFORMERS_CACHE=/path/to/cache
```

### 3. CUDA 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```python
# 减小batch size
cfg.batch_size = 8  # 从16减小到8

# 使用梯度累积
trainer = Trainer(
    accelerator='gpu',
    devices=cfg.gpus,
    accumulate_grad_batches=2,  # 累积梯度
    max_epochs=cfg.max_epochs
)

# 启用混合精度训练
trainer = Trainer(
    accelerator='gpu',
    devices=cfg.gpus,
    precision=16,  # 使用16位精度
    max_epochs=cfg.max_epochs
)
```

### 4. 数据集路径错误

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```

**解决方案**:
```python
# 检查配置文件中的路径
print(f"数据路径: {cfg.root_dir}")
print(f"图像文件夹: {cfg.img_folder}")
print(f"信息文件: {cfg.info_file}")

# 确保路径存在
import os
if not os.path.exists(cfg.root_dir):
    print(f"错误: 数据根目录不存在: {cfg.root_dir}")
if not os.path.exists(cfg.img_folder):
    print(f"错误: 图像文件夹不存在: {cfg.img_folder}")
if not os.path.exists(cfg.info_file):
    print(f"错误: 信息文件不存在: {cfg.info_file}")
```

### 5. 依赖包版本冲突

**错误信息**:
```
ImportError: cannot import name '...' from '...'
```

**解决方案**:
```bash
# 创建新的虚拟环境
conda create -n memeclip_cot python=3.8
conda activate memeclip_cot

# 安装特定版本的依赖
pip install torch==1.9.0
pip install transformers==4.20.0
pip install pytorch-lightning==1.5.0
pip install clip-by-openai==1.0
```

## 测试步骤

### 步骤1: 测试CoT模块
```bash
python simple_cot_training.py --mode test_cot
```

### 步骤2: 测试数据集
```bash
python simple_cot_training.py --mode test_dataset
```

### 步骤3: 完整训练
```bash
python simple_cot_training.py --mode train
```

## 调试技巧

### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 检查模型参数
```python
def check_model_parameters(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
```

### 3. 检查数据加载
```python
def check_data_loading(dataset, collator):
    try:
        sample = dataset[0]
        print(f"样本键: {list(sample.keys())}")
        
        batch = collator([sample])
        print(f"批处理键: {list(batch.keys())}")
        print(f"图像特征形状: {batch['image_features'].shape}")
        print(f"文本特征形状: {batch['text_features'].shape}")
        
        return True
    except Exception as e:
        print(f"数据加载错误: {e}")
        return False
```

## 性能优化建议

### 1. 内存优化
```python
# 使用梯度检查点
model.gradient_checkpointing_enable()

# 使用动态批处理
trainer = Trainer(
    accelerator='gpu',
    devices=cfg.gpus,
    accumulate_grad_batches=2,
    max_epochs=cfg.max_epochs
)
```

### 2. 速度优化
```python
# 使用混合精度训练
trainer = Trainer(
    accelerator='gpu',
    devices=cfg.gpus,
    precision=16,
    max_epochs=cfg.max_epochs
)

# 使用多进程数据加载
train_loader = DataLoader(
    dataset_train, 
    batch_size=cfg.batch_size, 
    shuffle=True,
    collate_fn=collator, 
    num_workers=4  # 增加worker数量
)
```

### 3. 推理优化
```python
# 模型量化
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 推理缓存
@torch.no_grad()
def cached_inference(model, batch):
    # 实现推理缓存
    pass
```

## 常见配置问题

### 1. 配置文件路径
确保 `configs.py` 中的路径正确:
```python
cfg.root_dir = '/path/to/your/data'  # 修改为实际路径
cfg.img_folder = os.path.join(cfg.root_dir, 'meme')
cfg.info_file = os.path.join(cfg.root_dir, 'final_data.csv')
cfg.checkpoint_path = os.path.join(cfg.root_dir, 'checkpoints')
```

### 2. GPU配置
```python
cfg.device = 'cuda'
cfg.gpus = [0]  # 使用第一个GPU
# 或者使用CPU
# cfg.device = 'cpu'
# cfg.gpus = []
```

### 3. 模型配置
```python
cfg.batch_size = 8  # 根据GPU内存调整
cfg.max_epochs = 3
cfg.lr = 1e-4
cfg.image_size = 224
```

## 联系支持

如果遇到其他问题，请:

1. 检查错误日志文件
2. 运行测试脚本确认各组件工作正常
3. 检查环境配置和依赖版本
4. 提供详细的错误信息和环境信息

## 快速修复脚本

```bash
#!/bin/bash
# 快速修复常见问题

echo "检查Python环境..."
python --version

echo "检查CUDA..."
nvidia-smi

echo "检查依赖包..."
pip list | grep -E "(torch|transformers|pytorch-lightning)"

echo "测试CoT模块..."
python simple_cot_training.py --mode test_cot

echo "测试数据集..."
python simple_cot_training.py --mode test_dataset

echo "检查完成!"
``` 