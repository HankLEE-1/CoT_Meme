import os
from yacs.config import CfgNode 
import random

cfg = CfgNode()
cfg.root_dir = '/root/megrez-tmp/MemeCLIP/CNmeme'
cfg.img_folder = os.path.join(cfg.root_dir, 'meme')
cfg.info_file = os.path.join(cfg.root_dir, 'final_data.csv')
cfg.checkpoint_path = os.path.join(cfg.root_dir, 'checkpoints')
cfg.checkpoint_file = os.path.join(cfg.checkpoint_path,'model_cot.ckpt')

cfg.clip_variant = "ViT-L/14"
cfg.dataset_name = 'Pride'
cfg.name = 'MemeCLIP' 
cfg.label = 'hate'
cfg.seed = random.randint(0, 100)
cfg.test_only = False
cfg.device = 'cuda'
cfg.gpus = [0]

if cfg.label =='hate':
    cfg.class_names = ['Benign Meme', 'Harmful Meme']
elif cfg.label == 'humour':
    cfg.class_names = ['No Humour', 'Humour']
elif cfg.label == 'target':
    cfg.class_names = ['No particular target', 'Individual', 'Community', 'Organization']
elif cfg.label == 'stance':
    cfg.class_names = ['Neutral', 'Support', 'Oppose']
  
cfg.batch_size = 256
cfg.image_size = 224
cfg.num_mapping_layers = 1
cfg.unmapped_dim = 768
cfg.map_dim = 1024
cfg.num_pre_output_layers = 1
cfg.drop_probs = [0.1, 0.3, 0.1]
cfg.lr = 2e-4
cfg.max_epochs = 500
cfg.weight_decay = 1e-4
cfg.num_classes = len(cfg.class_names)
cfg.scale = 30 
cfg.print_model = True
cfg.fast_process = True
cfg.reproduce = False 
cfg.ratio = 0.5

# CoT相关配置
cfg.use_cot = True            # 启用Chain-of-Thought推理
cfg.num_hops = 3              # 推理步数
cfg.mfb_output_dim = 256      # CoT模块中融合特征的维度
cfg.mfb_factor = 8          # MFB因子化常数
cfg.consistency_weight = 0.1  # CoT步骤间一致性损失的权重

# 文本知识增强（数据集中新增列：meme，为每个梗图的模因解释）
cfg.use_meme_explanation = True   # 是否使用meme解释增强文本
cfg.meme_field = 'meme'           # 数据集中解释列名
cfg.meme_prefix = '解释：'         # 连接到文本前的前缀
cfg.meme_joiner = ' \n '          # 原文本与解释之间的连接符
cfg.meme_position = 'append'      # append 或 prepend，将解释拼接在原文后或前
cfg.meme_max_chars = 350          # 拼接后文本的最大字符数（避免过长），<=0 表示不截断

