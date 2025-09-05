import os
import pandas as pd
import torch
import clip

from PIL import Image
from torch.utils.data import Dataset
from configs import cfg
from transformers import AutoProcessor, CLIPVisionModel
from torchvision import transforms as T


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_default_dtype(torch.float32)

from configs import cfg

# 自动编码检测的简单回退读取
_DEF_ENCODINGS = [
    'utf-8',        # 常见
    'utf-8-sig',    # 带BOM
    'gbk',          # 中文Windows常见
    'latin1'        # 兜底
]

def _read_csv_with_fallback(path):
    last_err = None
    for enc in _DEF_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            # 其它异常也尝试下一个编码
            last_err = e
            continue
    # 最后再尝试一次宽松解析（新版本pandas支持encoding_errors）
    try:
        return pd.read_csv(path, encoding='latin1')
    except Exception:
        if last_err is not None:
            raise last_err
        raise

class Custom_Dataset(Dataset):
    def __init__(self, cfg, root_folder, dataset, label, split='train', image_size=224, fast=True):
        super(Custom_Dataset, self).__init__()
        self.cfg = cfg
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split
        self.label = label

        self.image_size = image_size
        self.fast = fast

        self.info_file = cfg.info_file
        # 兼容不同编码的数据文件
        self.df = _read_csv_with_fallback(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

        if self.label == 'target':
            self.df = self.df[self.df['hate'] == 1].reset_index(drop=True)

        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row['text'] == 'None':
            txt = 'null'
        else:
            txt = row['text']

        # 如果配置开启并且数据中存在meme解释列，则拼接
        joined_txt = txt
        if getattr(self.cfg, 'use_meme_explanation', False) and getattr(self.cfg, 'meme_field', 'meme') in row.index:
            meme_exp = row.get(self.cfg.meme_field, '')
            if isinstance(meme_exp, str) and len(meme_exp.strip()) > 0:
                if getattr(self.cfg, 'meme_position', 'append') == 'prepend':
                    joined_txt = f"{self.cfg.meme_prefix}{meme_exp}{self.cfg.meme_joiner}{txt}"
                else:
                    joined_txt = f"{txt}{self.cfg.meme_joiner}{self.cfg.meme_prefix}{meme_exp}"
                max_chars = int(getattr(self.cfg, 'meme_max_chars', 0))
                if max_chars > 0 and len(joined_txt) > max_chars:
                    joined_txt = joined_txt[:max_chars]

        image_fn = row['name']
        image = Image.open(f"{self.cfg.img_folder}/{image_fn}").convert('RGB')\
            .resize((self.image_size, self.image_size))
        
        item = {
            'image': image,
            'text': joined_txt,
            'label': row[self.label],
            'idx_meme': row['name'],
            'origin_text': txt,
            'meme_text': row.get(getattr(self.cfg, 'meme_field', 'meme'), '')
        }

        return item

class Custom_Collator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_model, _ = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        _, self.clip_preprocess = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        self.clip_model.float().eval()
        # 随机数据增强（提高每次训练的随机性与泛化）
        self.augment = T.Compose([
            T.RandomResizedCrop(self.cfg.image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])

    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = [item['idx_meme'] for item in batch]

        batch_new = {'labels': labels,
                     'idx_memes': idx_memes,
                     }
        
        image_embed_list = []
        text_embed_list = []

        for item in batch:

            # 应用随机增强，然后再交给CLIP预处理
            img_aug = self.augment(item['image'])
            pixel_values = self.clip_preprocess(img_aug).unsqueeze(0)
            text = clip.tokenize(item['text'], context_length=77, truncate=True)

            image_features, text_features = self.compute_CLIP_features_without_proj(self.clip_model,
                                                                    pixel_values.to(self.cfg.device),
                                                                    text.to(self.cfg.device))
            text_embed_list.append(text_features.cpu().detach())
            image_embed_list.append(image_features.cpu().detach())

        image_features = torch.cat([item for item in image_embed_list], dim=0)
        text_features = torch.cat([item for item in text_embed_list], dim=0)

        batch_new['image_features'] = image_features
        batch_new['text_features'] = text_features

        return batch_new
    
    def compute_CLIP_features_without_proj(self, clip_model, img_input, text_input):
        image_features = clip_model.visual(img_input.type(clip_model.dtype))

        x = clip_model.token_embedding(text_input).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        text_features = x[torch.arange(x.shape[0]), text_input.argmax(dim=-1)]

        return image_features, text_features


def load_dataset(cfg, split):
    dataset = Custom_Dataset(cfg = cfg, root_folder=cfg.root_dir, dataset=cfg.dataset_name, split=split,
                           image_size=cfg.image_size, label = cfg.label, fast=cfg.fast_process)

    return dataset
