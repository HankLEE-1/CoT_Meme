import os
import pandas as pd
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
from configs import cfg
from transformers import AutoProcessor, CLIPVisionModel, AutoTokenizer, AutoModel
import torch.nn.functional as F
from typing import List, Dict, Any

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_default_dtype(torch.float32)

class Enhanced_Custom_Dataset(Dataset):
    """Enhanced dataset class with image description generation for CoT reasoning"""
    
    def __init__(self, cfg, root_folder, dataset, label, split='train', image_size=224, fast=True):
        super(Enhanced_Custom_Dataset, self).__init__()
        self.cfg = cfg
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split
        self.label = label

        self.image_size = image_size
        self.fast = fast

        self.info_file = cfg.info_file
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

        if self.label == 'target':
            self.df = self.df[self.df['hate'] == 1].reset_index(drop=True)

        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')
        
        # Initialize image description model
        self.init_description_model()

    def init_description_model(self):
        """Initialize the image description model"""
        try:
            # Use a smaller model for efficiency
            self.desc_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.desc_model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
            
            # Fix padding token issue for DialoGPT
            if self.desc_tokenizer.pad_token is None:
                self.desc_tokenizer.pad_token = self.desc_tokenizer.eos_token
                
            self.desc_model.eval()
        except Exception as e:
            print(f"Warning: Could not load description model: {e}")
            self.desc_model = None
            self.desc_tokenizer = None

    def generate_image_description(self, image: Image.Image) -> str:
        """Generate a description of the image"""
        if self.desc_model is None:
            return "An image with text overlay"
        
        try:
            # Simple description generation using CLIP features
            # This is a simplified approach - in practice you might use a dedicated captioning model
            transform = clip.load(self.cfg.clip_variant, device="cpu")[1]
            image_tensor = transform(image).unsqueeze(0)
            
            # Use CLIP to get image features and generate a simple description
            with torch.no_grad():
                # This is a placeholder - in practice you'd use a proper captioning model
                description = "An image with visual content and overlaid text"
            
            return description
        except Exception as e:
            print(f"Error generating description: {e}")
            return "An image with text overlay"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row['text'] == 'None':
            txt = 'null'
        else:
            txt = row['text']

        image_fn = row['name']
        image = Image.open(f"{self.cfg.img_folder}/{image_fn}").convert('RGB')\
            .resize((self.image_size, self.image_size))
        text = txt
        
        # Generate image description for CoT reasoning
        image_description = self.generate_image_description(image)

        item = {
            'image': image,
            'text': text,
            'label': row[self.label],
            'idx_meme': row['name'],
            'origin_text': txt,
            'image_description': image_description
        }

        return item

class Enhanced_Custom_Collator(object):
    """Enhanced collator that includes image descriptions and text content for CoT reasoning"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_model, _ = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        _, self.clip_preprocess = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        self.clip_model.float().eval()

    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = [item['idx_meme'] for item in batch]
        image_descriptions = [item.get('image_description', '') for item in batch]
        text_contents = [item.get('text', '') for item in batch]

        batch_new = {
            'labels': labels,
            'idx_memes': idx_memes,
            'image_descriptions': image_descriptions,
            'text_contents': text_contents
        }
        
        image_embed_list = []
        text_embed_list = []

        for item in batch:
            pixel_values = self.clip_preprocess(item['image']).unsqueeze(0)
            text = clip.tokenize(item['text'], context_length=77, truncate=True)

            image_features, text_features = self.compute_CLIP_features_without_proj(
                self.clip_model,
                pixel_values.to(self.cfg.device),
                text.to(self.cfg.device)
            )
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

def load_enhanced_dataset(cfg, split):
    """Load enhanced dataset with CoT reasoning support"""
    dataset = Enhanced_Custom_Dataset(
        cfg=cfg, 
        root_folder=cfg.root_dir, 
        dataset=cfg.dataset_name, 
        split=split,
        image_size=cfg.image_size, 
        label=cfg.label, 
        fast=cfg.fast_process
    )
    return dataset

class CoTReasoningDataset(Dataset):
    """Dataset specifically designed for Chain of Thought reasoning"""
    
    def __init__(self, base_dataset: Enhanced_Custom_Dataset, reasoning_steps: int = 3):
        self.base_dataset = base_dataset
        self.reasoning_steps = reasoning_steps
        
        # Reasoning templates for different tasks
        self.reasoning_templates = {
            'hate': {
                'steps': [
                    "Analyze the visual content",
                    "Analyze the text content", 
                    "Analyze the interaction between visual and text",
                    "Identify potential harmful elements",
                    "Make final classification"
                ],
                'guidelines': [
                    "Consider cultural context",
                    "Look for stereotypes or biases",
                    "Check for discriminatory language",
                    "Evaluate the overall impact"
                ]
            },
            'target': {
                'steps': [
                    "Identify mentioned groups or individuals",
                    "Analyze the specificity of targeting",
                    "Determine the scope of targeting",
                    "Classify the target type"
                ],
                'guidelines': [
                    "Look for specific names or groups",
                    "Consider implied targets",
                    "Evaluate the level of specificity"
                ]
            },
            'stance': {
                'steps': [
                    "Analyze the overall tone",
                    "Identify supporting or opposing elements",
                    "Evaluate the strength of stance",
                    "Make final stance classification"
                ],
                'guidelines': [
                    "Look for positive/negative language",
                    "Consider context and implications",
                    "Evaluate neutrality vs. bias"
                ]
            },
            'humor': {
                'steps': [
                    "Identify humor attempts",
                    "Analyze humor type and style",
                    "Evaluate appropriateness",
                    "Make final humor classification"
                ],
                'guidelines': [
                    "Look for jokes or wordplay",
                    "Consider cultural humor norms",
                    "Evaluate potential offensiveness"
                ]
            }
        }
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        base_item = self.base_dataset[idx]
        
        # Get reasoning template for the current task
        task_type = self.base_dataset.label
        template = self.reasoning_templates.get(task_type, self.reasoning_templates['hate'])
        
        # Generate reasoning steps
        reasoning_steps = []
        for i, step in enumerate(template['steps']):
            reasoning_steps.append({
                'step_id': i,
                'step_description': step,
                'guidelines': template['guidelines']
            })
        
        # Enhanced item with reasoning information
        enhanced_item = {
            **base_item,
            'reasoning_steps': reasoning_steps,
            'reasoning_template': template,
            'task_type': task_type
        }
        
        return enhanced_item

def create_cot_reasoning_dataset(cfg, split='train'):
    """Create a dataset with CoT reasoning support"""
    base_dataset = load_enhanced_dataset(cfg, split)
    cot_dataset = CoTReasoningDataset(base_dataset)
    return cot_dataset 