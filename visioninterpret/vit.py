from PIL import Image, ImageFilter
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import torch 

from lib.processor import Processor

class VitAttentionProcessor(Processor):
    def __init__(self, processor: ViTImageProcessor, model: ViTForImageClassification):
        super().__init__(processor, model)
    
    def get_layer_attentions(self, image: np.ndarray | Image.Image) -> list[torch.Tensor]:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs, output_attentions=True)
        return outputs.attentions

    def get_attention_rollout(self, layer_attentions: list[torch.Tensor]) -> np.ndarray:
        rollout = torch.eye(layer_attentions[0].size(-1)).to(layer_attentions[0].device)
        
        for attention in layer_attentions:
            layer_mean = attention.mean(dim=1)
            residual_attention = layer_mean + torch.eye(layer_mean.size(-1)).to(layer_mean.device)
            normalized_residual_attention = residual_attention / residual_attention.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(rollout, normalized_residual_attention)
        return rollout
    
    def grayscale_attention_map(self, cls_attention: np.ndarray, image: np.ndarray | Image.Image) -> np.ndarray:
        cls_attention = Image.fromarray((cls_attention * 255).astype(np.uint8)).resize((image.width, image.height), resample=Image.Resampling.BICUBIC)
        cls_attention = cls_attention.filter(ImageFilter.GaussianBlur(radius=2)) 
        return cls_attention

    def colored_attention_map(self, grayscale_attention: np.ndarray, alpha_overlay: int = 100) -> np.ndarray:
        grayscale_attention = np.array(grayscale_attention.convert("L"))
        colored_attention = np.stack([grayscale_attention]*3 + [grayscale_attention], axis=-1)
        colored_attention = Image.fromarray(colored_attention, mode="RGBA")
        colored_attention.putalpha(alpha_overlay) 
        return colored_attention
    
    def blend_image_with_attention(self, image: np.ndarray | Image.Image, colored_attention: np.ndarray | Image.Image) -> np.ndarray:
        image = image.convert("RGBA")
        colored_attention = colored_attention.resize(image.size)
        return Image.alpha_composite(image, colored_attention)

    def process(self, image: np.ndarray | Image.Image, alpha_overlay: int = 100) -> np.ndarray:
        layer_attentions = self.get_layer_attentions(image)
        attention_rollout = self.get_attention_rollout(layer_attentions)
        patch_size = np.sqrt(attention_rollout.shape[1] - 1).astype(int)
        
        cls_attention = attention_rollout[0, 0, 1:]
        cls_attention = 1 - cls_attention.reshape(patch_size, patch_size)
        cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
        cls_attention = cls_attention.detach().cpu().numpy()
        
        gray_scale_attention = self.grayscale_attention_map(cls_attention, image)
        colored_attention = self.colored_attention_map(gray_scale_attention, alpha_overlay)
        
        return self.blend_image_with_attention(image, colored_attention)
    