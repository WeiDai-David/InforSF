# vision/clip_encoder.py

import torch
from transformers import CLIPModel

class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()
        self.model = CLIPModel.from_pretrained(
            clip_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.vision = self.model.vision_model

    def forward(self, imgs):
        out = self.vision(imgs)
        pooled = out[1]  # CLS
        return pooled.float()
