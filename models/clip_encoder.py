# models/clip_encoder.py

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class ClipImageEncoder(nn.Module):
    """
    A wrapper around CLIP ViT-L/14 image encoder.
    Loads from local path: weights/clip/
    """

    def __init__(self, clip_path="weights/clip", device="cuda"):
        super().__init__()

        print(f"[CLIP] Loading CLIP ViT-L/14 from {clip_path} ...")

        self.processor = CLIPProcessor.from_pretrained(
            clip_path,
            local_files_only=True
        )
        self.model = CLIPModel.from_pretrained(
            clip_path,
            local_files_only=True
        ).to(device)

        # Usually CLIP ViT-L/14 → 768 or 1024 dims (depending on version)
        self.feature_dim = self.model.visual_projection.out_features
        print(f"[CLIP] Feature dimension = {self.feature_dim}")

        self.device = device

    @torch.no_grad()
    def encode_image(self, pil_image):
        """
        Encode a single PIL image → feature vector (tensor)
        """
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.get_image_features(**inputs)
        return outputs  # shape [1, feature_dim]

    @torch.no_grad()
    def forward(self, images):
        """
        Encode batch of images.
        images: Tensor or PIL list (depending on train pipeline)
        """
        if isinstance(images, list):  # List of PIL images
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            raise ValueError("CLIP encoder currently expects list of PIL images.")

        image_features = self.model.get_image_features(**inputs)
        return image_features  # [B, feature_dim]
