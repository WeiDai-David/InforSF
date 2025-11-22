# models/llama_classifier.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LLaMAClassifier(nn.Module):
    """
    LLaMA2-7B classifier using CLIP features as input.
    Suitable for InfoSF (Source-Free Transfer) where LLaMA remains frozen.
    """

    def __init__(
        self,
        llama_path="weights/llama2-7b-hf",
        embed_dim=1024,      # CLIP ViT-L/14
        num_classes=65,
        freeze_llama=True,
        device="cuda"
    ):
        super().__init__()

        self.device = device

        # -----------------------------------------------------
        # Load tokenizer (necessary only for model initialization)
        # -----------------------------------------------------
        print(f"[LLaMAClassifier] Loading tokenizer from {llama_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llama_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # -----------------------------------------------------
        # Load LLaMA model
        # -----------------------------------------------------
        print(f"[LLaMAClassifier] Loading LLaMA-2-7B from {llama_path} ...")

        self.llama = AutoModel.from_pretrained(
            llama_path,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)

        # Get hidden size
        self.hidden_dim = self.llama.config.hidden_size  # = 4096 for LLaMA2-7B

        # -----------------------------------------------------
        # Freeze LLaMA backbone (recommended for InfoSF)
        # -----------------------------------------------------
        if freeze_llama:
            for p in self.llama.parameters():
                p.requires_grad = False
            print("[LLaMAClassifier] LLaMA backbone is frozen.")

        # -----------------------------------------------------
        # CLIP feature → LLaMA embedding space projection
        # -----------------------------------------------------
        self.feature_proj = nn.Linear(embed_dim, self.hidden_dim)

        # -----------------------------------------------------
        # Classification head
        # -----------------------------------------------------
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    # ---------------------------------------------------------
    # Forward: clip_features ∈ R^{B×1024}
    # ---------------------------------------------------------
    def forward(self, clip_features):
        """
        clip_features: Tensor [B, embed_dim]

        Pipeline:
            CLIP embed (1024)
                -> projection (4096)
                -> expand 1-token sequence
                -> LLaMA transformer
                -> CLS representation (token 0)
                -> linear classifier
        """

        # 1) map to LLaMA hidden dim
        x = self.feature_proj(clip_features)      # [B, 4096]

        # 2) LLaMA expects a sequence → treat as 1-token sequence
        x = x.unsqueeze(1)                        # [B, 1, 4096]

        # 3) Forward through LLaMA
        outputs = self.llama(inputs_embeds=x)
        last_hidden = outputs.last_hidden_state    # [B, 1, 4096]

        # 4) CLS token = position 0
        cls = last_hidden[:, 0, :]                # [B, 4096]

        # 5) Classifier
        logits = self.classifier(cls)             # [B, 65]

        return logits
