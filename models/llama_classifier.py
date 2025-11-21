# models/llama_classifier.py

import torch
import torch.nn as nn
from transformers import AutoModel

class LLaMAClassifier(nn.Module):
    def __init__(self, llama_model, feature_dim, num_classes):
        super().__init__()
        self.llama = AutoModel.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        hidden = self.llama.config.hidden_size
        self.project = nn.Linear(feature_dim, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, feat):
        emb = self.project(feat)
        logits = self.classifier(emb)
        return logits
