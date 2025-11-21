# utils/metrics.py

import torch

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()
