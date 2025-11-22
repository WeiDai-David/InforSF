# evaluate.py

import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset.office_home import OfficeHomeDataset
from models.clip_encoder import ClipImageEncoder
from models.llama_classifier import LlamaClassifier


def evaluate():
    print("========== InfoSF Evaluation ==========")

    # -------------------------------------------------------------
    # 1. Load dataset (same as training)
    # -------------------------------------------------------------
    print("[*] Loading OfficeHome validation set...")

    val_ds = OfficeHomeDataset(cfg.DATA_ROOT)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------------------------------------------------------------
    # 2. Load CLIP image encoder (frozen)
    # -------------------------------------------------------------
    print("[*] Loading CLIP encoder...")

    clip_img_encoder = ClipImageEncoder(cfg.CLIP_WEIGHTS).to(cfg.DEVICE)
    for p in clip_img_encoder.parameters():
        p.requires_grad = False

    # -------------------------------------------------------------
    # 3. Load trained student model
    # -------------------------------------------------------------
    print("[*] Loading trained InfoSF student model...")

    model = LlamaClassifier(
        llama_path=cfg.LLAMA_WEIGHTS,
        clip_dim=cfg.CLIP_FEATURE_DIM,
        num_classes=cfg.NUM_CLASSES,
        freeze_llama=True   # same as training
    ).to(cfg.DEVICE)

    ckpt_path = f"{cfg.OUTPUT_DIR}/student_final.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))

    model.eval()

    # -------------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------------
    print("[*] Evaluating...")

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:

            imgs = imgs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            logits = model(imgs)
            pred = logits.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100.0

    print(f"\n[✓] Accuracy: {acc:.2f}%")
    print("=========================================\n")


if __name__ == "__main__":
    evaluate()
