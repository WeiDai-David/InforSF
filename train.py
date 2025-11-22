# main.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# load config
from config import cfg

# Dataset
from dataset.office_home import OfficeHomeDataset

# Models
from models.clip_encoder import ClipImageEncoder
from models.llama_classifier import LlamaClassifier

# Fisher
from losses.fisher_utils import compute_fisher_init

# Trainer (InfoSF)
from trainer.trainer_dynamic import InfoSFTrainer


def main():

    print("========== InfoSF Training ==========")

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------
    print("[*] Loading OfficeHome dataset...")

    train_ds = OfficeHomeDataset(cfg.DATA_ROOT)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # -----------------------------------------------------
    # 2. Load CLIP image encoder (frozen)
    # -----------------------------------------------------
    print("[*] Loading CLIP-ViT-L/14 encoder...")

    clip_img_encoder = ClipImageEncoder(cfg.CLIP_WEIGHTS).to(cfg.DEVICE)

    # Freeze all CLIP parameters
    for p in clip_img_encoder.parameters():
        p.requires_grad = False

    # -----------------------------------------------------
    # 3. Load LLaMA classifier (teacher & student)
    # -----------------------------------------------------
    print("[*] Loading LLaMA-2-7B classifier...")

    # teacher (frozen)
    teacher_model = LlamaClassifier(
        llama_path=cfg.LLAMA_WEIGHTS,
        clip_dim=cfg.CLIP_FEATURE_DIM,
        num_classes=cfg.NUM_CLASSES,
        freeze_llama=True
    ).to(cfg.DEVICE)

    # student (same structure but trainable classifier head)
    student_model = LlamaClassifier(
        llama_path=cfg.LLAMA_WEIGHTS,
        clip_dim=cfg.CLIP_FEATURE_DIM,
        num_classes=cfg.NUM_CLASSES,
        freeze_llama=cfg.FREEZE_LLAMA
    ).to(cfg.DEVICE)

    # -----------------------------------------------------
    # 4. Compute teacher Fisher Information
    # -----------------------------------------------------
    print("[*] Computing Fisher Information (teacher θ0)...")

    fisher_list, fisher_idx = compute_fisher_init(
        model=teacher_model,
        dataloader=train_loader,
        max_batches=cfg.FISHER_BATCHES,
        param_ratio=cfg.FISHER_RATIO,
        device=cfg.DEVICE
    )

    print(f"[*] Fisher computed on {len(fisher_idx)} sampled parameters.")

    # -----------------------------------------------------
    # 5. Build optimizer (only student trainable params)
    # -----------------------------------------------------
    print("[*] Building optimizer...")

    opt = optim.Adam(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=cfg.LR
    )

    # -----------------------------------------------------
    # 6. Initialize InfoSF Trainer
    # -----------------------------------------------------
    print("[*] Initializing InfoSF trainer...")

    trainer = InfoSFTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        fisher_list=fisher_list,
        fisher_indices=fisher_idx,
        dataloader=train_loader,
        optimizer=opt,
        device=cfg.DEVICE,
        N1=cfg.N1
    )

    # -----------------------------------------------------
    # 7. Training loop
    # -----------------------------------------------------
    print("[*] Starting training...")

    trainer.fit(cfg.EPOCHS)

    # -----------------------------------------------------
    # 8. Save trained student model
    # -----------------------------------------------------
    print("[*] Saving final student model...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    torch.save(
        student_model.state_dict(),
        os.path.join(cfg.OUTPUT_DIR, "student_final.pth")
    )

    print(f"[✓] Training finished. Model saved at {cfg.OUTPUT_DIR}/student_final.pth")


if __name__ == "__main__":
    main()
