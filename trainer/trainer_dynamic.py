# trainer/trainer_dynamic.py

import torch
import torch.nn as nn
from tqdm import tqdm

from losses.ratio_loss import ratio_loss
from losses.compute_t import compute_t
from losses.compute_w1 import compute_w1


class InfoSFTrainer:
    """
    Trainer for InfoSF:
        L = CE + w1 * RatioLoss
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        fisher_list,
        fisher_indices,
        dataloader,
        optimizer,
        device="cuda",
        N1=50,
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)

        self.fisher = fisher_list
        self.fisher_idx = fisher_indices

        self.loader = dataloader
        self.opt = optimizer
        self.device = device
        self.N1 = N1

        self.ce = nn.CrossEntropyLoss()


    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()   # frozen

        total_loss = 0
        total_ce = 0
        total_ratio = 0

        for imgs, labels in tqdm(self.loader, desc=f"Epoch {epoch}"):

            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            # --------------------------------------------------------------
            # 1) student forward
            # --------------------------------------------------------------
            logits_s = self.student(imgs)

            # --------------------------------------------------------------
            # 2) teacher forward (with no grad)
            # --------------------------------------------------------------
            with torch.no_grad():
                logits_t = self.teacher(imgs)

            # --------------------------------------------------------------
            # 3) CE loss (supervised)
            # --------------------------------------------------------------
            ce_loss = self.ce(logits_s, labels)

            # --------------------------------------------------------------
            # 4) Ratio loss (new version, target-based)
            # --------------------------------------------------------------
            rat_loss = ratio_loss(logits_t, logits_s, labels)

            # --------------------------------------------------------------
            # 5) Compute Fisher distance t
            # --------------------------------------------------------------
            t_value = compute_t(
                model_student=self.student,
                model_teacher=self.teacher,
                fisher_list=self.fisher,
                indices=self.fisher_idx
            )

            # --------------------------------------------------------------
            # 6) Dynamic w1
            # --------------------------------------------------------------
            w1 = compute_w1(t_value, self.N1)

            # --------------------------------------------------------------
            # 7) Final loss
            # --------------------------------------------------------------
            loss = ce_loss + w1 * rat_loss

            # --------------------------------------------------------------
            # 8) Backprop
            # --------------------------------------------------------------
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Logs
            total_loss  += loss.item()
            total_ce    += ce_loss.item()
            total_ratio += rat_loss.item()

        log_dict = {
            "loss": total_loss / len(self.loader),
            "ce_loss": total_ce / len(self.loader),
            "ratio_loss": total_ratio / len(self.loader),
        }

        return log_dict


    def fit(self, epochs):

        log = {"epochs": [], "ce": [], "ratio": [], "t": [], "w1": []}

        for e in range(1, epochs + 1):
            stats = self.train_epoch(e)

            # append values
            log["epochs"].append(e)
            log["ce"].append(stats["ce_loss"])
            log["ratio"].append(stats["ratio_loss"])

            # compute t and w1 for logging
            t_value = compute_t(
                self.student, self.teacher,
                self.fisher, self.fisher_idx
            )
            w1_value = compute_w1(t_value, self.N1)

            log["t"].append(t_value)
            log["w1"].append(w1_value)

            print(f"[Epoch {e}] loss={stats['loss']:.4f}  "
                f"ce={stats['ce_loss']:.4f}  "
                f"ratio={stats['ratio_loss']:.4f}  "
                f"t={t_value:.6f} w1={w1_value:.6f}")

        # Save JSON
        import json
        with open("output/infoSF_log.json", "w") as f:
            json.dump(log, f, indent=2)


