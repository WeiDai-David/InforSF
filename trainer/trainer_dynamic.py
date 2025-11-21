# trainer/trainer_dynamic.py

import torch
import torch.nn as nn

from losses.ratio_loss import ratio_loss_p0_over_pt
from losses.fisher_utils import estimate_fisher_diagonal
from losses.compute_t import compute_t
from losses.compute_w1 import compute_W1


class DynamicTrainer:
    def __init__(self, model, model_init, feature_extractor, dataloader, cfg):
        self.model = model
        self.model_init = model_init
        self.feature_extractor = feature_extractor
        self.loader = dataloader
        self.cfg = cfg

        self.ce = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        self.theta_init = self._flatten_params(self.model_init)
        self.W1 = 0.0

        print(">>> A2: Dynamic Fisher + Adaptive Transfer <<<")

    def _flatten_params(self, model):
        parts = []
        for p in model.parameters():
            parts.append(p.detach().clone().flatten())
        return torch.cat(parts)

    def train(self):
        for epoch in range(self.cfg.epochs):

            print(f"\n===== Epoch {epoch} =====")
            print(f"W1 = {self.W1:.6f}")

            # (1) 用当前 W1 做一轮优化
            self.model.train()
            for imgs, labels in self.loader:
                imgs = imgs.cuda()
                labels = labels.cuda()

                feat = self.feature_extractor(imgs)

                logits_t = self.model(feat)
                with torch.no_grad():
                    logits_0 = self.model_init(feat)

                L_sup = self.ce(logits_t, labels)
                L_ratio = ratio_loss_p0_over_pt(logits_t, logits_0, labels)
                L_total = L_sup + self.W1 * L_ratio

                self.opt.zero_grad()
                L_total.backward()
                self.opt.step()

            print(f"L_sup={L_sup.item():.4f}, L_ratio={L_ratio.item():.4f}")

            # (2) 每个 epoch 估计一次 Fisher
            print("Estimating Fisher (abs grad + EMA + 1%) ...")
            fisher_diag = estimate_fisher_diagonal(
                model=self.model,
                dataloader=self.loader,
                loss_fn=self.ce,
                num_samples=self.cfg.fisher_samples,
                scale=self.cfg.fisher_scale,
                device="cuda"
            ).cuda()

            # (3) 更新 t
            theta_now = self._flatten_params(self.model).cuda()
            t_val = compute_t(theta_now, self.theta_init.cuda(), fisher_diag)
            print(f"t = {t_val:.6f}")

            # (4) 更新 W1
            self.W1 = compute_W1(t_val, self.cfg.N1)
            print(f"W1 updated to {self.W1:.6f}")
