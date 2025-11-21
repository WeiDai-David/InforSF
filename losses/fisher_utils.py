# losses/fisher_utils.py

import torch

def estimate_fisher_diagonal(model, dataloader, loss_fn, num_samples=128, scale=0.01, device="cuda"):
    fisher = None
    count = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = loss_fn(logits, labels)

        model.zero_grad()
        loss.backward()

        grad_vec = []
        for p in model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.detach().clone().flatten())
        grad_vec = torch.cat(grad_vec)

        grad_abs = grad_vec.abs()

        if fisher is None:
            fisher = grad_abs
        else:
            fisher = 0.9*fisher + 0.1*grad_abs

        count += 1
        if count >= num_samples:
            break

    fisher = fisher * scale
    return fisher
