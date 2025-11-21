# losses/compute_t.py

import torch

def compute_t(theta_now, theta_init, fisher_diag):
    delta = theta_now - theta_init
    d = len(delta)
    val = torch.sum(delta * fisher_diag * delta) / d
    return val.item()
