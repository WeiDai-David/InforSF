# losses/ratio_loss.py

import torch
import torch.nn.functional as F

def ratio_loss_p0_over_pt(logits_t, logits_0, labels, eps=1e-8):
    """
    L2 = log( p0 / pt )
    """
    p_t = F.softmax(logits_t, dim=-1)
    p_0 = F.softmax(logits_0, dim=-1)

    pt = p_t[range(len(labels)), labels]
    p0 = p_0[range(len(labels)), labels]

    ratio = torch.log((p0 + eps) / (pt + eps))
    return torch.mean(ratio)
