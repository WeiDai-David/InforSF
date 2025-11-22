# losses/ratio_loss.py

import torch
import torch.nn as nn

def ratio_loss(logits_teacher, logits_student, labels):
    """
    InfoSF Ratio Loss:
    L_ratio = (p0 / pt) * (-log pt)
    
    Args:
        logits_teacher : teacher model outputs (batch, C)
        logits_student : student model outputs (batch, C)
        labels         : ground truth labels (batch,)

    Returns:
        scalar ratio loss
    """

    # Convert logits to probability
    p0 = torch.softmax(logits_teacher, dim=-1)   # teacher prob
    pt = torch.softmax(logits_student, dim=-1)   # student prob

    # gather p0 and pt for the ground truth class y
    p0_y = p0.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt_y = pt.gather(1, labels.unsqueeze(1)).squeeze(1)

    # Student CE: -log pt(y)
    ce_y = -torch.log(pt_y + 1e-12)

    # Probability ratio: p0(y) / pt(y)
    ratio = p0_y / (pt_y + 1e-12)

    # Final InfoSF ratio loss:
    loss = ratio * ce_y
    return torch.mean(loss)
