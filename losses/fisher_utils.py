# losses/fisher_utils.py

import torch
import torch.nn as nn
import random


# --------------------------------------------------------------
# Utility: sample a subset of parameters (1% by default)
# --------------------------------------------------------------
def sample_params(params, ratio=0.01):
    """
    Randomly select ratio (e.g., 1%) of parameters from a list.
    Returns:
        - selected parameter tensors
        - corresponding indices in the original list
    """
    params = list(params)
    total = len(params)
    k = max(1, int(total * ratio))
    idx = sorted(random.sample(range(total), k))
    selected = [params[i] for i in idx]
    return selected, idx



# --------------------------------------------------------------
# Step 1: Compute Fisher Information Approximation of θ₀
#
#   F ≈ mean( grad(log p(y|x; θ₀))^2 )
#
#   You will call this ONCE before training student.
# --------------------------------------------------------------
def compute_fisher_init(model, dataloader, device="cuda", 
                        max_batches=4, param_ratio=0.01):
    """
    Args:
        model: frozen teacher model (LLaMA classifier)
        dataloader: labeled target dataset (few-shot)
        max_batches: number of batches to approximate fisher
        param_ratio: fraction of parameters to use (e.g., 1%)

    Returns:
        fisher_vector: list of same length as sampled params,
                       each element is a tensor of squared gradients
        sampled_indices: positions in original param list
    """

    # Step 1: Sample subset of parameters
    full_params = [p for p in model.parameters() if p.requires_grad is False]
    selected_params, indices = sample_params(full_params, ratio=param_ratio)

    # Create fisher accumulators: same shape as selected params
    fisher_accum = [torch.zeros_like(p, device=device) for p in selected_params]

    model.eval()

    count = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward
        logits = model(imgs)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # We use supervised Fisher: -log p(y|x)
        nll = nn.functional.nll_loss(log_probs, labels)
        
        # Backward wrt teacher params (allowed for fisher)
        model.zero_grad()
        nll.backward()

        # Accumulate squared gradients
        for i, p in enumerate(selected_params):
            if p.grad is not None:
                fisher_accum[i] += (p.grad.detach() ** 2)

        count += 1
        if count >= max_batches:
            break

    # Average across batches
    for i in range(len(fisher_accum)):
        fisher_accum[i] /= count

    return fisher_accum, indices



# --------------------------------------------------------------
# Step 2: compute Fisher distance:
#
#   t = (θ - θ₀)^T F (θ - θ₀) / d
#
# --------------------------------------------------------------
def compute_fisher_distance(model, model_init, fisher_accum, indices,
                            device="cuda"):
    """
    Args:
        model       : current student
        model_init  : frozen initial teacher
        fisher_accum: approx FIM values for sampled params
        indices     : which parameters were sampled
    
    Returns:
        scalar t
    """

    # flatten parameter lists
    student_params = [p for p in model.parameters()]
    init_params    = [p for p in model_init.parameters()]

    t_num = 0.0
    t_den = 0.0

    for F, idx in zip(fisher_accum, indices):
        p  = student_params[idx]
        p0 = init_params[idx]

        diff = (p - p0).detach()
        t_num += torch.sum(F * diff * diff).item()
        t_den += F.numel()

    t = t_num / (t_den + 1e-8)
    return t
