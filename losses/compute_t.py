# losses/compute_t.py

import torch

def compute_t(model_student, model_teacher, fisher_list, indices):
    """
    Compute Fisher distance t between teacher θ0 and student θ.

    Args:
        model_student : nn.Module (student model)
        model_teacher : nn.Module (teacher model, frozen)
        fisher_list   : list of tensors, Fisher diagonal entries for sampled params
        indices       : list of param indices sampled from the model

    Returns:
        scalar t
    """

    # flatten model parameters
    student_params = [p for p in model_student.parameters()]
    teacher_params = [p for p in model_teacher.parameters()]

    num = 0.0   # numerator
    den = 0.0   # denominator (sum of fisher elements count)

    for F_i, idx in zip(fisher_list, indices):
        p_t = student_params[idx]
        p_0 = teacher_params[idx]

        diff = (p_t - p_0).detach()

        num += torch.sum(F_i * diff * diff).item()
        den += F_i.numel()

    t_value = num / (den + 1e-8)
    return t_value
