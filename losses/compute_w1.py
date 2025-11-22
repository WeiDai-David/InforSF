# losses/compute_w1.py

def compute_w1(t_value, N1):
    """
    Compute dynamic weight w1 from fisher distance t.

    w1 = 1 / (1 + t * N1)

    Args:
        t_value : scalar fisher distance
        N1      : hyperparameter (e.g., 10, 50, 100...)

    Returns:
        scalar w1 (float)
    """

    w1 = 1.0 / (1.0 + t_value * N1)
    return w1
