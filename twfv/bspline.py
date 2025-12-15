import torch


def make_open_uniform_knots(num_ctrl: int, degree: int, device=None) -> torch.Tensor:
    """
    Build an open-uniform knot vector on [0, 1].

    Args:
        num_ctrl: number of control points M.
        degree: spline degree q.
        device: optional torch device.

    Returns:
        Tensor of shape (num_ctrl + degree + 1,).
    """
    if num_ctrl <= degree:
        raise ValueError("num_ctrl must exceed degree")
    interior = num_ctrl - degree - 1
    knots = torch.zeros(num_ctrl + degree + 1, device=device)
    if interior > 0:
        knots[degree:-degree] = torch.linspace(0.0, 1.0, interior + 2, device=device)
    knots[-(degree + 1):] = 1.0
    return knots


def bspline_basis(u: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate B-spline basis at points u using Cox-de Boor.

    Args:
        u: tensor (T,) in [0, 1].
        knots: tensor (K + degree + 2,) nondecreasing.
        degree: spline degree.

    Returns:
        Basis matrix B of shape (T, M) where M = len(knots) - degree - 1.
    """
    device = u.device
    p = degree
    n_basis = knots.numel() - p - 1
    if n_basis <= 0:
        raise ValueError("invalid knot configuration")

    # Degree 0
    pieces = []
    for i in range(n_basis):
        cond = (u >= knots[i]) & (u < knots[i + 1])
        if i == n_basis - 1:
            cond = cond | (u == knots[i + 1])
        pieces.append(cond.to(u.dtype))
    B = torch.stack(pieces, dim=1)  # (T, M)

    for k in range(1, p + 1):
        B_next = torch.zeros_like(B)
        for i in range(n_basis):
            left_den = (knots[i + k] - knots[i]).clamp_min(1e-12)
            right_den = (knots[i + k + 1] - knots[i + 1]).clamp_min(1e-12)

            left_num = (u - knots[i]) * B[:, i]
            right_num = (knots[i + k + 1] - u) * (B[:, i + 1] if i + 1 < n_basis else torch.zeros_like(B[:, 0]))

            B_next[:, i] = left_num / left_den + right_num / right_den
        B = B_next
    return B

