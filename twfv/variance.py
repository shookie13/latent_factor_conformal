import torch

from .bspline import bspline_basis


def build_a2_from_C_and_warp(
    C: torch.Tensor, knots: torch.Tensor, degree: int, u_tilde: torch.Tensor
):
    """
    Compute log-variance and variance using warped B-splines.

    Args:
        C: control points, shape (I, r, M).
        knots: knot vector for the shared basis.
        degree: spline degree.
        u_tilde: warped times, shape (I, T, r).

    Returns:
        log_a2: tensor (I, T, r).
        a2: tensor (I, T, r) = exp(log_a2).
    """
    I, T, r = u_tilde.shape
    _, _, M_ctrl = C.shape
    device = C.device

    log_a2 = torch.empty((I, T, r), device=device)
    a2 = torch.empty_like(log_a2)

    for i in range(I):
        for k in range(r):
            u_vec = u_tilde[i, :, k]
            B = bspline_basis(u_vec, knots, degree)  # (T, M_ctrl)
            s_t = B @ C[i, k].view(M_ctrl, 1)
            s_t = s_t.squeeze(1)
            log_a2[i, :, k] = s_t
            a2[i, :, k] = torch.exp(s_t)

    return log_a2, a2

