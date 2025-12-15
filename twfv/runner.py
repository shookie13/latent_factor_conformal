import torch
from torch import nn

from .variance import build_a2_from_C_and_warp
from .warp import update_time_warp
from .bspline import make_open_uniform_knots
from .factor import factor_E_step, factor_log_likelihood


def run_em_like(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    r: int,
    M_ctrl: int,
    degree: int = 3,
    max_outer: int = 20,
    grad_steps: int = 5,
    lr: float = 1e-2,
    device=None,
):
    """
    Minimal EM-like loop following the provided specification.

    Args:
        Y: (I, T, J) residual tensor.
        M_mask: bool mask (I, T, J).
        r: number of factors.
        M_ctrl: number of B-spline control points per curve.
        degree: spline degree.
        max_outer: outer iterations (E/M alternation).
        grad_steps: gradient steps per outer iteration.
        lr: optimizer learning rate.
        device: torch device (e.g., "cpu" or "cuda").

    Returns:
        L, psi, s, C, u_tilde, F_tilde
    """
    Y = Y.to(device)
    M_mask = M_mask.to(device)
    I, T, J = Y.shape
    knots = make_open_uniform_knots(M_ctrl, degree, device=device)

    # Parameters
    L = nn.Parameter(torch.randn(J, r, device=device) * 0.1)
    psi_raw = nn.Parameter(torch.zeros(J, device=device))
    s_raw = nn.Parameter(torch.zeros(I, device=device))
    C = nn.Parameter(torch.zeros(I, r, M_ctrl, device=device))

    u_lin = torch.linspace(0.0, 1.0, T, device=device)
    u_tilde = torch.stack([u_lin for _ in range(I * r)]).view(I, T, r)

    optim = torch.optim.Adam([L, psi_raw, s_raw, C], lr=lr)

    prev_ll = None
    for _ in range(max_outer):
        with torch.no_grad():
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            F_tilde, ll_val = factor_E_step(Y, M_mask, L, psi, s, a2)
            u_tilde = update_time_warp(F_tilde)

        for _ in range(grad_steps):
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            log_a2, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            ll = factor_log_likelihood(Y, M_mask, L, psi, s, a2)
            loss = -ll
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Optional centering: keep mean log variance near zero per (i, k).
            with torch.no_grad():
                mean_log = log_a2.mean(dim=1, keepdim=True)  # (I,1,r)
                C.data[:, :, 0] -= mean_log.squeeze(1)

        if prev_ll is not None and torch.abs(ll_val - prev_ll) < 1e-3:
            break
        prev_ll = ll_val

    psi = torch.nn.functional.softplus(psi_raw) + 1e-4
    s = torch.nn.functional.softplus(s_raw) + 1e-4
    _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
    F_tilde, _ = factor_E_step(Y, M_mask, L, psi, s, a2)

    return L, psi, s, C, u_tilde, F_tilde

