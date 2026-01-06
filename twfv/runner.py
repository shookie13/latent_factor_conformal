import torch
from torch import nn

from .variance import build_a2_from_C_and_warp
from .warp import update_time_warp
from .bspline import make_open_uniform_knots
from .factor import (
    build_mask_pattern_groups,
    factor_E_step,
    factor_E_step_fast,
    factor_log_likelihood,
    factor_log_likelihood_fast,
)
from .woodbury import factor_log_likelihood_woodbury_minibatch


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
    use_woodbury: bool = True,
    return_history: bool = True,
    minibatch_size: int | None = None,   # NEW
    minibatch_seed: int | None = None,   # NEW (optional)
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
    pattern_groups = build_mask_pattern_groups(M_mask) if use_woodbury else None
    # NEW: list of valid flattened (i,t) rows (those with >=1 observed channel)
    if use_woodbury:
        valid_flat = torch.cat([g.flat for g in pattern_groups], dim=0)  # (N_valid,)
        N_valid = int(valid_flat.numel())
        gen = None
        if minibatch_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(minibatch_seed)
    # Parameters
    L = nn.Parameter(torch.randn(J, r, device=device) * 0.1)
    psi_raw = nn.Parameter(torch.zeros(J, device=device))
    s_raw = nn.Parameter(torch.zeros(I, device=device))
    C = nn.Parameter(torch.zeros(I, r, M_ctrl, device=device))

    u_lin = torch.linspace(0.0, 1.0, T, device=device)
    u_tilde = torch.stack([u_lin for _ in range(I * r)]).view(I, T, r)

    optim = torch.optim.Adam([L, psi_raw, s_raw, C], lr=lr)

    history: dict[str, list[float]] = {"ll_outer": [], "ll_post_m": []}
    prev_ll = None
    for _ in range(max_outer):
        with torch.no_grad():
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            if use_woodbury:
                F_tilde, ll_val = factor_E_step_fast(
                    Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups
                )
            else:
                F_tilde, ll_val = factor_E_step(Y, M_mask, L, psi, s, a2)
            u_tilde = update_time_warp(F_tilde)
            history["ll_outer"].append(float(ll_val.detach().cpu().item()))

        for _ in range(grad_steps):
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            log_a2, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            if use_woodbury:
                ll = factor_log_likelihood_fast(
                    Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups
                )
            else:
                ll = factor_log_likelihood(Y, M_mask, L, psi, s, a2)
            loss = -ll
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Optional centering: keep mean log variance near zero per (i, k).
            with torch.no_grad():
                mean_log = log_a2.mean(dim=1, keepdim=True)  # (I,1,r)
                C.data[:, :, 0] -= mean_log.squeeze(1)

        with torch.no_grad():
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            if use_woodbury:
                ll_post = factor_log_likelihood_fast(
                    Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups
                )
            else:
                ll_post = factor_log_likelihood(Y, M_mask, L, psi, s, a2)
            history["ll_post_m"].append(float(ll_post.detach().cpu().item()))

        if prev_ll is not None and torch.abs(ll_val - prev_ll) < 1e-3:
            break
        prev_ll = ll_val

    psi = torch.nn.functional.softplus(psi_raw) + 1e-4
    s = torch.nn.functional.softplus(s_raw) + 1e-4
    _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
    if use_woodbury and (minibatch_size is not None) and (minibatch_size < N_valid):
        # NEW: sample WITHOUT replacement from valid_flat
        perm = torch.randperm(N_valid, device=device, generator=gen)[:minibatch_size]
        flat_batch = valid_flat.index_select(0, perm)

        ll = factor_log_likelihood_woodbury_minibatch(
            Y, M_mask, L, psi, s, a2,
            pattern_groups=pattern_groups,
            flat_batch=flat_batch,
            total_n=N_valid,
        )
    elif use_woodbury:
        F_tilde, _ = factor_E_step_fast(Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups)
    else:
        F_tilde, _ = factor_E_step(Y, M_mask, L, psi, s, a2)

    if return_history:
        return L, a2, psi, s, C, u_tilde, F_tilde, history
    return L, a2, psi, s, C, u_tilde, F_tilde

