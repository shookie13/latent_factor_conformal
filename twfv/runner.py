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
    warp_eta: float = 1.0,
    collect_diagnostics: bool = False,
    diagnostics_t_head: int = 8,
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

    # Parameters
    L = nn.Parameter(torch.randn(J, r, device=device) * 0.1)
    psi_raw = nn.Parameter(torch.zeros(J, device=device))
    s_raw = nn.Parameter(torch.zeros(I, device=device))
    C = nn.Parameter(torch.zeros(I, r, M_ctrl, device=device))

    u_lin = torch.linspace(0.0, 1.0, T, device=device)
    u_tilde = torch.stack([u_lin for _ in range(I * r)]).view(I, T, r)

    optim = torch.optim.Adam([L, psi_raw, s_raw, C], lr=lr)

    history: dict[str, list[float]] = {"ll_outer": [], "ll_post_m": []}
    if collect_diagnostics:
        # Control point C[:,:,0] drift stats
        history.update(
            {
                "C0_mean": [],
                "C0_std": [],
                "C0_max_abs": [],
                # Early-time warp / variance summaries (averaged over subjects and factors)
                "u_head_mean": [],          # length diagnostics_t_head (flattened into list)
                "u_head_inc_mean": [],      # length diagnostics_t_head-1
                "u_head_inc_min": [],       # scalar
                "a2_head_mean": [],         # length diagnostics_t_head
                "a2_head_max": [],          # length diagnostics_t_head
            }
        )
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
            u_new = update_time_warp(F_tilde)
            eta = float(warp_eta)
            if eta <= 0.0:
                # freeze warp
                pass
            elif eta >= 1.0:
                u_tilde = u_new
            else:
                # Damp warp updates for stability; preserves monotonicity and endpoints.
                u_tilde = (1.0 - eta) * u_tilde + eta * u_new
            history["ll_outer"].append(float(ll_val.detach().cpu().item()))

            if collect_diagnostics:
                # Keep this extremely cheap: summaries only.
                c0 = C[:, :, 0]  # (I, r)
                history["C0_mean"].append(float(c0.mean().detach().cpu().item()))
                history["C0_std"].append(float(c0.std(unbiased=False).detach().cpu().item()))
                history["C0_max_abs"].append(float(c0.abs().max().detach().cpu().item()))

                t_head = int(min(max(diagnostics_t_head, 1), T))
                # Warp summaries
                u_head = u_tilde[:, :t_head, :].mean(dim=(0, 2))  # (t_head,)
                if t_head >= 2:
                    du = (u_tilde[:, 1:t_head, :] - u_tilde[:, : t_head - 1, :]).mean(dim=(0, 2))  # (t_head-1,)
                    du_min = (u_tilde[:, 1:t_head, :] - u_tilde[:, : t_head - 1, :]).min().item()
                else:
                    du = torch.empty((0,), device=u_tilde.device)
                    du_min = float("nan")
                history["u_head_mean"].append(u_head.detach().cpu().tolist())
                history["u_head_inc_mean"].append(du.detach().cpu().tolist())
                history["u_head_inc_min"].append(float(du_min))

                # Variance summaries at early times: average and max over (i,k)
                a2_head = a2[:, :t_head, :]  # (I, t_head, r)
                a2_mean = a2_head.mean(dim=(0, 2))  # (t_head,)
                a2_max = a2_head.amax(dim=(0, 2))   # (t_head,)
                history["a2_head_mean"].append(a2_mean.detach().cpu().tolist())
                history["a2_head_max"].append(a2_max.detach().cpu().tolist())

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
                # Shift the *entire* spline curve by subtracting a constant from all control points.
                # B-spline bases form a partition of unity, so adding the same constant to every
                # control point shifts log_a2(t) by that constant for all t (up to numerical error).
                C.data -= mean_log.squeeze(1).unsqueeze(-1)

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
    if use_woodbury:
        F_tilde, _ = factor_E_step_fast(Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups)
    else:
        F_tilde, _ = factor_E_step(Y, M_mask, L, psi, s, a2)

    if return_history:
        return L, a2, psi, s, C, u_tilde, F_tilde, history
    return L, a2, psi, s, C, u_tilde, F_tilde

