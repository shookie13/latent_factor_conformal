import torch
from torch import nn
import time
import numpy as np

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
    freeze_warp: bool = False,
    collect_diagnostics: bool = False,
    diagnostics_t_head: int = 8,
    diagnostics_n_H_pairs: int = 20,
    diagnostics_seed: int = 0,
    lambda_C: float = 1e-2,
    lambda_L: float = 1e-4,
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
    kappa_raw = nn.Parameter(torch.zeros(I, device=device))
    C = nn.Parameter(torch.zeros(I, r, M_ctrl, device=device))

    u_lin = torch.linspace(0.0, 1.0, T, device=device)
    u_tilde = torch.stack([u_lin for _ in range(I * r)]).view(I, T, r)

    optim = torch.optim.Adam([L, psi_raw, s_raw, kappa_raw, C], lr=lr)

    history: dict[str, list[float]] = {
        "ll_outer": [],
        "ll_post_m": [],
        # timing (seconds)
        "t_outer_total_s": [],   # total time per outer iteration
        "t_estep_s": [],         # E-step time per outer iteration
        "t_warp_s": [],          # warp update time per outer iteration
        "t_mstep_s": [],         # grad_steps time per outer iteration
        "t_post_s": [],          # post-M likelihood eval time per outer iteration
        "t_total_s": [],         # cumulative wall time since function start
    }
    if collect_diagnostics:
        # Control point C[:,:,0] drift stats
        history.update(
            {
                "C0_mean": [],
                "C0_std": [],
                "C0_max_abs": [],
                # Subject-scale (factor) diagnostics
                "kappa_mean": [],
                "kappa_std": [],
                "kappa_max": [],
                # Early-time warp / variance summaries (averaged over subjects and factors)
                "u_head_mean": [],          # length diagnostics_t_head (flattened into list)
                "u_head_inc_mean": [],      # length diagnostics_t_head-1
                "u_head_inc_min": [],       # scalar
                "a2_head_mean": [],         # length diagnostics_t_head
                "a2_head_max": [],          # length diagnostics_t_head
                # Parameter-change diagnostics (relative)
                "dL_rel": [],
                "dC_rel": [],
                "dlogpsi_l2": [],
                "dlogs_l2": [],
                "dlogkappa_l2": [],
                "du_rel": [],
                "da2_rel": [],
                # Identifiability-aware change metrics:
                # Var(y) surface change (scalar summaries)
                "dvar_y_rmse": [],
                "dvar_y_rel_l2": [],
                "dvar_y_max_rel": [],
                # H_it covariance change on a fixed sample of observed patterns
                "dH_frob_mean": [],
                "dH_rel_frob_mean": [],
                "dH_rel_frob_max": [],
            }
        )
        # State carried across outer iterations for delta computations
        prev_state: dict[str, torch.Tensor] = {}
        prev_var_y: torch.Tensor | None = None

        # Pre-sample (i,t) pairs that have at least one observed channel for H_it diagnostics.
        with torch.no_grad():
            valid_rows = torch.nonzero(M_mask.any(dim=2).reshape(-1), as_tuple=False).squeeze(1)  # flat indices
            if valid_rows.numel() > 0:
                g = torch.Generator(device=device)
                g.manual_seed(int(diagnostics_seed))
                n_take = int(min(max(diagnostics_n_H_pairs, 1), valid_rows.numel()))
                perm = torch.randperm(valid_rows.numel(), generator=g, device=device)[:n_take]
                flat_sample = valid_rows.index_select(0, perm).detach().cpu().tolist()
                sample_pairs = [(int(f // T), int(f % T)) for f in flat_sample]
            else:
                sample_pairs = []
    t_start = time.perf_counter()
    prev_ll = None
    for _ in range(max_outer):
        t_outer0 = time.perf_counter()
        t_estep0 = time.perf_counter()
        with torch.no_grad():
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            kappa = torch.nn.functional.softplus(kappa_raw) + 1e-4
            _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            if use_woodbury:
                F_tilde, ll_val = factor_E_step_fast(
                    Y, M_mask, L, psi, s, kappa, a2, pattern_groups=pattern_groups
                )
            else:
                F_tilde, ll_val = factor_E_step(Y, M_mask, L, psi, s, kappa, a2)
        t_estep1 = time.perf_counter()
        t_warp0 = time.perf_counter()
        with torch.no_grad():
            u_new = update_time_warp(F_tilde)
            eta = 0.0 if bool(freeze_warp) else float(warp_eta)
            if eta <= 0.0:
                # freeze warp
                pass
            elif eta >= 1.0:
                u_tilde = u_new
            else:
                # Damp warp updates for stability; preserves monotonicity and endpoints.
                u_tilde = (1.0 - eta) * u_tilde + eta * u_new
            history["ll_outer"].append(float(ll_val.detach().cpu().item()))
        t_warp1 = time.perf_counter()

        if collect_diagnostics:
            # Keep this extremely cheap: summaries only.
            with torch.no_grad():
                c0 = C[:, :, 0]  # (I, r)
                history["C0_mean"].append(float(c0.mean().detach().cpu().item()))
                history["C0_std"].append(float(c0.std(unbiased=False).detach().cpu().item()))
                history["C0_max_abs"].append(float(c0.abs().max().detach().cpu().item()))

                # kappa summaries (positive)
                history["kappa_mean"].append(float(kappa.mean().detach().cpu().item()))
                history["kappa_std"].append(float(kappa.std(unbiased=False).detach().cpu().item()))
                history["kappa_max"].append(float(kappa.max().detach().cpu().item()))

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

        t_mstep0 = time.perf_counter()
        for _ in range(grad_steps):
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            kappa = torch.nn.functional.softplus(kappa_raw) + 1e-4
            log_a2, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            if use_woodbury:
                ll = factor_log_likelihood_fast(
                    Y, M_mask, L, psi, s, kappa, a2, pattern_groups=pattern_groups
                )
            else:
                ll = factor_log_likelihood(Y, M_mask, L, psi, s, kappa, a2)
            loss = -ll

            # Smoothness penalty on spline control points (second differences along control index)
            if float(lambda_C) != 0.0 and C.shape[-1] >= 3:
                diff2 = C[:, :, 2:] - 2.0 * C[:, :, 1:-1] + C[:, :, :-2]  # (I, r, M_ctrl-2)
                pen_C = (diff2 * diff2).sum()
                loss = loss + float(lambda_C) * pen_C

            # L2 penalty on loadings
            if float(lambda_L) != 0.0:
                pen_L = (L * L).sum()
                loss = loss + float(lambda_L) * pen_L
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
        t_mstep1 = time.perf_counter()

        t_post0 = time.perf_counter()
        with torch.no_grad():
            psi = torch.nn.functional.softplus(psi_raw) + 1e-4
            s = torch.nn.functional.softplus(s_raw) + 1e-4
            kappa = torch.nn.functional.softplus(kappa_raw) + 1e-4
            _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
            if use_woodbury:
                ll_post = factor_log_likelihood_fast(
                    Y, M_mask, L, psi, s, kappa, a2, pattern_groups=pattern_groups
                )
            else:
                ll_post = factor_log_likelihood(Y, M_mask, L, psi, s, kappa, a2)
            history["ll_post_m"].append(float(ll_post.detach().cpu().item()))
        t_post1 = time.perf_counter()

        # --- change diagnostics (end-of-outer snapshot) ---
        if collect_diagnostics:
            with torch.no_grad():
                eps = 1e-12

                def rel_frob(a: torch.Tensor, b: torch.Tensor) -> float:
                    # ||a-b||_F / (||b||_F + eps)
                    num = torch.linalg.norm(a - b)
                    den = torch.linalg.norm(b) + eps
                    return float((num / den).detach().cpu().item())

                # Current snapshot
                psi_snap = torch.nn.functional.softplus(psi_raw) + 1e-4
                s_snap = torch.nn.functional.softplus(s_raw) + 1e-4
                kappa_snap = torch.nn.functional.softplus(kappa_raw) + 1e-4
                logpsi = torch.log(psi_snap)
                logs = torch.log(s_snap)
                logkappa = torch.log(kappa_snap)

                if "L" in prev_state:
                    history["dL_rel"].append(rel_frob(L, prev_state["L"]))
                    history["dC_rel"].append(rel_frob(C, prev_state["C"]))
                    history["dlogpsi_l2"].append(
                        float(torch.linalg.norm(logpsi - prev_state["logpsi"]).detach().cpu().item())
                    )
                    history["dlogs_l2"].append(
                        float(torch.linalg.norm(logs - prev_state["logs"]).detach().cpu().item())
                    )
                    history["dlogkappa_l2"].append(
                        float(torch.linalg.norm(logkappa - prev_state["logkappa"]).detach().cpu().item())
                    )
                    history["du_rel"].append(rel_frob(u_tilde, prev_state["u_tilde"]))
                    history["da2_rel"].append(rel_frob(a2, prev_state["a2"]))
                else:
                    # first iteration: no previous reference
                    history["dL_rel"].append(float("nan"))
                    history["dC_rel"].append(float("nan"))
                    history["dlogpsi_l2"].append(float("nan"))
                    history["dlogs_l2"].append(float("nan"))
                    history["dlogkappa_l2"].append(float("nan"))
                    history["du_rel"].append(float("nan"))
                    history["da2_rel"].append(float("nan"))

                # Identifiability-aware change: Var(y) surface
                # Var(y_{i,t,j}) = kappa_i*sum_k L_{j,k}^2*a2_{i,t,k} + s_i^2 * psi_j
                var_y = (
                    kappa_snap[:, None, None] * torch.einsum("itr,jr->itj", a2, L.pow(2))
                    + (s_snap.pow(2))[:, None, None] * psi_snap[None, None, :]
                )
                if prev_var_y is not None:
                    diff = var_y - prev_var_y
                    rmse = torch.sqrt(torch.mean(diff * diff))
                    rel_l2 = torch.linalg.norm(diff) / (torch.linalg.norm(prev_var_y) + eps)
                    max_rel = torch.max(diff.abs() / (prev_var_y.abs() + eps))
                    history["dvar_y_rmse"].append(float(rmse.detach().cpu().item()))
                    history["dvar_y_rel_l2"].append(float(rel_l2.detach().cpu().item()))
                    history["dvar_y_max_rel"].append(float(max_rel.detach().cpu().item()))
                else:
                    history["dvar_y_rmse"].append(float("nan"))
                    history["dvar_y_rel_l2"].append(float("nan"))
                    history["dvar_y_max_rel"].append(float("nan"))
                prev_var_y = var_y.detach()

                # Identifiability-aware change: H_it on fixed sample pairs
                if len(sample_pairs) > 0 and "L" in prev_state:
                    dH_abs = []
                    dH_rel = []
                    for (i_s, t_s) in sample_pairs:
                        obs_idx = torch.nonzero(M_mask[i_s, t_s], as_tuple=False).squeeze(1)
                        if obs_idx.numel() == 0:
                            continue

                        def H_it(
                            Lm: torch.Tensor,
                            a2m: torch.Tensor,
                            psim: torch.Tensor,
                            sm: torch.Tensor,
                            km: torch.Tensor,
                        ) -> torch.Tensor:
                            L_obs = Lm.index_select(0, obs_idx)  # (J_obs, r)
                            A = torch.diag(km[i_s] * a2m[i_s, t_s])        # (r,r)
                            psi_obs = psim.index_select(0, obs_idx)  # (J_obs,)
                            s2 = sm[i_s].pow(2)
                            Psi_obs = torch.diag(psi_obs * s2)  # (J_obs,J_obs)
                            return L_obs @ A @ L_obs.T + Psi_obs

                        H_cur = H_it(L, a2, psi_snap, s_snap, kappa_snap)
                        H_prev = H_it(prev_state["L"], prev_state["a2"], prev_state["psi"], prev_state["s"], prev_state["kappa"])
                        diffH = H_cur - H_prev
                        frob = torch.linalg.norm(diffH)
                        rel = frob / (torch.linalg.norm(H_prev) + eps)
                        dH_abs.append(float(frob.detach().cpu().item()))
                        dH_rel.append(float(rel.detach().cpu().item()))

                    if len(dH_abs) > 0:
                        history["dH_frob_mean"].append(float(np.mean(dH_abs)))
                        history["dH_rel_frob_mean"].append(float(np.mean(dH_rel)))
                        history["dH_rel_frob_max"].append(float(np.max(dH_rel)))
                    else:
                        history["dH_frob_mean"].append(float("nan"))
                        history["dH_rel_frob_mean"].append(float("nan"))
                        history["dH_rel_frob_max"].append(float("nan"))
                else:
                    history["dH_frob_mean"].append(float("nan"))
                    history["dH_rel_frob_mean"].append(float("nan"))
                    history["dH_rel_frob_max"].append(float("nan"))

                # Update prev snapshots
                prev_state["L"] = L.detach().clone()
                prev_state["C"] = C.detach().clone()
                prev_state["logpsi"] = logpsi.detach().clone()
                prev_state["logs"] = logs.detach().clone()
                prev_state["logkappa"] = logkappa.detach().clone()
                prev_state["psi"] = psi_snap.detach().clone()
                prev_state["s"] = s_snap.detach().clone()
                prev_state["kappa"] = kappa_snap.detach().clone()
                prev_state["u_tilde"] = u_tilde.detach().clone()
                prev_state["a2"] = a2.detach().clone()

        t_outer1 = time.perf_counter()
        history["t_estep_s"].append(float(t_estep1 - t_estep0))
        history["t_warp_s"].append(float(t_warp1 - t_warp0))
        history["t_mstep_s"].append(float(t_mstep1 - t_mstep0))
        history["t_post_s"].append(float(t_post1 - t_post0))
        history["t_outer_total_s"].append(float(t_outer1 - t_outer0))
        history["t_total_s"].append(float(t_outer1 - t_start))

        if prev_ll is not None and torch.abs(ll_val - prev_ll) < 1e-3:
            break
        prev_ll = ll_val

    psi = torch.nn.functional.softplus(psi_raw) + 1e-4
    s = torch.nn.functional.softplus(s_raw) + 1e-4
    kappa = torch.nn.functional.softplus(kappa_raw) + 1e-4
    _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)
    if use_woodbury:
        F_tilde, _ = factor_E_step_fast(Y, M_mask, L, psi, s, kappa, a2, pattern_groups=pattern_groups)
    else:
        F_tilde, _ = factor_E_step(Y, M_mask, L, psi, s, kappa, a2)

    if return_history:
        return L, a2, psi, s, kappa, C, u_tilde, F_tilde, history
    return L, a2, psi, s, kappa, C, u_tilde, F_tilde

