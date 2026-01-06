import torch

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


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


def build_time_warp_from_proxy(vol_proxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Build monotone time warp from a volatility proxy.

    Args:
        vol_proxy: tensor (I, T, r) with nonnegative entries.
        eps: small positive constant to keep weights > 0.

    Returns:
        u_tilde: tensor (I, T, r) in (0, 1], strictly increasing over t.
    """
    w = vol_proxy + eps
    S = w.cumsum(dim=1)
    total = S[:, -1:, :]
    u_tilde = S / total.clamp_min(1e-12)
    return u_tilde


def update_time_warp(F_tilde: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Update warp using squared posterior factors as the proxy.

    Args:
        F_tilde: tensor (I, T, r).
        eps: positive floor for weights.

    Returns:
        u_tilde: tensor (I, T, r).
    """
    vol = F_tilde.pow(2)
    return build_time_warp_from_proxy(vol, eps=eps)

@dataclass
class PatternGroup:
    """
    A group of (i,t) rows sharing the same observed channel index set.

    idx: (J_obs,) observed channel indices for this pattern.
    flat: (n,) flattened row indices flat = i*T + t
    i_idx: (n,) subject indices i for each row (for s[i])
    J_obs: number of observed channels in this pattern
    """

    idx: torch.Tensor
    flat: torch.Tensor
    i_idx: torch.Tensor
    J_obs: int


def build_pattern_groups(M_mask: torch.Tensor) -> List[PatternGroup]:
    """
    Group (i,t) by observed-channel pattern.

    Args:
        M_mask: bool tensor (I, T, J) where True means observed.

    Returns:
        List of PatternGroup objects on the same device as M_mask.

    Notes:
        - This grouping depends only on the mask, so it can be computed once and reused.
        - It uses Python dicts/tuples, so it is intended for moderate I*T; for very large
          datasets consider a vectorized hash-based approach.
    """
    I, T, _ = M_mask.shape
    device = M_mask.device

    groups: Dict[Tuple[int, ...], Tuple[List[int], List[int]]] = {}
    # value is (flat_list, i_list)
    for i in range(I):
        for t in range(T):
            idx = torch.nonzero(M_mask[i, t], as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            key = tuple(idx.detach().cpu().tolist())
            flat = i * T + t
            if key not in groups:
                groups[key] = ([], [])
            groups[key][0].append(flat)
            groups[key][1].append(i)

    out: List[PatternGroup] = []
    for key, (flat_list, i_list) in groups.items():
        idx_t = torch.tensor(key, dtype=torch.long, device=device)
        flat_t = torch.tensor(flat_list, dtype=torch.long, device=device)
        i_t = torch.tensor(i_list, dtype=torch.long, device=device)
        out.append(PatternGroup(idx=idx_t, flat=flat_t, i_idx=i_t, J_obs=int(idx_t.numel())))
    return out


def factor_E_step_woodbury(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    pattern_groups: List[PatternGroup] | None = None,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute posterior factor means and log-likelihood using Woodbury + pattern caching.

    Args:
        Y: (I, T, J)
        M_mask: (I, T, J) bool
        L: (J, r)
        psi: (J,) positive
        s: (I,) positive
        a2: (I, T, r) positive factor variances
        pattern_groups: optional precomputed pattern groups from build_pattern_groups(M_mask)
        eps: numerical epsilon
        jitter: added to diagonal of S for Cholesky stability

    Returns:
        F_tilde: (I, T, r)
        ll: scalar (sum over i,t of -0.5*(logdet+quad), dropping constants)
    """
    I, T, J = Y.shape
    r = L.shape[1]
    device = Y.device
    dtype = Y.dtype

    if pattern_groups is None:
        pattern_groups = build_pattern_groups(M_mask)

    Y_flat = Y.reshape(I * T, J)
    a2_flat = a2.reshape(I * T, r)

    F_tilde_flat = torch.zeros((I * T, r), dtype=dtype, device=device)
    ll = torch.zeros((), dtype=dtype, device=device)

    eye_r = torch.eye(r, dtype=dtype, device=device)

    for g in pattern_groups:
        idx = g.idx
        flat = g.flat
        i_idx = g.i_idx
        Jo = g.J_obs
        n = flat.numel()
        if n == 0:
            continue

        # Gather block data for this pattern: Y_blk (n, Jo)
        Y_blk = Y_flat.index_select(0, flat).index_select(1, idx)

        psi_idx = psi.index_select(0, idx)  # (Jo,)
        d_sqrt_inv = torch.rsqrt(psi_idx + eps)  # D^{-1/2}
        logdet_D = torch.log(psi_idx + eps).sum()  # log|D|

        L_idx = L.index_select(0, idx)  # (Jo, r)
        W = d_sqrt_inv.unsqueeze(1) * L_idx  # (Jo, r)
        G = W.T @ W  # (r, r)

        Z = Y_blk * d_sqrt_inv.unsqueeze(0)  # (n, Jo)
        V = Z @ W  # (n, r)  where V[row]=W^T z
        z2 = (Z * Z).sum(dim=1)  # (n,)

        A_rows = a2_flat.index_select(0, flat)  # (n, r)
        s2 = (s.index_select(0, i_idx) ** 2)  # (n,)

        invA = 1.0 / (A_rows + eps)  # (n, r)
        invs2 = 1.0 / (s2 + eps)  # (n,)

        # S = A^{-1} + s^{-2} G, batched (n, r, r)
        S = torch.diag_embed(invA) + invs2.view(n, 1, 1) * G
        if jitter > 0:
            S = S + jitter * eye_r

        Lch = torch.linalg.cholesky(S)  # (n, r, r)
        logdet_S = 2.0 * torch.log(torch.diagonal(Lch, dim1=-2, dim2=-1)).sum(dim=1)  # (n,)

        # w = S^{-1} v
        w = torch.cholesky_solve(V.unsqueeze(-1), Lch).squeeze(-1)  # (n, r)
        v_dot_w = (V * w).sum(dim=1)  # (n,)

        # y^T H^{-1} y = s^{-2} z2 - s^{-4} v^T w
        yHy = invs2 * z2 - (invs2 * invs2) * v_dot_w

        # log|H| = log|A| + J_obs log s^2 + log|D| + log|S|
        logdet_A = torch.log(A_rows + eps).sum(dim=1)  # (n,)
        logdet_H = logdet_A + (Jo * torch.log(s2 + eps)) + logdet_D + logdet_S

        ll = ll + (-0.5) * (logdet_H + yHy).sum()

        # posterior mean: xHy = s^{-2}(v - s^{-2} G w), F = A * xHy
        Gw = w @ G.T  # (n, r)
        xHy = invs2.unsqueeze(1) * (V - invs2.unsqueeze(1) * Gw)  # (n, r)
        F_rows = A_rows * xHy  # (n, r)

        F_tilde_flat.index_copy_(0, flat, F_rows)

    F_tilde = F_tilde_flat.view(I, T, r)
    return F_tilde, ll


def factor_log_likelihood_woodbury(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    pattern_groups: List[PatternGroup] | None = None,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Log-likelihood-only version (still computes the same terms but does not return F_tilde).
    """
    _, ll = factor_E_step_woodbury(
        Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups, eps=eps, jitter=jitter
    )
    return ll

def _mask_observed(M_mask: torch.Tensor, i: int, t: int):
    return torch.nonzero(M_mask[i, t], as_tuple=False).squeeze(1)


def factor_E_step(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
):
    """
    Compute posterior means of factors and log-likelihood contribution.

    Args:
        Y: (I, T, J) residuals.
        M_mask: bool mask (I, T, J) True=observed.
        L: loadings (J, r).
        psi: diag entries of Psi, shape (J,).
        s: subject scales (I,).
        a2: factor variances (I, T, r).

    Returns:
        F_tilde: (I, T, r) posterior means.
        ll: scalar log-likelihood.
    """
    I, T, J = Y.shape
    r = L.shape[1]
    device = Y.device
    F_tilde = torch.zeros((I, T, r), device=device)
    ll = torch.zeros((), device=device)

    Psi_diag = psi
    for i in range(I):
        Psi_i = torch.diag(Psi_diag * (s[i] ** 2))
        for t in range(T):
            obs_idx = _mask_observed(M_mask, i, t)
            if obs_idx.numel() == 0:
                continue
            y_obs = Y[i, t, obs_idx]
            L_obs = L[obs_idx]
            Psi_obs = Psi_i[obs_idx][:, obs_idx]
            A_it = torch.diag(a2[i, t])

            H_obs = L_obs @ A_it @ L_obs.T + Psi_obs
            H_inv_y = torch.linalg.solve(H_obs, y_obs)
            F_mean = A_it @ L_obs.T @ H_inv_y
            F_tilde[i, t] = F_mean

            sign, logdet = torch.linalg.slogdet(H_obs)
            quad = y_obs @ H_inv_y
            ll = ll - 0.5 * (logdet + quad)
    return F_tilde, ll


def factor_log_likelihood(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
):
    """
    Same as factor_E_step but returns only log-likelihood (differentiable).
    """
    _, ll = factor_E_step(Y, M_mask, L, psi, s, a2)
    return ll


def build_mask_pattern_groups(M_mask: torch.Tensor) -> list[PatternGroup]:
    """
    Convenience wrapper to build and cache mask pattern groups (depends only on M_mask).
    """
    return build_pattern_groups(M_mask)


def factor_E_step_fast(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    pattern_groups: list[PatternGroup] | None = None,
):
    """
    Fast E-step using Woodbury + pattern caching.
    """
    return factor_E_step_woodbury(Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups)


def factor_log_likelihood_fast(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    pattern_groups: list[PatternGroup] | None = None,
):
    """
    Fast log-likelihood using Woodbury + pattern caching.
    """
    return factor_log_likelihood_woodbury(Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups)

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
    if use_woodbury:
        F_tilde, _ = factor_E_step_fast(Y, M_mask, L, psi, s, a2, pattern_groups=pattern_groups)
    else:
        F_tilde, _ = factor_E_step(Y, M_mask, L, psi, s, a2)

    if return_history:
        return L, a2, psi, s, C, u_tilde, F_tilde, history
    return L, a2, psi, s, C, u_tilde, F_tilde