"""
Woodbury + determinant-lemma engine for the factor model with diagonal factor covariance.

We observe (possibly masked) channel residuals y_{i,t} in R^{J_obs}:

    y = L_obs f + u
    f ~ N(0, A),        A = diag(a2)  (r x r)
    u ~ N(0, s_i^2 D),  D = diag(psi_obs) (J_obs x J_obs)

Then the observed covariance is:

    H = L_obs A L_obs^T + s^2 D.                      (J_obs x J_obs)

Direct Cholesky/solve on H is O(J_obs^3). When r << J_obs we can work in r-space.

Whiten by D^{-1/2}:

    W = D^{-1/2} L_obs                                 (J_obs x r)
    z = D^{-1/2} y                                     (J_obs)

Let:

    G = W^T W                                          (r x r)
    v = W^T z                                          (r)
    z2 = ||z||^2                                       (scalar)

Define the small SPD matrix:

    S = A^{-1} + s^{-2} G                              (r x r)

Then (Woodbury + determinant lemma) give:

    log|H| = log|s^2 D| + log|A| + log|S|
           = J_obs*log(s^2) + log|D| + sum_k log(a2_k) + log|S|.

and

    y^T H^{-1} y = s^{-2} z2 - s^{-4} v^T S^{-1} v.

The posterior mean of factors is:

    E[f | y] = A L_obs^T H^{-1} y
             = A * xHy,

where

    xHy := L_obs^T H^{-1} y
         = s^{-2} (v - s^{-2} G S^{-1} v).

We implement this batched over all rows sharing the same observed-channel pattern,
so that G and log|D| are computed once per pattern (pattern caching).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


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

def factor_log_likelihood_woodbury_minibatch(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    pattern_groups: List[PatternGroup],
    flat_batch: torch.Tensor,         # (B,) flattened indices into i*T+t
    total_n: int,                     # total number of valid (i,t) rows (with >=1 obs)
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Minibatch log-likelihood using the same Woodbury+pattern code, but only on flat_batch.
    Returns an unbiased estimate of the FULL sum log-likelihood by scaling total_n / batch_n.
    """
    I, T, J = Y.shape
    r = L.shape[1]
    device = Y.device
    dtype = Y.dtype

    # flatten once
    Y_flat = Y.reshape(I * T, J)
    a2_flat = a2.reshape(I * T, r)

    # mark which rows are in the batch
    batch_flag = torch.zeros((I * T,), dtype=torch.bool, device=device)
    batch_flag[flat_batch] = True

    ll = torch.zeros((), dtype=dtype, device=device)
    eye_r = torch.eye(r, dtype=dtype, device=device)

    for g in pattern_groups:
        # select only rows of this pattern that are in the minibatch
        sel = batch_flag.index_select(0, g.flat)          # (n_g,)
        if not torch.any(sel):
            continue

        flat = g.flat[sel]
        i_idx = g.i_idx[sel]
        idx = g.idx
        Jo = g.J_obs
        n = flat.numel()

        # same math as factor_E_step_woodbury (but NO F_tilde writing)
        Y_blk = Y_flat.index_select(0, flat).index_select(1, idx)  # (n, Jo)

        psi_idx = psi.index_select(0, idx)                       # (Jo,)
        d_sqrt_inv = torch.rsqrt(psi_idx + eps)                  # D^{-1/2}
        logdet_D = torch.log(psi_idx + eps).sum()                # log|D|

        L_idx = L.index_select(0, idx)                           # (Jo, r)
        W = d_sqrt_inv.unsqueeze(1) * L_idx                      # (Jo, r)
        G = W.T @ W                                              # (r, r)

        Z = Y_blk * d_sqrt_inv.unsqueeze(0)                      # (n, Jo)
        V = Z @ W                                                # (n, r)
        z2 = (Z * Z).sum(dim=1)                                  # (n,)

        A_rows = a2_flat.index_select(0, flat)                   # (n, r)
        s2 = (s.index_select(0, i_idx) ** 2)                     # (n,)

        invA = 1.0 / (A_rows + eps)                              # (n, r)
        invs2 = 1.0 / (s2 + eps)                                 # (n,)

        S = torch.diag_embed(invA) + invs2.view(n, 1, 1) * G      # (n, r, r)
        if jitter > 0:
            S = S + jitter * eye_r

        Lch = torch.linalg.cholesky(S)                            # (n, r, r)
        logdet_S = 2.0 * torch.log(torch.diagonal(Lch, dim1=-2, dim2=-1)).sum(dim=1)

        w = torch.cholesky_solve(V.unsqueeze(-1), Lch).squeeze(-1) # (n, r)
        v_dot_w = (V * w).sum(dim=1)                               # (n,)

        yHy = invs2 * z2 - (invs2 * invs2) * v_dot_w               # (n,)

        logdet_A = torch.log(A_rows + eps).sum(dim=1)              # (n,)
        logdet_H = logdet_A + (Jo * torch.log(s2 + eps)) + logdet_D + logdet_S

        ll = ll + (-0.5) * (logdet_H + yHy).sum()

    # scale minibatch sum to estimate full-data sum (unbiased gradient)
    batch_n = flat_batch.numel()
    ll = ll * (float(total_n) / float(batch_n))
    return ll

