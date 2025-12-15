"""
Simulation utilities for the time-warped B-spline factor variance model.

This adapts the earlier kernel-smoothed driver to:
  - build subject–factor specific warped times from a volatility proxy
  - evaluate log-variance via shared B-spline basis and control points
  - sample factors/innovations accordingly
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np


def make_open_uniform_knots_np(num_ctrl: int, degree: int) -> np.ndarray:
    if num_ctrl <= degree:
        raise ValueError("num_ctrl must exceed degree")
    interior = num_ctrl - degree - 1
    knots = np.zeros(num_ctrl + degree + 1, dtype=float)
    if interior > 0:
        knots[degree:-degree] = np.linspace(0.0, 1.0, interior + 2)
    knots[-(degree + 1) :] = 1.0
    return knots


def bspline_basis_np(u: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    p = degree
    n_basis = knots.size - p - 1
    if n_basis <= 0:
        raise ValueError("invalid knot configuration")

    # degree 0
    B = []
    for i in range(n_basis):
        cond = (u >= knots[i]) & (u < knots[i + 1])
        if i == n_basis - 1:
            cond = cond | (u == knots[i + 1])
        B.append(cond.astype(float))
    B = np.stack(B, axis=1)  # (T, M)

    for k in range(1, p + 1):
        B_next = np.zeros_like(B)
        for i in range(n_basis):
            left_den = max(knots[i + k] - knots[i], 1e-12)
            right_den = max(knots[i + k + 1] - knots[i + 1], 1e-12)

            left_num = (u - knots[i]) * B[:, i]
            right_num = (knots[i + k + 1] - u) * (B[:, i + 1] if i + 1 < n_basis else 0.0)
            B_next[:, i] = left_num / left_den + right_num / right_den
        B = B_next
    return B  # (T, M)


def build_time_warp_from_proxy_np(vol_proxy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    vol_proxy: (I, T, r) nonnegative
    returns u_tilde in (0,1], same shape
    """
    w = np.asarray(vol_proxy, dtype=float) + eps
    S = np.cumsum(w, axis=1)
    total = S[:, -1:, :]
    return S / np.clip(total, 1e-12, None)


def missing_prob_from_factor_variance(
    var_f: np.ndarray,
    X_true: np.ndarray,
    *,
    normalize: str = "std",  # "std" | "minmax" | "none"
    beta0: float = -1.5,
    beta1: float = 3.0,
    clip: Tuple[float, float] = (0.15, 0.85),
) -> np.ndarray:
    """
    Compute per-channel missing probabilities p_miss[j] as a monotone function of the
    factor-induced channel variance v_ch[j] = sum_k X_{j,k}^2 * var_f[k].
    """
    var_f = np.asarray(var_f, dtype=float)  # (r,)
    X_true = np.asarray(X_true, dtype=float)  # (J, r)
    J, r = X_true.shape
    assert var_f.shape == (r,)
    v_ch = (X_true ** 2) @ var_f  # (J,)
    z = v_ch.copy()
    if normalize == "std":
        mu = float(np.mean(z))
        sd = float(np.std(z)) + 1e-12
        z = (z - mu) / sd
    elif normalize == "minmax":
        lo = float(np.min(z))
        hi = float(np.max(z))
        z = (z - lo) / (hi - lo + 1e-12)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")
    p = 1.0 / (1.0 + np.exp(-(beta0 + beta1 * z)))
    p = np.clip(p, clip[0], clip[1])
    return p


def sample_mask_from_varf(
    I: int,
    T: int,
    X_true: np.ndarray,
    a2_true: np.ndarray,
    kappa_true: np.ndarray,
    *,
    seed: Optional[int] = None,
    normalize: str = "std",
    beta0: float = -1.5,
    beta1: float = 3.0,
    clip: Tuple[float, float] = (0.0, 0.95),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as previous script: sample mask with probabilities tied to factor variance.
    """
    rng = np.random.default_rng(seed)
    X_true = np.asarray(X_true, dtype=float)
    a2_true = np.asarray(a2_true, dtype=float)  # (I, T, r)
    kappa_true = np.asarray(kappa_true, dtype=float)  # (I,)
    I2, T2, r = a2_true.shape
    assert I2 == I and T2 == T
    J, r2 = X_true.shape
    assert r2 == r
    assert kappa_true.shape == (I,)

    M = np.zeros((I, T, J), dtype=bool)
    P = np.zeros((I, T, J), dtype=float)
    for i in range(I):
        for t_idx in range(T):
            var_f = kappa_true[i] * a2_true[i, t_idx, :]
            p_miss = missing_prob_from_factor_variance(
                var_f, X_true, normalize=normalize, beta0=beta0, beta1=beta1, clip=clip
            )
            P[i, t_idx, :] = p_miss
            draw = rng.uniform(size=J)
            M[i, t_idx, :] = draw > p_miss  # True = observed
    return M, P


def simulate_factor_data_bspline(
    I: int,
    T: int,
    J: int,
    r: int,
    *,
    M_ctrl: int,
    degree: int = 3,
    seed: int = 0,
    X_true: Optional[np.ndarray] = None,  # (J, r)
    Psi_true: Optional[np.ndarray] = None,  # (J,)
    s_true: Optional[np.ndarray] = None,  # (I,)
    kappa_true: Optional[np.ndarray] = None,  # (I,)
    C_true: Optional[np.ndarray] = None,  # (I, r, M_ctrl)
    vol_proxy: Optional[np.ndarray] = None,  # (I, T, r) for warp; if None, auto-generate
    eps_warp: float = 1e-3,
    miss_prob_per_channel: Optional[np.ndarray] = None,  # (J,)
    use_variance_tied_missing: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic innovations Y (I,T,J) and mask M (I,T,J) using the B-spline
    time-warped variance model:
      - subject-factor warp from a positive proxy via cumulative weights
      - log variance from shared B-spline basis and subject-specific control points
      - factor draws F ~ N(0, kappa_i * diag(a2_true[i,t]))
      - innovations e_it = X_true F_it + noise
    """
    rng = np.random.default_rng(seed)
    if X_true is None:
        X = rng.normal(size=(J, r))
        X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
        X_true = X+np.sign(X)*0.3
    else:
        X_true = np.asarray(X_true, dtype=float)
        assert X_true.shape == (J, r)

    if Psi_true is None:
        Psi_true = rng.uniform(0.1, 0.4, size=J)
    else:
        Psi_true = np.clip(np.asarray(Psi_true, dtype=float), 1e-8, None)
        assert Psi_true.shape == (J,)

    if s_true is None:
        s_true = np.exp(rng.normal(0.0, 0.1, size=I))
    else:
        s_true = np.clip(np.asarray(s_true, dtype=float), 1e-8, None)
        assert s_true.shape == (I,)

    if kappa_true is None:
        kappa_true = np.exp(rng.normal(0.0, 0.1, size=I))
    else:
        kappa_true = np.clip(np.asarray(kappa_true, dtype=float), 1e-8, None)
        assert kappa_true.shape == (I,)

    if miss_prob_per_channel is None:
        miss_prob_per_channel = np.zeros(J, dtype=float)
    else:
        miss_prob_per_channel = np.clip(np.asarray(miss_prob_per_channel, dtype=float), 0.0, 1.0)
        assert miss_prob_per_channel.shape == (J,)

    # Control points
    if C_true is None:
        C_true = rng.normal(scale=0.4, size=(I, r, M_ctrl))
    else:
        C_true = np.asarray(C_true, dtype=float)
        assert C_true.shape == (I, r, M_ctrl)

    # Volatility proxy for warp
    if vol_proxy is None:
        t = np.arange(T, dtype=float)
        base = 0.5 + 0.5 * np.sin(2 * np.pi * (t + 1.0) / float(T))
        vol_proxy = np.zeros((I, T, r), dtype=float)
        for i in range(I):
            for k in range(r):
                bump_loc = rng.integers(low=int(0.2 * T), high=max(int(0.6 * T), 1))
                bump_end = min(T - 1, bump_loc + max(1, int(0.1 * T)))
                v = base.copy()
                v[bump_loc : bump_end + 1] += 0.8
                v += rng.normal(scale=0.05, size=T)
                vol_proxy[i, :, k] = np.clip(v, 1e-3, None)
    else:
        vol_proxy = np.asarray(vol_proxy, dtype=float)
        assert vol_proxy.shape == (I, T, r)

    knots = make_open_uniform_knots_np(M_ctrl, degree)
    u_tilde = build_time_warp_from_proxy_np(vol_proxy, eps=eps_warp)  # (I,T,r)

    # Build log-variance via B-spline at warped times
    log_a2 = np.zeros((I, T, r), dtype=float)
    a2_true = np.zeros_like(log_a2)
    for i in range(I):
        for k in range(r):
            B = bspline_basis_np(u_tilde[i, :, k], knots, degree)  # (T, M_ctrl)
            s_t = B @ C_true[i, k, :].reshape(-1, 1)
            s_t = s_t.ravel()
            log_a2[i, :, k] = s_t
            a2_true[i, :, k] = np.exp(s_t)

    # Normalize per subject to mean 1 to avoid scale drift
    means = np.clip(a2_true.mean(axis=1, keepdims=True), 1e-6, None)
    a2_true = a2_true / means
    log_a2 = np.log(a2_true)

    Y = np.zeros((I, T, J), dtype=float)
    M = np.zeros((I, T, J), dtype=bool)
    rng_noise = np.random.default_rng(seed + 123)

    for i in range(I):
        for t_idx in range(T):
            # var_f = kappa_true[i] * a2_true[i, t_idx, :]
            #   where var_f[k] = \kappa_i * a^2_{i, t, k}
            var_f = kappa_true[i] * a2_true[i, t_idx, :]  # (r,) elementwise

            # F_it ~ N(0, diag(var_f)), so F_it[k] ~ N(0, var_f[k]) for k = 0,...,r-1
            F_it = rng.normal(loc=0.0, scale=np.sqrt(var_f), size=r)  # (r,)

            # u_it ~ N(0, s_true[i]^2 * diag(Psi_true)), so u_it[j] ~ N(0, s_true[i]^2 * Psi_true[j])
            u_it = rng_noise.normal(loc=0.0, scale=s_true[i] * np.sqrt(Psi_true), size=J)  # (J,)

            # e_it = X_true @ F_it + u_it  (dim: J = X_true (J,r) @ F_it (r,) + u_it (J,))
            #   Y[i, t, :] = e_it
            e_it = X_true @ F_it + u_it
            Y[i, t_idx, :] = e_it

            if use_variance_tied_missing:
                p_miss = missing_prob_from_factor_variance(var_f, X_true)
                draw = rng.uniform(size=J)
                M[i, t_idx, :] = draw > p_miss
            else:
                draw = rng.uniform(size=J)
                M[i, t_idx, :] = draw > miss_prob_per_channel

    info: Dict[str, Any] = dict(
        X_true=X_true,
        Psi_true=Psi_true,
        s_true=s_true,
        kappa_true=kappa_true,
        C_true=C_true,
        a2_true=a2_true,
        log_a2_true=log_a2,
        u_tilde_true=u_tilde,
        vol_proxy=vol_proxy,
        knots=knots,
        degree=degree,
        miss_prob_per_channel=miss_prob_per_channel,
        use_variance_tied_missing=use_variance_tied_missing,
        seed=seed,
    )
    return Y, M, info

