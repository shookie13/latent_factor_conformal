import numpy as np
from typing import Tuple
from .config import SimConfig
from .utils import rng, sigmoid


def _standardize_theta_per_subject(theta: np.ndarray, Rk: int) -> np.ndarray:
    I, T, R = theta.shape
    Rk_eff = min(Rk, R)
    Z = theta[:, :, :Rk_eff].copy()
    mean = Z.mean(axis=1, keepdims=True)
    std = Z.std(axis=1, keepdims=True) + 1e-8
    return (Z - mean) / std


def _calibrate_gamma0_for_target_p(scores: np.ndarray, p_target: float, tol: float = 1e-5, max_iter: int = 50) -> float:
    lo, hi = -6.0, 6.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = sigmoid(mid + scores).mean()
        if p < p_target:
            lo = mid
        else:
            hi = mid
        if abs(p - p_target) < tol:
            return mid
    return 0.5 * (lo + hi)


def _init_mar_params_if_needed(cfg: SimConfig, tilde_theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # This function initializes parameters gamma (G) and gamma0 for the MAR (Missing At Random) model if not provided in cfg.
    # These parameters govern the probability model for observing data (entries "missing" or not, MAR mechanism)
    r = rng(cfg.seed + 31)
    I, T, Rk_eff = tilde_theta.shape

    # If gamma (G) is not provided, initialize as random normal and scale row norm to 0.8.
    # Each row of G corresponds to a variable J and projects subject latent states to the "missingness" score space.
    if cfg.gamma is None:
        G = r.normal(size=(cfg.J, Rk_eff))
        norms = np.linalg.norm(G, axis=1, keepdims=True) + 1e-12  # Normalize for stability
        G = G * (0.8 / norms)
    else:
        # If provided, validate shape.
        G = np.asarray(cfg.gamma)
        if G.shape != (cfg.J, Rk_eff):
            raise ValueError("gamma must have shape (J, Rk)")

    # If gamma0 is not provided, calibrate for each variable, so that simulated missingness matches target marginal (~70% obs).
    # Use a sample of subject-time pairs to efficiently estimate appropriate gamma0 for each variable.
    if cfg.gamma0 is None:
        gamma0 = np.zeros(cfg.J, dtype=float)
        idx_i = r.integers(0, I, size=min(I, 50))   # Sample up to 50 subjects
        idx_t = r.integers(0, T, size=min(T, 80))   # Sample up to 80 timepoints
        Z = tilde_theta[np.ix_(idx_i, idx_t)].reshape(-1, Rk_eff)  # Subsample of latent state vectors
        for j in range(cfg.J):
            scores = Z @ G[j]
            # Calibrate gamma0[j] for this variable so that Pr(obs) ≈ 0.7 in marginal for generated states
            gamma0[j] = _calibrate_gamma0_for_target_p(scores, p_target=0.4)
    else:
        # If provided, validate shape.
        gamma0 = np.asarray(cfg.gamma0, dtype=float)
        if gamma0.shape != (cfg.J,):
            raise ValueError("gamma0 must have shape (J,)")

    # Return (gamma0, G) to be used in downstream MAR simulation.
    return gamma0, G


def sample_mar(cfg: SimConfig, theta: np.ndarray, O: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = rng(cfg.seed + 37)
    I, T, R = theta.shape
    J = O.shape[1]
    tilde = _standardize_theta_per_subject(theta, cfg.Rk)
    gamma0, G = _init_mar_params_if_needed(cfg, tilde)
    t = np.arange(T)[None, None, :]

    if cfg.mar_mode in ("linear", "seasonal"):
        score = np.einsum("itr,jr->ijt", tilde, G) + gamma0[None, :, None]
        if cfg.mar_mode == "seasonal" and abs(cfg.gamma_s) > 0:
            score = score + cfg.gamma_s * np.sin(2.0 * np.pi * t / cfg.Psea)
        p = sigmoid(score)
        return (r.random(size=(I, J, T)) < p).astype(np.int8), score

    if cfg.mar_mode == "duration":
        A = np.zeros((I, J, T), dtype=np.int8)
        D = np.zeros((I, J), dtype=int)
        for tt in range(T):
            base = cfg.delta0 + cfg.delta2 * D[:, :, None]
            if tt > 0:
                prevA = A[:, :, tt - 1:tt]
                base = base + cfg.delta1 * prevA
            gterm = np.einsum("jr,ir->ij", G, tilde[:, tt, :])
            logits = base[:, :, 0] + gterm
            p = sigmoid(logits)
            A[:, :, tt] = (r.random(size=(I, J)) < p).astype(np.int8)
            D = np.where(A[:, :, tt] == 1, D + 1, 0)
        return A, score

    p = 0.7
    return (r.random(size=(I, J, T)) < p).astype(np.int8), score


