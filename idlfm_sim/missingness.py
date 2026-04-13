from __future__ import annotations

import numpy as np
from typing import Literal


def gaussian_time_kernel_matrix(T: int, h: float) -> np.ndarray:
    """
    Build a Gaussian kernel over discrete time points {0,...,T-1}:
      K_h(t, t') = exp(-(t - t')^2 / (2 h^2))
    Returns K with shape (T, T) where K[u, t] = K_h(u, t).
    """
    if h <= 0:
        raise ValueError("Bandwidth h must be positive.")
    t = np.arange(T, dtype=float)
    diff = t[None, :] - t[:, None]  # (T, T) with [u, t] = u - t
    K = np.exp(-(diff ** 2) / (2.0 * (h ** 2)))
    return K


def estimate_missingness_kernel(
    observed_mask: np.ndarray,
    h: float,
    mode: Literal["time_local"] = "time_local",
) -> np.ndarray:
    """
    Kernel-smoothed estimator for missingness probabilities over X = (i,j,t).

    Let Z_{i,j,t} = 1{observed}, observed_mask[i,j,t] in {False, True}.
    For each (i,j,t), estimate:
        p_hat(i,j,t) = sum_{i',j',t'} K_h(X, X') Z_{i',j',t'} / sum_{i',j',t'} K_h(X, X')

    mode == "time_local":
        Uses a separable kernel that only compares within the same subject-stream (i,j),
        i.e., K_h((i,j,t),(i',j',t')) = 1{i=i', j=j'} * exp(-(t - t')^2 / (2 h^2)).
        This yields an O(I*J*T^2) computation but vectorizes efficiently with einsum.

    Inputs:
      observed_mask: boolean array (I, J, T) with True where observed (O*A==1).
      h: bandwidth (>0).
      mode: currently only "time_local" supported.

    Returns:
      p_hat: float array (I, J, T) with values in [0,1].
    """
    if mode != "time_local":
        raise NotImplementedError("Only mode='time_local' is implemented in this version.")
    if observed_mask.ndim != 3:
        raise ValueError("observed_mask must have shape (I, J, T).")
    I, J, T = observed_mask.shape
    K = gaussian_time_kernel_matrix(T, h)  # (T, T) with K[u, t]
    denom = K.sum(axis=1)  # (T,)
    # numerator[i,j,u] = sum_t observed_mask[i,j,t] * K[u, t]
    numerator = np.einsum("ijt,ut->iju", observed_mask.astype(float), K)
    p_hat = numerator / (denom[None, None, :] + 1e-12)
    # Clip to [0,1] for numerical stability
    p_hat = np.clip(p_hat, 0.0, 1.0)
    return p_hat


