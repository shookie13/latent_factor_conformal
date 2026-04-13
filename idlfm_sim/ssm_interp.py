from __future__ import annotations

import numpy as np
from typing import Optional


def _kalman_smoother_local_trend(
    y: np.ndarray,
    mask: np.ndarray,
    sigma_level: float = 0.1,
    sigma_trend: float = 0.01,
    sigma_obs: Optional[float] = None,
) -> np.ndarray:
    """
    Local linear trend state-space model per series with missing observations.

    State: x_t = [level_t, trend_t]^T
      x_t = F x_{t-1} + w_t,   F = [[1, 1], [0, 1]],   w_t ~ N(0, Q), Q = diag(q_level, q_trend)
      y_t = H x_t + v_t,       H = [1, 0],             v_t ~ N(0, R)

    Inputs:
      y: (T,) series with arbitrary values (ignored when mask[t] is False)
      mask: (T,) boolean, True where y_t is observed
      sigma_level, sigma_trend: process std devs
      sigma_obs: measurement std dev; if None, estimated from observed residuals of a simple smoother

    Returns:
      y_hat: (T,) smoothed estimate of y_t (posterior mean of level_t)
    """
    T = y.shape[0]
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)  # 1x2
    Q = np.diag([sigma_level ** 2, sigma_trend ** 2])

    # If no sigma_obs supplied, estimate as std of first-difference on observed
    if sigma_obs is None:
        obs = y[mask]
        if obs.size >= 3:
            diffs = np.diff(obs)
            est = np.std(diffs) / np.sqrt(2.0)
            sigma_obs = float(max(est, 1e-3))
        else:
            sigma_obs = 0.1
    R = np.array([[sigma_obs ** 2]], dtype=float)

    # Initialization: diffuse prior
    # Use first observed value for level; trend = 0; large covariance
    if mask.any():
        first_idx = int(np.flatnonzero(mask)[0])
        m0 = np.array([y[first_idx], 0.0], dtype=float)
    else:
        m0 = np.array([0.0, 0.0], dtype=float)
    P0 = np.eye(2) * 1e4

    # Allocate arrays
    m = np.zeros((T, 2), dtype=float)
    P = np.zeros((T, 2, 2), dtype=float)
    a = np.zeros((T, 2), dtype=float)
    C = np.zeros((T, 2, 2), dtype=float)

    # Forward Kalman filter
    mt, Pt = m0, P0
    for t in range(T):
        # Predict
        at = F @ mt
        Ct = F @ Pt @ F.T + Q
        # Update if observed
        if mask[t]:
            yt = np.array([[y[t]]])
            S = H @ Ct @ H.T + R  # 1x1
            K = (Ct @ H.T) / S  # 2x1
            mt = at + (K @ (yt - H @ at)).ravel()
            Pt = Ct - K @ H @ Ct
        else:
            mt, Pt = at, Ct
        a[t] = at
        C[t] = Ct
        m[t] = mt
        P[t] = Pt

    # Backward RTS smoother
    ms = np.zeros_like(m)
    Ps = np.zeros_like(P)
    ms[-1] = m[-1]
    Ps[-1] = P[-1]
    for t in range(T - 2, -1, -1):
        Ct = C[t]
        Pt = P[t]
        J = Pt @ F.T @ np.linalg.pinv(Ct + 1e-12 * np.eye(2))
        ms[t] = m[t] + (J @ (ms[t + 1] - a[t])).ravel()
        Ps[t] = P[t] + J @ (Ps[t + 1] - C[t]) @ J.T

    # Predicted/Smoothed y is level component
    y_hat = ms[:, 0]
    return y_hat


def interpolate_local_trend(
    Y: np.ndarray,
    mask: np.ndarray,
    sigma_level: float = 0.1,
    sigma_trend: float = 0.01,
    sigma_obs: Optional[float] = None,
) -> np.ndarray:
    """
    Interpolate all series with a local linear trend model.

    Inputs:
      Y: (I, J, T) values (arbitrary at missing positions)
      mask: (I, J, T) boolean True where observed
      sigma_*: model hyperparameters shared for all series

    Returns:
      Y_hat: (I, J, T) smoothed estimates over full grid
    """
    I, J, T = Y.shape
    Y_hat = np.zeros_like(Y, dtype=float)
    for i in range(I):
        for j in range(J):
            y = Y[i, j]
            m = mask[i, j]
            if m.sum() == 0:
                Y_hat[i, j] = y  # nothing to do
            else:
                Y_hat[i, j] = _kalman_smoother_local_trend(y, m, sigma_level, sigma_trend, sigma_obs)
    return Y_hat


