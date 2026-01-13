"""
Small, notebook-friendly diagnostics for understanding early-time instability and undercoverage.

This module is intentionally light-weight: it provides summaries that help answer questions like:
  - Are early time points being "compressed" by the warp u_tilde?
  - Are control points (especially C[:,:,0]) drifting to extreme values?
  - Do scaled residuals |y-yhat|/sigma have heavier tails early in TEST than CAL?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .cp import unflatten


def summarize_em_diagnostics(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the `history` returned by `run_em_like(..., collect_diagnostics=True)` into
    simple NumPy arrays for plotting.
    """
    out: Dict[str, Any] = {}
    for k in ("ll_outer", "ll_post_m", "C0_mean", "C0_std", "C0_max_abs", "u_head_inc_min"):
        if k in history:
            out[k] = np.asarray(history[k], dtype=float)

    # Nested list fields: (n_outer, t_head) or (n_outer, t_head-1)
    for k in ("u_head_mean", "u_head_inc_mean", "a2_head_mean", "a2_head_max"):
        if k in history:
            out[k] = np.asarray(history[k], dtype=float)

    return out


def time_from_flat(idx: np.ndarray, *, I: int, J: int, T: int) -> np.ndarray:
    """
    Convert flattened indices k = i*(J*T) + j*T + t into t.
    """
    idx = np.asarray(idx, dtype=int).ravel()
    t = np.empty(idx.size, dtype=int)
    for n, k in enumerate(idx.tolist()):
        _, _, tt = unflatten(int(k), I, J, T)
        t[n] = tt
    return t


def scaled_residual(y_true: np.ndarray, y_hat: np.ndarray, sigma: np.ndarray, *, sigma_min: float = 1e-6) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sig = np.maximum(sigma, float(sigma_min))
    return np.abs(y_true - y_hat) / sig


def binned_tail_stats(
    t: np.ndarray,
    s: np.ndarray,
    *,
    q: Optional[float] = None,
    n_bins: int = 10,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Bin by time into equal-width bins and compute (count, mean(s), and optionally P(s>q)).
    """
    t = np.asarray(t, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    if t.size != s.size:
        raise ValueError("t and s must have the same length")
    if t.size == 0:
        return {"edges": np.array([]), "centers": np.array([]), "n": np.array([]), "mean_s": np.array([]), "tail": np.array([])}

    if t_min is None:
        t_min = float(np.nanmin(t))
    if t_max is None:
        t_max = float(np.nanmax(t))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("invalid t_min/t_max for binning")

    edges = np.linspace(t_min, t_max, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, int(n_bins) - 1)

    n = np.zeros(int(n_bins), dtype=int)
    mean_s = np.full(int(n_bins), np.nan, dtype=float)
    tail = np.full(int(n_bins), np.nan, dtype=float)
    for b in range(int(n_bins)):
        m = bin_id == b
        n[b] = int(np.sum(m))
        if n[b] == 0:
            continue
        mean_s[b] = float(np.mean(s[m]))
        if q is not None and np.isfinite(q):
            tail[b] = float(np.mean(s[m] > float(q)))
    return {"edges": edges, "centers": centers, "n": n, "mean_s": mean_s, "tail": tail}


@dataclass
class CalTestScaledResidualDiagnostics:
    """
    Summary comparing CAL vs TEST scaled residual behavior over time.
    """

    cal: Dict[str, Any]
    test: Dict[str, Any]


def cal_test_scaled_residual_diagnostics(
    *,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    Y_flat_true: np.ndarray,
    Y_flat_fit: np.ndarray,
    sigma_flat: np.ndarray,
    I: int,
    J: int,
    T: int,
    q: Optional[float] = None,
    n_time_bins: int = 10,
) -> CalTestScaledResidualDiagnostics:
    """
    Build time-binned summaries of scaled residuals for CAL and TEST.

    Inputs are "flat" arrays aligned with the flatten convention k=i*(J*T)+j*T+t.
    """
    cal_idx = np.asarray(cal_idx, dtype=int).ravel()
    test_idx = np.asarray(test_idx, dtype=int).ravel()
    Y_flat_true = np.asarray(Y_flat_true, dtype=float).ravel()
    Y_flat_fit = np.asarray(Y_flat_fit, dtype=float).ravel()
    sigma_flat = np.asarray(sigma_flat, dtype=float).ravel()

    s_cal = scaled_residual(Y_flat_true[cal_idx], Y_flat_fit[cal_idx], sigma_flat[cal_idx])
    s_test = scaled_residual(Y_flat_true[test_idx], Y_flat_fit[test_idx], sigma_flat[test_idx])
    t_cal = time_from_flat(cal_idx, I=I, J=J, T=T)
    t_test = time_from_flat(test_idx, I=I, J=J, T=T)

    # Use the global time range for consistent bins across CAL and TEST.
    t_min = 0.0
    t_max = float(max(T - 1, 1))
    cal = binned_tail_stats(t_cal, s_cal, q=q, n_bins=n_time_bins, t_min=t_min, t_max=t_max)
    test = binned_tail_stats(t_test, s_test, q=q, n_bins=n_time_bins, t_min=t_min, t_max=t_max)
    return CalTestScaledResidualDiagnostics(cal=cal, test=test)


def quickplot_em_diagnostics(diag: Dict[str, Any], *, show: bool = True):
    """
    Minimal Matplotlib visualization of EM diagnostics.

    Expects output from `summarize_em_diagnostics(history)`.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
    x = np.arange(diag.get("C0_mean", np.array([])).size)

    if "C0_max_abs" in diag:
        axes[0].plot(x, diag["C0_max_abs"], marker="o", label="max |C[:,:,0]|")
        axes[0].set_ylabel("max |C0|")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

    if "u_head_inc_min" in diag:
        axes[1].plot(x, diag["u_head_inc_min"], marker="o", label="min Δu over (i,t,k) in head")
        axes[1].set_ylabel("min Δu")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")

    if "a2_head_max" in diag:
        a2_head_max = np.asarray(diag["a2_head_max"], dtype=float)
        # plot first time point in head, and max over head
        axes[2].plot(x, a2_head_max[:, 0], marker="o", label="max a2 at earliest t (head)")
        axes[2].plot(x, a2_head_max.max(axis=1), marker="o", label="max a2 over head")
        axes[2].set_ylabel("a2 max")
        axes[2].set_xlabel("outer iteration")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


