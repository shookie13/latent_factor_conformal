"""
Diagnosis plots for conformal prediction performance over time.

Creates:
  1) coverage vs time (binned)
  2) histogram of scaled score s for early vs late, with q marked
  3) mean interval width vs time (binned)

Intended usage from a notebook:

    from diagnosis import make_diagnosis_plots

    figs = make_diagnosis_plots(
        t=t_test,                      # (n_test,)
        y_true=Y_test_true,            # (n_test,)
        y_hat=Y_test_fit,              # (n_test,) or None (defaults to 0)
        sigma=sigma_test,              # (n_test,)
        lower=lower_tmfv,              # (n_test,)
        upper=upper_tmfv,              # (n_test,)
        q=q_tmfv,                      # scalar
        alpha=0.1,
        early_t_max=30,
        n_time_bins=10,
        title_prefix="TWFV scaled CP",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _as_1d(x, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    return x


def _safe_float_array(x, name: str) -> np.ndarray:
    x = _as_1d(x, name).astype(float)
    return x


def _annotate_point_values(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    fmt: str = "{:.3f}",
    dy_points: int = -10,
    fontsize: int = 8,
    alpha: float = 0.75,
) -> None:
    """
    Annotate numeric y-values under each (x, y) point.

    dy_points < 0 places text below the marker in screen (point) coordinates.
    """
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            ax.annotate(
                fmt.format(float(y)),
                (float(x), float(y)),
                textcoords="offset points",
                xytext=(0, dy_points),
                ha="center",
                va="top" if dy_points < 0 else "bottom",
                fontsize=fontsize,
                alpha=alpha,
                clip_on=False,
            )


@dataclass
class BinnedSeries:
    centers: np.ndarray  # (B,)
    mean: np.ndarray     # (B,)
    count: np.ndarray    # (B,)
    edges: np.ndarray    # (B+1,)


def bin_mean_by_time(t: np.ndarray, values: np.ndarray, *, n_bins: int = 10, t_min: Optional[float] = None, t_max: Optional[float] = None) -> BinnedSeries:
    """
    Bin points by time into equal-width bins and compute mean within each bin.
    """
    t = _safe_float_array(t, "t")
    v = _safe_float_array(values, "values")
    if t.size != v.size:
        raise ValueError("t and values must have the same length")
    if t.size == 0:
        return BinnedSeries(np.array([]), np.array([]), np.array([]), np.array([]))

    if t_min is None:
        t_min = float(np.nanmin(t))
    if t_max is None:
        t_max = float(np.nanmax(t))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("invalid t_min/t_max for binning")

    edges = np.linspace(t_min, t_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, n_bins - 1)

    mean = np.full(n_bins, np.nan, dtype=float)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bin_id == b
        count[b] = int(np.sum(m))
        if count[b] > 0:
            mean[b] = float(np.nanmean(v[m]))

    return BinnedSeries(centers=centers, mean=mean, count=count, edges=edges)


def make_diagnosis_plots(
    *,
    t: np.ndarray,
    y_true: np.ndarray,
    y_hat: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    q: Optional[float] = None,
    alpha: float = 0.1,
    early_t_max: int = 30,
    n_time_bins: int = 10,
    y_max: Optional[float] = None,
    title_prefix: str = "",
) -> Dict[str, plt.Figure]:
    """
    Build the three requested diagnosis plots.

    Required for plot (1) and (3): t, y_true, lower, upper.
    Required for plot (2): y_true, y_hat (or None->0), sigma, q.

    Returns:
        dict of figures: {"coverage_vs_time": fig1, "score_hist": fig2, "width_vs_time": fig3}
    """
    t = _safe_float_array(t, "t")
    y_true = _safe_float_array(y_true, "y_true")
    if t.size != y_true.size:
        raise ValueError("t and y_true must have the same length")
    n = t.size

    if y_hat is None:
        y_hat = np.zeros(n, dtype=float)
    else:
        y_hat = _safe_float_array(y_hat, "y_hat")
        if y_hat.size != n:
            raise ValueError("y_hat must have same length as y_true")

    figs: Dict[str, plt.Figure] = {}
    prefix = (title_prefix + " - ") if title_prefix else ""

    # --- (1) coverage vs time (binned) ---
    if lower is None or upper is None:
        raise ValueError("lower and upper are required for coverage/width plots")
    lower = _safe_float_array(lower, "lower")
    upper = _safe_float_array(upper, "upper")
    if lower.size != n or upper.size != n:
        raise ValueError("lower/upper must have same length as y_true")

    covered = (y_true >= lower) & (y_true <= upper)
    cov_bin = bin_mean_by_time(t, covered.astype(float), n_bins=n_time_bins)

    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(cov_bin.centers, cov_bin.mean, marker="o", label="binned coverage")
    ax1.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1, alpha=0.7, label=f"target {1-alpha:.2f}")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("time (bin centers)")
    ax1.set_ylabel("coverage")
    ax1.set_title(prefix + "Coverage vs time (binned)")
    ax1.grid(True, alpha=0.25)
    for x, y, c in zip(cov_bin.centers, cov_bin.mean, cov_bin.count):
        if np.isfinite(y):
            ax1.annotate(f"n={int(c)}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.6)
    # numeric value under each marker
    _annotate_point_values(ax1, cov_bin.centers, cov_bin.mean, fmt="{:.3f}", dy_points=-10, fontsize=8, alpha=0.75)
    ax1.legend(loc="best")
    fig1.tight_layout()
    figs["coverage_vs_time"] = fig1

    # --- (3) mean interval width vs time (binned) ---
    width = upper - lower
    width_bin = bin_mean_by_time(t, width, n_bins=n_time_bins)

    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.plot(width_bin.centers, width_bin.mean, marker="o", label="binned mean width")
    if y_max is not None:
        y_max_f = float(y_max)
        if not np.isfinite(y_max_f) or y_max_f <= 0:
            raise ValueError("y_max must be a positive finite number when provided")
        ax3.set_ylim(0.0,y_max_f)
    ax3.set_xlabel("time (bin centers)")
    ax3.set_ylabel("avg interval length")
    ax3.set_title(prefix + "Mean interval length vs time (binned)")
    ax3.grid(True, alpha=0.25)
    for x, y, c in zip(width_bin.centers, width_bin.mean, width_bin.count):
        if np.isfinite(y):
            ax3.annotate(f"n={int(c)}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.6)
    # numeric value under each marker
    _annotate_point_values(ax3, width_bin.centers, width_bin.mean, fmt="{:.3f}", dy_points=-10, fontsize=8, alpha=0.75)
    ax3.legend(loc="best")
    fig3.tight_layout()
    figs["width_vs_time"] = fig3

    # --- (2) histogram of scaled score s for early vs late, with q marked ---
    if sigma is None or q is None:
        raise ValueError("sigma and q are required for the scaled-score histogram")
    sigma = _safe_float_array(sigma, "sigma")
    if sigma.size != n:
        raise ValueError("sigma must have same length as y_true")
    q = float(q)

    sigma_eff = np.maximum(sigma, 1e-12)
    s_score = np.abs(y_true - y_hat) / sigma_eff

    early_mask = t <= float(early_t_max)
    late_mask = t > float(early_t_max)

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    bins = 40
    ax2.hist(s_score[early_mask], bins=bins, density=True, alpha=0.5, label=f"early (t≤{early_t_max})")
    ax2.hist(s_score[late_mask], bins=bins, density=True, alpha=0.5, label=f"late (t>{early_t_max})")
    ax2.axvline(q, color="black", linestyle="--", linewidth=1.5, label=f"q={q:.3g}")
    ax2.set_xlabel("scaled score s = |y - yhat| / sigma")
    ax2.set_ylabel("density")
    ax2.set_title(prefix + "Scaled score histogram: early vs late")
    ax2.grid(True, alpha=0.25)

    # annotate tail rates
    def tail_rate(mask):
        m = int(np.sum(mask))
        if m == 0:
            return float("nan"), 0
        return float(np.mean(s_score[mask] > q)), m

    tr_early, n_early = tail_rate(early_mask)
    tr_late, n_late = tail_rate(late_mask)
    ax2.text(
        0.99,
        0.95,
        f"P(s>q) early={tr_early:.3f} (n={n_early})\nP(s>q) late={tr_late:.3f} (n={n_late})",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax2.legend(loc="best")
    fig2.tight_layout()
    figs["score_hist"] = fig2

    return figs


