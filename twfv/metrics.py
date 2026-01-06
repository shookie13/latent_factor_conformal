"""
Evaluation utilities for prediction intervals.

Includes conditional (binned) coverage and interval length summaries using
quantile-based partitions of any scalar variable (e.g., time, y, |residual|, sigma).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _as_1d(x: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    return x


def _quantile_edges(values: np.ndarray, n_bins: int, quantiles: Optional[np.ndarray] = None) -> np.ndarray:
    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 1 or quantiles.size < 2:
        raise ValueError("quantiles must be a 1D array with at least 2 elements")
    if not (np.isclose(quantiles[0], 0.0) and np.isclose(quantiles[-1], 1.0)):
        raise ValueError("quantiles should start at 0 and end at 1")
    edges = np.quantile(values, quantiles)
    return edges


def binned_interval_stats_by_quantiles(
    *,
    values: np.ndarray | Iterable[float],
    y_true: np.ndarray | Iterable[float],
    lower: np.ndarray | Iterable[float],
    upper: np.ndarray | Iterable[float],
    n_bins: int = 10,
    quantiles: Optional[np.ndarray] = None,
    name: str = "value",
) -> Dict[str, Any]:
    """
    Compute conditional coverage and average interval length by quantile bins of `values`.

    Args:
        values: 1D array used for partitioning (e.g., time, y_true, abs residual, sigma).
        y_true: 1D array of true outcomes for the same points.
        lower, upper: 1D arrays of prediction interval bounds for the same points.
        n_bins: number of quantile bins if `quantiles` is None.
        quantiles: optional explicit quantile grid (length n_bins+1). Must start at 0 and end at 1.
        name: label for this partitioning variable (used in output).

    Returns:
        dict with:
          - name: variable name
          - edges: array (n_bins+1,) of bin edges
          - rows: list of dicts with per-bin stats:
              bin, left, right, n, coverage, avg_length, avg_value, avg_y
    """
    v = _as_1d(values, "values")
    y = _as_1d(y_true, "y_true")
    lo = _as_1d(lower, "lower")
    hi = _as_1d(upper, "upper")
    if not (v.size == y.size == lo.size == hi.size):
        raise ValueError("values, y_true, lower, upper must have the same length")
    if v.size == 0:
        return {"name": name, "edges": np.array([]), "rows": []}

    length = hi - lo
    covered = (y >= lo) & (y <= hi)

    edges = _quantile_edges(v, n_bins=n_bins, quantiles=quantiles)
    # Assign bins using interior edges; last edge is inclusive.
    bin_id = np.searchsorted(edges[1:-1], v, side="right")  # 0..n_bins-1

    rows: List[Dict[str, Any]] = []
    n_bins_eff = edges.size - 1
    for b in range(n_bins_eff):
        m = bin_id == b
        nb = int(np.sum(m))
        left = float(edges[b])
        right = float(edges[b + 1])
        if nb == 0:
            rows.append(
                dict(
                    bin=b,
                    left=left,
                    right=right,
                    n=0,
                    coverage=float("nan"),
                    avg_length=float("nan"),
                    avg_value=float("nan"),
                    avg_y=float("nan"),
                )
            )
            continue
        rows.append(
            dict(
                bin=b,
                left=left,
                right=right,
                n=nb,
                coverage=float(np.mean(covered[m])),
                avg_length=float(np.mean(length[m])),
                avg_value=float(np.mean(v[m])),
                avg_y=float(np.mean(y[m])),
            )
        )

    return {"name": name, "edges": edges, "rows": rows}


def conditional_interval_report(
    *,
    y_true: np.ndarray | Iterable[float],
    lower: np.ndarray | Iterable[float],
    upper: np.ndarray | Iterable[float],
    x: Optional[np.ndarray | Iterable[float]] = None,
    y_hat: Optional[np.ndarray | Iterable[float]] = None,
    n_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper: compute binned stats for common partition variables:
      - X (if provided)
      - Y (always)
      - |residual| (if y_hat provided)

    Returns a dict mapping keys {"X","Y","abs_resid"} -> binned stats dict.
    """
    y = _as_1d(y_true, "y_true")
    lo = _as_1d(lower, "lower")
    hi = _as_1d(upper, "upper")
    if not (y.size == lo.size == hi.size):
        raise ValueError("y_true, lower, upper must have the same length")

    out: Dict[str, Dict[str, Any]] = {}
    out["Y"] = binned_interval_stats_by_quantiles(values=y, y_true=y, lower=lo, upper=hi, n_bins=n_bins, name="Y")

    if x is not None:
        xv = _as_1d(x, "x")
        out["X"] = binned_interval_stats_by_quantiles(values=xv, y_true=y, lower=lo, upper=hi, n_bins=n_bins, name="X")

    if y_hat is not None:
        yh = _as_1d(y_hat, "y_hat")
        if yh.size != y.size:
            raise ValueError("y_hat must have same length as y_true")
        abs_resid = np.abs(y - yh)
        out["abs_resid"] = binned_interval_stats_by_quantiles(
            values=abs_resid, y_true=y, lower=lo, upper=hi, n_bins=n_bins, name="abs_resid"
        )

    return out


