"""
Small, notebook-friendly diagnostics for understanding early-time instability and undercoverage.

This module is intentionally light-weight: it provides summaries that help answer questions like:
  - Are early time points being "compressed" by the warp u_tilde?
  - Are control points (especially C[:,:,0]) drifting to extreme values?
  - Do scaled residuals |y-yhat|/sigma have heavier tails early in TEST than CAL?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import scipy
from .metrics import conditional_interval_report
from .simulate import split_cal_test_from_train_and_pmiss
import matplotlib.pyplot as plt


def unflatten(k: int, I: int, J: int, T: int) -> Tuple[int, int, int]:
    """
    Map a flattened index k into (i, j, t) using row-major order: k = i*(J*T) + j*T + t.
    """
    i = k // (J * T)
    rem = k % (J * T)
    j = rem // T
    t = rem % T
    return int(i), int(j), int(t)


def summarize_em_diagnostics(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the `history` returned by `run_em_like(..., collect_diagnostics=True)` into
    simple NumPy arrays for plotting.
    """
    out: Dict[str, Any] = {}
    for k in (
        "ll_outer",
        "ll_post_m",
        "C0_mean",
        "C0_std",
        "C0_max_abs",
        "kappa_mean",
        "kappa_std",
        "kappa_max",
        "u_head_inc_min",
        "dlogkappa_l2",
    ):
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


def _quantile_higher(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile with 'higher' interpolation:
        q_level = ceil((m+1)*(1-alpha))/m.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    m = int(scores.size)
    if m == 0:
        return float("nan")
    q_level = min(1.0, float(np.ceil((m + 1) * (1.0 - float(alpha))) / m))
    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, q_level, interpolation="higher"))


def conformal_scaled_abs(
    Y_cal_true: np.ndarray,
    Y_cal_fit: np.ndarray,
    sigma_cal: np.ndarray,
    Y_test_fit: np.ndarray,
    sigma_test: np.ndarray,
    *,
    alpha: float = 0.1,
    sigma_min: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Variance-scaled split conformal with absolute residual scores:
        S = |y - yhat| / sigma
    Returns:
        lower, upper, q  where interval is yhat ± q*sigma.
    """
    sigma_cal = np.maximum(np.asarray(sigma_cal, dtype=float), float(sigma_min))
    sigma_test = np.maximum(np.asarray(sigma_test, dtype=float), float(sigma_min))
    Y_cal_true = np.asarray(Y_cal_true, dtype=float)
    Y_cal_fit = np.asarray(Y_cal_fit, dtype=float)
    Y_test_fit = np.asarray(Y_test_fit, dtype=float)

    S_cal = np.abs(Y_cal_true - Y_cal_fit) / sigma_cal
    q = _quantile_higher(S_cal, alpha=float(alpha))
    if not np.isfinite(q):
        q = 0.0
    lower = Y_test_fit - q * sigma_test
    upper = Y_test_fit + q * sigma_test
    return lower, upper, float(q)


def resample_test_idx_and_collect_scaled_cp_artifacts(
    *,
    Y: np.ndarray,  # (I,T,J)
    train_idx: np.ndarray,  # flat indices kept fixed
    p_miss: np.ndarray,  # (I,T,J) or flat (I*J*T,)
    sigma_fit_flat: np.ndarray,  # flat (I*J*T,)
    sigma_oracle_flat: np.ndarray,  # flat (I*J*T,)
    t_res: int = 50,
    alpha: float = 0.1,
    seed: int = 123,
    n_bins: int = 5,
    # Optional: include LCP (Localized Split Conformal) baseline.
    include_lcp: bool = False,
    lcp_kwargs: Optional[Dict[str, Any]] = None,
    # Optional: include CPTD-R (Lin et al., 2022) baseline.
    include_cptdr: bool = False,
    cptdr_kwargs: Optional[Dict[str, Any]] = None,
    # Optional: include CQR (Conformalized Quantile Regression) baseline.
    include_cqr: bool = False,
    cqr_kwargs: Optional[Dict[str, Any]] = None,
    cqr_random_state_base: int = 0,
    cqr_alpha: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Resample TEST points `t_res` times using `p_miss` while keeping `train_idx` fixed.
    For each resample, CAL is the remainder (not train, not test).

    This matches the notebook flow:
      - yhat is 0 for all points (innovations)
      - build sigma_fit/oracle/avg
      - run variance-scaled split CP on (CAL -> TEST)
      - compute conditional reports by time x=t_test

    Returns
    -------
    artifacts_list:
        List of dicts. Each dict contains:
          - cal_idx, test_idx
          - t_test, subj_test, chan_test
          - q_tmfv/q_oracle/q_avg, coverage_* scalars
          - lower_*/upper_* arrays, covered_* arrays
          - report_* conditional reports (key "X" uses x=t_test)
          - (optional) LCP outputs if include_lcp=True:
              lower_lcp, upper_lcp, covered_lcp, coverage_lcp, report_lcp
          - (optional) CPTD-R outputs if include_cptdr=True:
              lower_cptdr, upper_cptdr, covered_cptdr, coverage_cptdr, report_cptdr
          - (optional) CQR outputs if include_cqr=True:
              lower_cqr, upper_cqr, covered_cqr, coverage_cqr, report_cqr
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must have shape (I,T,J)")
    I, T, J = Y.shape

    train_idx = np.asarray(train_idx, dtype=int).ravel()
    sigma_fit_flat = np.asarray(sigma_fit_flat, dtype=float).ravel()
    sigma_oracle_flat = np.asarray(sigma_oracle_flat, dtype=float).ravel()
    if sigma_fit_flat.size != I * J * T or sigma_oracle_flat.size != I * J * T:
        raise ValueError("sigma_*_flat must have length I*J*T in flatten convention k=i*(J*T)+j*T+t")

    # Full flattened Y in k-order: (I,J,T) -> flat
    Y_flat = np.transpose(Y, (0, 2, 1)).reshape(-1)

    # Build X features for all points in the same k-order:
    #   X[k] = [i+1, j+1, t+1] where k = i*(J*T) + j*T + t.
    ii = np.repeat(np.arange(1, I + 1), J * T)
    jj = np.tile(np.repeat(np.arange(1, J + 1), T), I)
    tt = np.tile(np.arange(1, T + 1), I * J)
    X_all = np.stack([ii, jj, tt], axis=1).astype(float)  # (I*J*T, 3)

    if include_lcp:
        from twfv.cp import LocalizedSplitConformalCP

        lcp_kwargs = dict(lcp_kwargs or {})

    if include_cptdr:
        from twfv.cp import CPTDRSplitConformalCP

        cptdr_kwargs = dict(cptdr_kwargs or {})

    if include_cqr:
        try:
            from scripts.cqr import ConformalQuantileRegressor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "include_cqr=True requires `scripts/cqr.py` and its dependencies "
                "(notably scikit-learn). Install scikit-learn or set include_cqr=False."
            ) from e
        cqr_kwargs = dict(cqr_kwargs or {})
        # IMPORTANT: `ConformalQuantileRegressor(alpha=...)` uses alpha for CP miscoverage,
        # but sklearn.linear_model.QuantileRegressor(alpha=...) uses alpha for regularization.
        # To avoid ambiguous/buggy calls, treat cqr_kwargs["alpha"] as *regressor alpha*
        # and remap it to `reg_alpha`.
        if "alpha" in cqr_kwargs and "reg_alpha" not in cqr_kwargs:
            cqr_kwargs["reg_alpha"] = cqr_kwargs.pop("alpha")
        # Miscoverage level for CQR uses this function's `alpha` by default.
        # Important: do NOT read it from cqr_kwargs["alpha"], because some regressors
        # (notably sklearn.linear_model.QuantileRegressor) use `alpha` as a *regularization*
        # hyperparameter. Use `cqr_alpha=` if you need CQR to use a different miscoverage.
        cqr_alpha_use = float(alpha if cqr_alpha is None else cqr_alpha)

    rng = np.random.default_rng(int(seed))
    artifacts: List[Dict[str, Any]] = []
    for b in range(int(t_res)):
        # Deterministic per-resample split via RNG state.
        cal_idx, test_idx = split_cal_test_from_train_and_pmiss(train_idx, p_miss, rng=rng)

        # Train data is fixed across resamples.
        X_train = X_all[train_idx]
        y_train = Y_flat[train_idx]

        # True y and fitted mean (0) on cal/test
        Y_cal_true = Y_flat[cal_idx]
        Y_test_true = Y_flat[test_idx]
        Y_cal_fit = np.zeros_like(Y_cal_true)
        Y_test_fit = np.zeros_like(Y_test_true)

        # Sigmas
        sigma_cal = sigma_fit_flat[cal_idx]
        sigma_test = sigma_fit_flat[test_idx]
        sigma_cal_oracle = sigma_oracle_flat[cal_idx]
        sigma_test_oracle = sigma_oracle_flat[test_idx]
        sigma_cal_avg = 0.5 * (sigma_cal + sigma_cal_oracle)
        sigma_test_avg = 0.5 * (sigma_test + sigma_test_oracle)
        q95 = scipy.stats.norm.ppf(0.95)
        lower_oracle = -q95 * sigma_test_oracle+Y_test_fit
        upper_oracle =  q95 * sigma_test_oracle+Y_test_fit
        lower_avg = -q95 * sigma_test_avg+Y_test_fit
        upper_avg =  q95 * sigma_test_avg+Y_test_fit
        # CP intervals
        lower_tmfv, upper_tmfv, q_tmfv = conformal_scaled_abs(
            Y_cal_true, Y_cal_fit, sigma_cal, Y_test_fit, sigma_test, alpha=alpha
        )
        # lower_oracle, upper_oracle, q_oracle = conformal_scaled_abs(
        #     Y_cal_true, Y_cal_fit, sigma_cal_oracle, Y_test_fit, sigma_test_oracle, alpha=alpha
        # )
        # lower_avg, upper_avg, q_avg = conformal_scaled_abs(
        #     Y_cal_true, Y_cal_fit, sigma_cal_avg, Y_test_fit, sigma_test_avg, alpha=alpha
        # )

        covered_tmfv = (Y_test_true >= lower_tmfv) & (Y_test_true <= upper_tmfv)
        covered_oracle = (Y_test_true >= lower_oracle) & (Y_test_true <= upper_oracle)
        covered_avg = (Y_test_true >= lower_avg) & (Y_test_true <= upper_avg)

        # ids from flatten convention
        test_idx = np.asarray(test_idx, dtype=int).ravel()
        t_test = (test_idx % T).astype(int)
        subj_test = (test_idx // (J * T)).astype(int)
        chan_test = ((test_idx // T) % J).astype(int)

        report_tmfv = conditional_interval_report(
            y_true=Y_test_true, lower=lower_tmfv, upper=upper_tmfv, x=t_test, y_hat=Y_test_fit, n_bins=int(n_bins)
        )
        report_oracle = conditional_interval_report(
            y_true=Y_test_true, lower=lower_oracle, upper=upper_oracle, x=t_test, y_hat=Y_test_fit, n_bins=int(n_bins)
        )
        report_avg = conditional_interval_report(
            y_true=Y_test_true, lower=lower_avg, upper=upper_avg, x=t_test, y_hat=Y_test_fit, n_bins=int(n_bins)
        )

        art = dict(
                resample_id=int(b),
                seed=int(seed),
                I=int(I),
                T=int(T),
                J=int(J),
                alpha=float(alpha),
                train_idx=train_idx.copy(),
                cal_idx=np.asarray(cal_idx, dtype=int).copy(),
                test_idx=test_idx.copy(),
                # test-point arrays
                t_test=t_test,
                subj_test=subj_test,
                chan_test=chan_test,
                Y_test_true=Y_test_true,
                Y_test_fit=Y_test_fit,
                # sigma (test) per method
                sigma_test=sigma_test,
                sigma_test_oracle=sigma_test_oracle,
                sigma_test_avg=sigma_test_avg,
                # intervals & coverage
                q_tmfv=float(q_tmfv),
                q_oracle=float(q95),
                q_avg=float(q95),
                lower_tmfv=lower_tmfv,
                upper_tmfv=upper_tmfv,
                lower_oracle=lower_oracle,
                upper_oracle=upper_oracle,
                lower_avg=lower_avg,
                upper_avg=upper_avg,
                covered_tmfv=covered_tmfv,
                covered_oracle=covered_oracle,
                covered_avg=covered_avg,
                coverage_tmfv=float(np.mean(covered_tmfv)) if covered_tmfv.size else float("nan"),
                coverage_oracle=float(np.mean(covered_oracle)) if covered_oracle.size else float("nan"),
                coverage_avg=float(np.mean(covered_avg)) if covered_avg.size else float("nan"),
                # conditional reports (by time in key "X")
                report_tmfv=report_tmfv,
                report_oracle=report_oracle,
                report_avg=report_avg,
        )

        if include_lcp:
            # LCP uses only indices (to recover time via unflatten) plus (y_true, y_fit).
            # Here y_fit is 0 (innovations), consistent with this function's conventions.
            lcp = LocalizedSplitConformalCP(shape_ref=Y, J=J, alpha=float(alpha), **lcp_kwargs)
            lcp.calibrate_from_fit(
                X_cal_idx=np.asarray(cal_idx, dtype=int).ravel(),
                Y_cal_true=Y_cal_true,
                Y_cal_fit=Y_cal_fit,
            )
            intervals_lcp = lcp.predict_interval_from_fit(
                X_test_idx=test_idx,
                Y_test_fit=Y_test_fit,
                alpha=float(alpha),
            )
            lower_lcp = intervals_lcp[:, 0]
            upper_lcp = intervals_lcp[:, 1]
            covered_lcp = (Y_test_true >= lower_lcp) & (Y_test_true <= upper_lcp)
            report_lcp = conditional_interval_report(
                y_true=Y_test_true, lower=lower_lcp, upper=upper_lcp, x=t_test, y_hat=Y_test_fit, n_bins=int(n_bins)
            )
            art.update(
                dict(
                    lower_lcp=lower_lcp,
                    upper_lcp=upper_lcp,
                    covered_lcp=covered_lcp,
                    coverage_lcp=float(np.mean(covered_lcp)) if covered_lcp.size else float("nan"),
                    report_lcp=report_lcp,
                )
            )

        if include_cptdr:
            # CPTD-R uses temporally- and cross-sectionally-informed normalization.
            # We supply `train_idx` as additional "history" points to build residual histories,
            # but only CAL points are used for conformal quantile calibration.
            cp = CPTDRSplitConformalCP(shape_ref=Y, J=J, alpha=float(alpha), **cptdr_kwargs)
            cp.calibrate_from_fit(
                X_cal_idx=np.asarray(cal_idx, dtype=int).ravel(),
                Y_cal_true=Y_cal_true,
                Y_cal_fit=Y_cal_fit,
                X_hist_idx=np.asarray(train_idx, dtype=int).ravel(),
                Y_hist_true=Y_flat[np.asarray(train_idx, dtype=int).ravel()],
                Y_hist_fit=np.zeros_like(Y_flat[np.asarray(train_idx, dtype=int).ravel()]),
            )
            intervals_cptdr = cp.predict_interval_from_fit(
                X_test_idx=test_idx,
                Y_test_fit=Y_test_fit,
                alpha=float(alpha),
            )
            lower_cptdr = intervals_cptdr[:, 0]
            upper_cptdr = intervals_cptdr[:, 1]
            covered_cptdr = (Y_test_true >= lower_cptdr) & (Y_test_true <= upper_cptdr)
            report_cptdr = conditional_interval_report(
                y_true=Y_test_true,
                lower=lower_cptdr,
                upper=upper_cptdr,
                x=t_test,
                y_hat=Y_test_fit,
                n_bins=int(n_bins),
            )
            art.update(
                dict(
                    lower_cptdr=lower_cptdr,
                    upper_cptdr=upper_cptdr,
                    covered_cptdr=covered_cptdr,
                    coverage_cptdr=float(np.mean(covered_cptdr)) if covered_cptdr.size else float("nan"),
                    report_cptdr=report_cptdr,
                )
            )

        if include_cqr:
            # CQR uses (X_train, y_train) for training and (X_calib, y_calib) for conformal calibration.
            X_calib = X_all[cal_idx]
            y_calib = Y_cal_true  # same values
            X_test = X_all[test_idx]
            y_test = Y_test_true

            # Make per-resample random_state deterministic.
            rs = int(cqr_random_state_base) + int(seed) * 10_000 + int(b)
            cqr = ConformalQuantileRegressor(alpha=cqr_alpha_use, random_state=rs, **cqr_kwargs)
            cqr.fit(X_train, y_train, X_calib, y_calib)
            lower_cqr, upper_cqr = cqr.predict_interval(X_test)
            covered_cqr = (y_test >= lower_cqr) & (y_test <= upper_cqr)
            report_cqr = conditional_interval_report(
                y_true=y_test, lower=lower_cqr, upper=upper_cqr, x=t_test, y_hat=Y_test_fit, n_bins=int(n_bins)
            )
            art.update(
                dict(
                    lower_cqr=lower_cqr,
                    upper_cqr=upper_cqr,
                    covered_cqr=covered_cqr,
                    coverage_cqr=float(np.mean(covered_cqr)) if covered_cqr.size else float("nan"),
                    report_cqr=report_cqr,
                    cqr_alpha=float(cqr_alpha_use),
                    cqr_random_state=int(rs),
                )
            )

        artifacts.append(art)
    return artifacts

def plot_em_history_grid(history, *, max_cols=3, figsize_per=(5.2, 3.2), suptitle="EM diagnostics"):
    """
    Plot a grid of run_em_like(history) diagnostics.
    Expects `history` dict returned by run_em_like(..., collect_diagnostics=True).
    """
    # Pick the most useful diagnostics; only plot those that exist.
    keys = [
        # likelihood + timing
        ("ll_outer", "ll_outer"),
        ("ll_post_m", "ll_post_m"),
        ("t_outer_total_s", "t_outer_total_s"),
        ("t_estep_s", "t_estep_s"),
        ("t_warp_s", "t_warp_s"),
        ("t_mstep_s", "t_mstep_s"),
        ("t_post_s", "t_post_s"),
        # parameter deltas
        ("dL_rel", "dL_rel"),
        ("dC_rel", "dC_rel"),
        ("dlogpsi_l2", "dlogpsi_l2"),
        ("dlogs_l2", "dlogs_l2"),
        ("dlogkappa_l2", "dlogkappa_l2"),
        ("du_rel", "du_rel"),
        ("da2_rel", "da2_rel"),
        # identifiability-aware deltas
        ("dvar_y_rmse", "dvar_y_rmse"),
        ("dvar_y_rel_l2", "dvar_y_rel_l2"),
        ("dvar_y_max_rel", "dvar_y_max_rel"),
        ("dH_frob_mean", "dH_frob_mean"),
        ("dH_rel_frob_mean", "dH_rel_frob_mean"),
        ("dH_rel_frob_max", "dH_rel_frob_max"),
        # existing early-time diagnostics (optional)
        ("C0_max_abs", "max|C[:,:,0]|"),
        ("kappa_mean", "mean kappa"),
        ("kappa_std", "std kappa"),
        ("kappa_max", "max kappa"),
        ("u_head_inc_min", "min Δu (head)"),
    ]

    series = []
    for k, label in keys:
        if k in history:
            y = np.asarray(history[k], dtype=float)
            if y.ndim == 1:
                series.append((k, label, y))
            elif y.ndim == 2:
                # For head-series arrays, plot first element and max across head
                series.append((k + "[:,0]", label + " (t0)", y[:, 0]))
                series.append((k + ".max", label + " (max over head)", np.nanmax(y, axis=1)))

    if len(series) == 0:
        raise ValueError("No recognized diagnostics found in history.")

    n = len(series)
    ncols = min(int(max_cols), n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False,
    )

    x = np.arange(len(next(iter(series))[2]))
    for idx, (k, label, y) in enumerate(series):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("outer iteration")
        ax.set_ylabel(k)

        # Optional: log scale for strictly-positive sequences with heavy tails
        if np.all(np.isfinite(y)) and np.nanmin(y) > 0 and ("_rel" in k or "rmse" in k or k.startswith("t_")):
            # don't force; you can comment this out if you prefer linear
            pass

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return fig

# Example usage (history is em_result[-1] if you returned history):
# hist = em_result[-1]
# plot_em_history_grid(hist, max_cols=3)