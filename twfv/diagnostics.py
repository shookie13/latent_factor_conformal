"""
Small, notebook-friendly diagnostics for understanding early-time instability and undercoverage.

This module is intentionally light-weight: it provides summaries that help answer questions like:
  - Are early time points being "compressed" by the warp u_tilde?
  - Are control points (especially C[:,:,0]) drifting to extreme values?
  - Do scaled residuals |y-yhat|/sigma have heavier tails early in TEST than CAL?
"""

from __future__ import annotations
from scipy.stats import ks_2samp
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Sequence, Tuple

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


def _normalize_resample_methods(methods: Optional[Sequence[str]]) -> List[str]:
    if methods is None:
        methods = ("tmfv", "oracle", "avg")

    supported = {"scp", "tmfv", "oracle", "avg", "lcp", "cptdr", "cqr", "decomp"}
    out: List[str] = []
    seen = set()
    for method in methods:
        name = str(method).strip().lower()
        if name not in supported:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods are: {sorted(supported)}."
            )
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _collect_interval_artifact(
    *,
    art: Dict[str, Any],
    method: str,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int,
    q: Optional[float] = None,
    sigma_test: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    report = conditional_interval_report(
        y_true=y_true,
        lower=lower,
        upper=upper,
        x=x,
        y_hat=y_hat,
        n_bins=int(n_bins),
    )
    art.update(
        {
            f"lower_{method}": lower,
            f"upper_{method}": upper,
            f"covered_{method}": covered,
            f"coverage_{method}": float(np.mean(covered)) if covered.size else float("nan"),
            f"report_{method}": report,
        }
    )
    if q is not None:
        art[f"q_{method}"] = float(q)
    if sigma_test is not None:
        art[f"sigma_test_{method}"] = np.asarray(sigma_test, dtype=float)
    if extra:
        art.update(extra)


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
    methods: Optional[Sequence[str]] = None,
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
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
          - q_scp/q_tmfv/q_oracle/q_avg, coverage_* scalars
          - lower_*/upper_* arrays, covered_* arrays
          - report_* conditional reports (key "X" uses x=t_test)
          - (optional) method-specific outputs if selected via `methods`:
              lower_scp, upper_scp, covered_scp, coverage_scp, report_scp
              lower_lcp, upper_lcp, covered_lcp, coverage_lcp, report_lcp
              lower_cptdr, upper_cptdr, covered_cptdr, coverage_cptdr, report_cptdr
              lower_cqr, upper_cqr, covered_cqr, coverage_cqr, report_cqr
              lower_decomp, upper_decomp, covered_decomp, coverage_decomp, report_decomp
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must have shape (I,T,J)")
    I, T, J = Y.shape
    methods_use = _normalize_resample_methods(methods)
    method_kwargs = {str(k).strip().lower(): dict(v) for k, v in (method_kwargs or {}).items()}

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

    if "scp" in methods_use:
        from twfv.cp import NaiveAbsoluteResidualCP

        scp_kwargs = dict(method_kwargs.get("scp", {}))

    if "lcp" in methods_use:
        from twfv.cp import LocalizedSplitConformalCP

        lcp_kwargs = dict(method_kwargs.get("lcp", {}))

    if "cptdr" in methods_use:
        from twfv.cp import CPTDRSplitConformalCP

        cptdr_kwargs = dict(method_kwargs.get("cptdr", {}))

    if "cqr" in methods_use:
        try:
            from scripts.cqr import ConformalQuantileRegressor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "methods=['cqr', ...] requires `scripts/cqr.py` and its dependencies "
                "(notably scikit-learn). Install scikit-learn or remove 'cqr' from methods."
            ) from e
        cqr_kwargs = dict(method_kwargs.get("cqr", {}))
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
        cqr_alpha_use = float(cqr_kwargs.pop("cqr_alpha", alpha))
        cqr_random_state_base = int(cqr_kwargs.pop("random_state_base", 0))

    if "decomp" in methods_use:
        import torch
        from twfv.decomp import run_factor_channel_conformal

        decomp_kwargs = dict(method_kwargs.get("decomp", {}))
        if "em_result" not in decomp_kwargs:
            raise ValueError("method_kwargs['decomp'] must include 'em_result'.")
        decomp_em_result = decomp_kwargs["em_result"]
        decomp_L, decomp_a2, decomp_psi, decomp_s = (
            decomp_em_result[0], decomp_em_result[1],
            decomp_em_result[2], decomp_em_result[3],
        )
        decomp_device = decomp_L.device if hasattr(decomp_L, 'device') else torch.device('cpu')
        decomp_dtype = decomp_L.dtype if hasattr(decomp_L, 'dtype') else torch.float64
        decomp_factor_mode = str(decomp_kwargs.get("factor_mode", "leave_target_out"))
        decomp_score_kind = str(decomp_kwargs.get("score_kind", "standardized"))
        decomp_alpha = float(decomp_kwargs.get("alpha", alpha))
        Y_torch_decomp = torch.as_tensor(Y, dtype=decomp_dtype, device=decomp_device)

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

        if "scp" in methods_use:
            scp = NaiveAbsoluteResidualCP(shape_ref=Y, J=J, alpha=float(alpha), **scp_kwargs)
            scp.calibrate_from_fit(cal_idx, Y_cal_true, Y_cal_fit)
            intervals_scp = scp.predict_interval_from_fit(test_idx, Y_test_fit, alpha=float(alpha))
            lower_scp = intervals_scp[:, 0]
            upper_scp = intervals_scp[:, 1]
            q_scp = float(scp.q_abs) if np.isfinite(scp.q_abs) else 0.0

        if "tmfv" in methods_use:
            lower_tmfv, upper_tmfv, q_tmfv = conformal_scaled_abs(
                Y_cal_true, Y_cal_fit, sigma_cal, Y_test_fit, sigma_test, alpha=alpha
            )
        else:
            q_tmfv = float("nan")

        if "oracle" in methods_use:
            lower_oracle, upper_oracle, q_cal_oracle = conformal_scaled_abs(
                Y_cal_true, Y_cal_fit, sigma_cal_oracle, Y_test_fit, sigma_test_oracle, alpha=alpha
            )
        else:
            q_cal_oracle = float("nan")

        if "avg" in methods_use:
            lower_avg, upper_avg, q_avg = conformal_scaled_abs(
                Y_cal_true, Y_cal_fit, sigma_cal_avg, Y_test_fit, sigma_test_avg, alpha=alpha
            )
        else:
            q_avg = float("nan")

        S_cal_oracle = np.abs(Y_cal_true - Y_cal_fit) / sigma_cal_oracle
        S_test_oracle = np.abs(Y_test_true - Y_test_fit) / sigma_test_oracle
        delta_mean = S_test_oracle.mean() - S_cal_oracle.mean()
        q_test_oracle = _quantile_higher(S_test_oracle, alpha=float(alpha))
        delta_q = q_test_oracle - q_cal_oracle
        D, pval = ks_2samp(S_test_oracle, S_cal_oracle, alternative="two-sided", method="auto")

        # ids from flatten convention
        test_idx = np.asarray(test_idx, dtype=int).ravel()
        t_test = (test_idx % T).astype(int)
        subj_test = (test_idx // (J * T)).astype(int)
        chan_test = ((test_idx // T) % J).astype(int)

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
                delta_mean=float(delta_mean),
                delta_q=float(delta_q),
                D=float(D),
                pval=float(pval),
        )

        if "scp" in methods_use:
            _collect_interval_artifact(
                art=art,
                method="scp",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_scp,
                upper=upper_scp,
                n_bins=n_bins,
                q=q_scp,
            )

        if "tmfv" in methods_use:
            _collect_interval_artifact(
                art=art,
                method="tmfv",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_tmfv,
                upper=upper_tmfv,
                n_bins=n_bins,
                q=q_tmfv,
            )

        if "oracle" in methods_use:
            _collect_interval_artifact(
                art=art,
                method="oracle",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_oracle,
                upper=upper_oracle,
                n_bins=n_bins,
                q=q_cal_oracle,
            )

        if "avg" in methods_use:
            _collect_interval_artifact(
                art=art,
                method="avg",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_avg,
                upper=upper_avg,
                n_bins=n_bins,
                q=q_avg,
            )

        if "lcp" in methods_use:
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
            _collect_interval_artifact(
                art=art,
                method="lcp",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_lcp,
                upper=upper_lcp,
                n_bins=n_bins,
            )

        if "cptdr" in methods_use:
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
            _collect_interval_artifact(
                art=art,
                method="cptdr",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_cptdr,
                upper=upper_cptdr,
                n_bins=n_bins,
            )

        if "cqr" in methods_use:
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
            _collect_interval_artifact(
                art=art,
                method="cqr",
                y_true=y_test,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_cqr,
                upper=upper_cqr,
                n_bins=n_bins,
                extra={
                    "cqr_alpha": float(cqr_alpha_use),
                    "cqr_random_state": int(rs),
                },
            )

        if "decomp" in methods_use:
            flat_obs = np.zeros(I * J * T, dtype=bool)
            # flat_obs[train_idx] = True
            flat_obs[np.asarray(cal_idx, dtype=int).ravel()] = True
            M_decomp = flat_obs.reshape(I, J, T).transpose(0, 2, 1)
            M_decomp_t = torch.as_tensor(M_decomp, dtype=torch.bool, device=decomp_device)

            out_decomp = run_factor_channel_conformal(
                Y_torch_decomp, M_decomp_t,
                decomp_L, decomp_psi, decomp_s, decomp_a2,
                alpha=decomp_alpha,
                factor_mode=decomp_factor_mode,
                score_kind=decomp_score_kind,
            )

            lower_3d = out_decomp["intervals"]["lower"].cpu().detach().numpy()
            upper_3d = out_decomp["intervals"]["upper"].cpu().detach().numpy()

            i_k = test_idx // (J * T)
            rem_k = test_idx % (J * T)
            j_k = rem_k // T
            t_k = rem_k % T
            lower_decomp = lower_3d[i_k, t_k, j_k]
            upper_decomp = upper_3d[i_k, t_k, j_k]

            _collect_interval_artifact(
                art=art,
                method="decomp",
                y_true=Y_test_true,
                y_hat=Y_test_fit,
                x=t_test,
                lower=lower_decomp,
                upper=upper_decomp,
                n_bins=n_bins,
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