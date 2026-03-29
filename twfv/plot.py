import numpy as np
import matplotlib.pyplot as plt
import os
import re
import matplotlib.colors as mcolors
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

def unflatten(k: int, I: int, J: int, T: int) -> Tuple[int, int, int]:
    """
    Map a flattened index k into (i, j, t) using row-major order: k = i*(J*T) + j*T + t.
    """
    i = k // (J * T)
    rem = k % (J * T)
    j = rem // T
    t = rem % T
    return int(i), int(j), int(t)


def plot_intervals_on_series_from_q_sigma(
    *,
    Y: np.ndarray,                 # (I,T,J) full numeric
    I: int,
    T: int,
    J: int,
    subj_i: int,
    var_j: int,
    train_idx: np.ndarray,         # flattened indices
    cal_idx: np.ndarray,           # flattened indices
    test_idx: np.ndarray,          # flattened indices
    # Fitted mean and sigma for ALL points (optional yhat; required sigma)
    # - yhat can be None to default to 0 everywhere (appropriate for innovations)
    # - sigma can be shape (I,T,J) or flat shape (I*J*T,) using flatten convention k=i*(J*T)+j*T+t
    yhat: np.ndarray | None = None,
    sigma_tmfv: np.ndarray | None = None,
    # Quantiles
    q_tmfv: float,
    q_scp: float,
    # Visual options
    title: str | None = None,
    show_y_line: bool = True,
    train_center: str = "yhat",  # "y" or "yhat" (default: yhat/0 for all splits)
    band_alpha: float = 0.18,
    band_interp: bool = True,
):
    """
    Plot Y[subj_i,:,var_j] and overlay *pointwise* prediction intervals for three splits
    (train/cal/test) for two methods:
      - TWFV scaled: yhat ± q_tmfv * sigma_tmfv
      - SCP unscaled: yhat ± q_scp

    This version draws ONE band per method across the full time axis (for the chosen i,j),
    which is feasible because sigma_tmfv is provided for all (i,t,j).

    Assumptions:
      - idx arrays use flatten convention k=i*(J*T)+j*T+t (same as twfv.cp.unflatten).
      - sigma_tmfv provided for all points, either (I,T,J) or flat (I*J*T,) in (I,J,T)-flatten order.
      - yhat is optional; if provided, it should be (I,T,J) or flat (I*J*T,) in the same order as sigma.
      - train_center="y" centers intervals at observed y for training points (legacy option).
        By default, all splits use yhat as the center; if yhat is None, center is 0 everywhere.
    """
    Y = np.asarray(Y, dtype=float)
    y_line = Y[subj_i, :, var_j]
    tt = np.arange(T)

    if sigma_tmfv is None:
        raise ValueError("sigma_tmfv is required")

    def to_flat(arr: np.ndarray | None, name: str) -> np.ndarray | None:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            if a.size != I * J * T:
                raise ValueError(f"{name} flat must have length I*J*T={I*J*T}, got {a.size}")
            return a
        if a.ndim == 3:
            if a.shape != (I, T, J):
                raise ValueError(f"{name} must have shape (I,T,J)={(I,T,J)}, got {a.shape}")
            return np.transpose(a, (0, 2, 1)).reshape(-1)  # (I,J,T) -> flat
        raise ValueError(f"{name} must be flat (I*J*T,) or (I,T,J), got shape {a.shape}")

    sigma_flat = to_flat(sigma_tmfv, "sigma_tmfv")
    yhat_flat = to_flat(yhat, "yhat") if yhat is not None else None

    # Build full time-series centers and sigma for this (subj_i, var_j)
    # Note: flatten convention is k=i*(J*T)+j*T+t, so we build k for each t.
    k_series = np.array([subj_i * (J * T) + var_j * T + t for t in range(T)], dtype=int)
    sigma_series = sigma_flat[k_series]
    if yhat_flat is None:
        yhat_series = np.zeros(T, dtype=float)
    else:
        yhat_series = yhat_flat[k_series]

    # Collect per-time observed y, plus per-time predictions/intervals for each split
    y_train = np.full(T, np.nan)
    y_cal = np.full(T, np.nan)
    y_test = np.full(T, np.nan)

    # For each split, map time -> position in idx array (for this i,j)
    def build_pos_map(idx_arr: np.ndarray):
        pos = {}
        for n, k in enumerate(idx_arr.tolist()):
            i, j, t = unflatten(int(k), I, J, T)
            if i == subj_i and j == var_j:
                pos[t] = n
        return pos

    pos_train = build_pos_map(train_idx)
    pos_cal = build_pos_map(cal_idx)
    pos_test = build_pos_map(test_idx)

    for t, n in pos_train.items():
        y_train[t] = Y[subj_i, t, var_j]
    for t, n in pos_cal.items():
        y_cal[t] = Y[subj_i, t, var_j]
    for t, n in pos_test.items():
        y_test[t] = Y[subj_i, t, var_j]

    # Helper to scatter points for a split
    def plot_split_points(name, pos_map, marker, color):
        if len(pos_map) == 0:
            return
        t_pts = np.array(sorted(pos_map.keys()), dtype=int)
        y_true_pts = Y[subj_i, t_pts, var_j]

        # raw points
        plt.scatter(t_pts, y_true_pts, s=22, marker=marker, color=color, alpha=0.9, label=f"{name} points")

    plt.figure(figsize=(12, 4))
    if show_y_line:
        plt.plot(tt, y_line, color="C0", lw=1.0, alpha=0.35, label="Y (line)")

    # One continuous band per method across all t
    lo_tmfv = yhat_series - q_tmfv * sigma_series
    hi_tmfv = yhat_series + q_tmfv * sigma_series
    lo_scp = yhat_series - float(q_scp)
    hi_scp = yhat_series + float(q_scp)

    plt.fill_between(tt, lo_tmfv, hi_tmfv, color="red", alpha=band_alpha, label="TWFV band")
    plt.fill_between(tt, lo_scp, hi_scp, color="purple", alpha=band_alpha, label="SCP band")

    # Points
    plot_split_points("train", pos_train, marker=".", color="C0")
    plot_split_points("cal", pos_cal, marker="o", color="C0")
    plot_split_points("test", pos_test, marker="x", color="C0")

    # Highlight misses on TEST split only (computed using the full bands at those t)
    if len(pos_test) > 0:
        t_test_pts = np.array(sorted(pos_test.keys()), dtype=int)
        y_test_pts = Y[subj_i, t_test_pts, var_j]
        miss_tmfv = (y_test_pts < lo_tmfv[t_test_pts]) | (y_test_pts > hi_tmfv[t_test_pts])
        miss_scp = (y_test_pts < lo_scp[t_test_pts]) | (y_test_pts > hi_scp[t_test_pts])
        both = miss_tmfv & miss_scp
        tmfv_only = miss_tmfv & ~miss_scp
        scp_only = ~miss_tmfv & miss_scp

        if np.any(tmfv_only):
            plt.scatter(t_test_pts[tmfv_only], y_test_pts[tmfv_only], s=55, marker="x", color="red", label="miss (TWFV)")
        if np.any(scp_only):
            plt.scatter(t_test_pts[scp_only], y_test_pts[scp_only], s=55, marker="x", color="purple", label="miss (SCP)")
        if np.any(both):
            plt.scatter(t_test_pts[both], y_test_pts[both], s=65, marker="x", color="black", label="miss (both)")

    plt.title(title or f"Subject {subj_i}, variable {var_j}: intervals on train/cal/test points")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend(loc="best", ncol=3)
    plt.tight_layout()
    plt.show()


def plot_compare_conditional_reports(
    report_a: dict,
    report_b: dict,
    *,
    label_a: str = "TWFV",
    label_b: str = "SCP",
    # Coverage plotting mode:
    # - "coverage": plot raw conditional coverage (backward-compatible default)
    # - "gap": plot |cov-target| - |oracle_cov-target|
    # - "gap_rate": plot |cov-target| / |oracle_cov-target|
    coverage_mode: str = "coverage",
    # Oracle report needed for coverage_mode in {"gap","gap_rate"}.
    report_oracle: dict | None = None,
    label_oracle: str = "Oracle",
    oracle_eps: float = 1e-12,
    keys: tuple[str, ...] = ("X", "Y", "abs_resid"),
    title: str | None = None,
    alpha_target: float | None = None,
    # Optional: if provided, add two extra graphs:
    #  (a) coverage by channel j over all (i,t) points
    #  (b) histogram of per-subject coverage over (t,j)
    channel_ids: np.ndarray | None = None,     # (n_points,) int in [0, J-1]
    subject_ids: np.ndarray | None = None,     # (n_points,) int in [0, I-1]
    covered_a: np.ndarray | None = None,       # (n_points,) bool
    covered_b: np.ndarray | None = None,       # (n_points,) bool
    covered_oracle: np.ndarray | None = None,  # (n_points,) bool (required for gap modes if extras are used)
    show_point_values: bool = True,
    save_plots: bool = False,
    save_dir: str = "result_img",
    save_basename: str | None = None,
    show: bool = True,
    dpi: int = 200,
    annotation_fontsize: int = 10,
    show_heatmap_subject_channel: bool = True,
    heatmap_mode: str = "channel_subject",  # "channel_subject" | "subject_channel" | "timebin_subject" | "timebin_channel"
    time_ids: np.ndarray | None = None,     # (n_points,) time index for each provided point (needed for timebin modes)
    n_time_bins: int = 10,
    t_min: float | None = None,
    t_max: float | None = None,
    heatmap_style: str = "imshow",  # "imshow" | "contourf"
    heatmap_levels: int = 15,
    heatmap_upsample: int = 1,
):
    """
    Compare two conditional interval reports produced by twfv.metrics.conditional_interval_report.

    Each report is a dict mapping keys (e.g. "X","Y","abs_resid") -> {"edges":..., "rows":[...]}
    where each row has fields: bin, left, right, n, coverage, avg_length.

    Produces a grid of plots: one row per key, two columns (coverage, avg interval length).

    Coverage plot can be switched to oracle-relative diagnostics via `coverage_mode`:
      - "coverage": plot coverage as-is.
      - "gap": plot delta absolute coverage gap:
            |cov - target| - |oracle_cov - target|
      - "gap_rate": plot relative absolute coverage gap rate:
            |cov - target| / |oracle_cov - target|

    If channel_ids/subject_ids and covered_* are provided, also produces:
      1) coverage by channel j (aggregated over all provided points)
      2) histogram of per-subject coverage (aggregated over all provided points)
      3) a 2D coverage matrix as either a heatmap ("imshow") or filled contour ("contourf").
         Choose the axes via `heatmap_mode`:
           - "channel_subject": (rows=channel, cols=subject)  [default, backward-compatible]
           - "subject_channel": (rows=subject, cols=channel)
           - "timebin_subject": (rows=time bins, cols=subject)   (requires time_ids)
           - "timebin_channel": (rows=time bins, cols=channel)   (requires time_ids)

    Notes on smoothing:
      - For `heatmap_style="contourf"`, the matrix is upsampled using `scipy.ndimage.zoom`
        with cubic interpolation to make contours smoother.
    """
    mode = str(coverage_mode).lower().strip()
    if mode not in {"coverage", "gap", "gap_rate"}:
        raise ValueError("coverage_mode must be one of {'coverage','gap','gap_rate'}")
    if mode in {"gap", "gap_rate"} and report_oracle is None:
        raise ValueError("report_oracle is required when coverage_mode is 'gap' or 'gap_rate'")

    # Filter keys to those available in both reports
    if mode == "coverage":
        keys_use = [k for k in keys if (k in report_a and k in report_b)]
    else:
        keys_use = [k for k in keys if (k in report_a and k in report_b and k in report_oracle)]
    if len(keys_use) == 0:
        raise ValueError("No overlapping keys found in both reports.")

    nrows = len(keys_use)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5 * nrows), squeeze=False)

    # Target coverage
    target = (1.0 - float(alpha_target)) if alpha_target is not None else None

    def _gap_vals(cov: np.ndarray, cov_o: np.ndarray) -> np.ndarray:
        if target is None:
            raise ValueError("alpha_target must be provided when coverage_mode is 'gap' or 'gap_rate'")
        a = np.abs(cov - target)
        b = np.abs(cov_o - target)
        return a - b

    def _gap_rate_vals(cov: np.ndarray, cov_o: np.ndarray) -> np.ndarray:
        if target is None:
            raise ValueError("alpha_target must be provided when coverage_mode is 'gap' or 'gap_rate'")
        a = np.abs(cov - target)
        b = np.abs(cov_o - target)
        denom = np.maximum(b, float(oracle_eps))
        # If oracle gap is essentially zero, rate is not meaningful -> NaN.
        out = a / denom
        out = np.where(b > float(oracle_eps), out, np.nan)
        return out

    for row_i, k in enumerate(keys_use):
        rows_a = report_a[k]["rows"]
        rows_b = report_b[k]["rows"]
        if mode == "coverage":
            nb = min(len(rows_a), len(rows_b))
            rows_o = None
        else:
            rows_o = report_oracle[k]["rows"]
            nb = min(len(rows_a), len(rows_b), len(rows_o))

        cov_a = np.array([rows_a[i]["coverage"] for i in range(nb)], dtype=float)
        cov_b = np.array([rows_b[i]["coverage"] for i in range(nb)], dtype=float)
        cov_o = None if rows_o is None else np.array([rows_o[i]["coverage"] for i in range(nb)], dtype=float)
        len_a = np.array([rows_a[i]["avg_length"] for i in range(nb)], dtype=float)
        len_b = np.array([rows_b[i]["avg_length"] for i in range(nb)], dtype=float)
        n_a = np.array([rows_a[i]["n"] for i in range(nb)], dtype=float)
        n_b = np.array([rows_b[i]["n"] for i in range(nb)], dtype=float)

        # Bin x-axis: use bin centers for readability
        centers = np.array([(rows_a[i]["left"] + rows_a[i]["right"]) / 2.0 for i in range(nb)], dtype=float)

        ax_cov = axes[row_i, 0]
        ax_len = axes[row_i, 1]

        if mode == "coverage":
            y_a_cov = cov_a
            y_b_cov = cov_b
            y_o_cov = None
            ylabel_cov = "coverage"
            ref_line = (1.0 - float(alpha_target)) if alpha_target is not None else None
            ref_label = "target"
        elif mode == "gap":
            y_a_cov = _gap_vals(cov_a, cov_o)
            y_b_cov = _gap_vals(cov_b, cov_o)
            y_o_cov = np.zeros_like(y_a_cov)
            ylabel_cov = "|cov-target| - |oracle_cov-target|"
            ref_line = 0.0
            ref_label = "oracle baseline"
        else:  # gap_rate
            y_a_cov = _gap_rate_vals(cov_a, cov_o)
            y_b_cov = _gap_rate_vals(cov_b, cov_o)
            y_o_cov = np.ones_like(y_a_cov)
            ylabel_cov = "|cov-target| / |oracle_cov-target|"
            ref_line = 1.0
            ref_label = "oracle baseline"

        line_a_cov, = ax_cov.plot(centers, y_a_cov, marker="o", label=f"{label_a}")
        line_b_cov, = ax_cov.plot(centers, y_b_cov, marker="o", label=f"{label_b}")
        color_a_cov = line_a_cov.get_color()
        color_b_cov = line_b_cov.get_color()
        if ref_line is not None:
            ax_cov.axhline(float(ref_line), color="black", linestyle="--", linewidth=1, alpha=0.6, label=ref_label)
        if mode == "coverage":
            ax_cov.set_title(f"{k}: coverage by quantile bins")
        elif mode == "gap":
            ax_cov.set_title(f"{k}: delta abs coverage gap vs {label_oracle} by quantile bins")
        else:
            ax_cov.set_title(f"{k}: abs coverage gap rate vs {label_oracle} by quantile bins")
        ax_cov.set_xlabel(f"{k} (bin centers)")
        ax_cov.set_ylabel(ylabel_cov)
        if mode == "coverage":
            ax_cov.set_ylim(0.0, 1.0)
        ax_cov.grid(True, alpha=0.25)
        ax_cov.legend(loc="best")

        line_a_len, = ax_len.plot(centers, len_a, marker="o", label=f"{label_a}")
        line_b_len, = ax_len.plot(centers, len_b, marker="o", label=f"{label_b}")
        color_a_len = line_a_len.get_color()
        color_b_len = line_b_len.get_color()
        ax_len.set_title(f"{k}: avg interval length by quantile bins")
        ax_len.set_xlabel(f"{k} (bin centers)")
        ax_len.set_ylabel("avg length")
        ax_len.grid(True, alpha=0.25)
        ax_len.legend(loc="best")

        if show_point_values:
            for i in range(nb):
                if np.isfinite(centers[i]) and np.isfinite(y_a_cov[i]):
                    ax_cov.annotate(
                        f"{y_a_cov[i]:.2f}",
                        (centers[i], y_a_cov[i]),
                        textcoords="offset points",
                        xytext=(-5, -12),
                        ha="right",
                        fontsize=annotation_fontsize,
                        alpha=0.85,
                        color=color_a_cov,
                    )
                if np.isfinite(centers[i]) and np.isfinite(y_b_cov[i]):
                    ax_cov.annotate(
                        f"{y_b_cov[i]:.2f}",
                        (centers[i], y_b_cov[i]),
                        textcoords="offset points",
                        xytext=(5, -12),
                        ha="left",
                        fontsize=annotation_fontsize,
                        alpha=0.85,
                        color=color_b_cov,
                    )
                if np.isfinite(centers[i]) and np.isfinite(len_a[i]):
                    ax_len.annotate(
                        f"{len_a[i]:.2f}",
                        (centers[i], len_a[i]),
                        textcoords="offset points",
                        xytext=(-5, -12),
                        ha="right",
                        fontsize=annotation_fontsize,
                        alpha=0.85,
                        color=color_a_len,
                    )
                if np.isfinite(centers[i]) and np.isfinite(len_b[i]):
                    ax_len.annotate(
                        f"{len_b[i]:.2f}",
                        (centers[i], len_b[i]),
                        textcoords="offset points",
                        xytext=(5, -12),
                        ha="left",
                        fontsize=annotation_fontsize,
                        alpha=0.85,
                        color=color_b_len,
                    )

        # Annotate effective sample sizes lightly (use min of the two for compactness)
        for i in range(nb):
            if np.isfinite(centers[i]):
                y_anchor = y_a_cov[i] if np.isfinite(y_a_cov[i]) else (0.0 if mode != "gap_rate" else 1.0)
                ax_cov.annotate(f"n={int(min(n_a[i], n_b[i]))}", (centers[i], y_anchor),
                                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=annotation_fontsize, alpha=0.6)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    def _sanitize(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
        return s[:120] if len(s) > 120 else s

    base = save_basename or (title if title is not None else f"{label_a}_vs_{label_b}")
    base = _sanitize(base)

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{base}_bins.png"), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- Extra graphs: coverage by channel and coverage distribution by subject ---
    if channel_ids is None or subject_ids is None or covered_a is None or covered_b is None:
        return

    ch = np.asarray(channel_ids)
    sb = np.asarray(subject_ids)
    ca = np.asarray(covered_a).astype(bool)
    cb = np.asarray(covered_b).astype(bool)
    co = None if covered_oracle is None else np.asarray(covered_oracle).astype(bool)

    if not (ch.ndim == sb.ndim == ca.ndim == cb.ndim == 1):
        raise ValueError("channel_ids, subject_ids, covered_a, covered_b must be 1D arrays")
    if not (ch.size == sb.size == ca.size == cb.size):
        raise ValueError("channel_ids/subject_ids/covered_a/covered_b must have the same length")
    if mode in {"gap", "gap_rate"}:
        if target is None:
            raise ValueError("alpha_target must be provided when coverage_mode is 'gap' or 'gap_rate'")
        if co is None:
            raise ValueError("covered_oracle is required for extra plots when coverage_mode is 'gap' or 'gap_rate'")
        if co.ndim != 1 or co.size != ca.size:
            raise ValueError("covered_oracle must be a 1D array with the same length as covered_a/covered_b")

    # coverage by channel
    J_max = int(np.max(ch)) + 1 if ch.size > 0 else 0
    cov_by_ch_a = np.full(J_max, np.nan, dtype=float)
    cov_by_ch_b = np.full(J_max, np.nan, dtype=float)
    cov_by_ch_o = np.full(J_max, np.nan, dtype=float) if mode in {"gap", "gap_rate"} else None
    n_by_ch = np.zeros(J_max, dtype=int)
    for j in range(J_max):
        m = ch == j
        n_by_ch[j] = int(np.sum(m))
        if n_by_ch[j] > 0:
            cov_by_ch_a[j] = float(np.mean(ca[m]))
            cov_by_ch_b[j] = float(np.mean(cb[m]))
            if cov_by_ch_o is not None:
                cov_by_ch_o[j] = float(np.mean(co[m]))

    # per-subject coverage distribution
    I_max = int(np.max(sb)) + 1 if sb.size > 0 else 0
    cov_by_sb_a = np.full(I_max, np.nan, dtype=float)
    cov_by_sb_b = np.full(I_max, np.nan, dtype=float)
    cov_by_sb_o = np.full(I_max, np.nan, dtype=float) if mode in {"gap", "gap_rate"} else None
    n_by_sb = np.zeros(I_max, dtype=int)
    for i in range(I_max):
        m = sb == i
        n_by_sb[i] = int(np.sum(m))
        if n_by_sb[i] > 0:
            cov_by_sb_a[i] = float(np.mean(ca[m]))
            cov_by_sb_b[i] = float(np.mean(cb[m]))
            if cov_by_sb_o is not None:
                cov_by_sb_o[i] = float(np.mean(co[m]))

    if mode == "coverage":
        y_ch_a = cov_by_ch_a
        y_ch_b = cov_by_ch_b
        y_sb_a = cov_by_sb_a
        y_sb_b = cov_by_sb_b
        ylabel_extra = "coverage (over provided points)"
        ref_line = (1.0 - float(alpha_target)) if alpha_target is not None else None
        ref_label = "target"
        hist_xlabel = "per-subject coverage"
        # Set bins similar to old behavior.
        hist_bins = np.linspace(0.5, 1.0, 21)
    elif mode == "gap":
        y_ch_a = np.abs(cov_by_ch_a - target) - np.abs(cov_by_ch_o - target)
        y_ch_b = np.abs(cov_by_ch_b - target) - np.abs(cov_by_ch_o - target)
        y_sb_a = np.abs(cov_by_sb_a - target) - np.abs(cov_by_sb_o - target)
        y_sb_b = np.abs(cov_by_sb_b - target) - np.abs(cov_by_sb_o - target)
        ylabel_extra = "|cov-target| - |oracle_cov-target|"
        ref_line = 0.0
        ref_label = "oracle baseline"
        hist_xlabel = "per-subject delta abs coverage gap"
        # Robust symmetric bins around 0.
        vals = np.concatenate([y_sb_a[np.isfinite(y_sb_a)], y_sb_b[np.isfinite(y_sb_b)]]) if I_max > 0 else np.array([])
        if vals.size > 0:
            q = np.nanquantile(np.abs(vals), 0.95)
            q = float(max(q, 1e-3))
            hist_bins = np.linspace(-q, q, 31)
        else:
            hist_bins = np.linspace(-0.2, 0.2, 31)
    else:  # gap_rate
        o_gap_ch = np.abs(cov_by_ch_o - target)
        o_gap_sb = np.abs(cov_by_sb_o - target)
        a_gap_ch = np.abs(cov_by_ch_a - target)
        b_gap_ch = np.abs(cov_by_ch_b - target)
        a_gap_sb = np.abs(cov_by_sb_a - target)
        b_gap_sb = np.abs(cov_by_sb_b - target)
        y_ch_a = np.where(o_gap_ch > float(oracle_eps), a_gap_ch / np.maximum(o_gap_ch, float(oracle_eps)), np.nan)
        y_ch_b = np.where(o_gap_ch > float(oracle_eps), b_gap_ch / np.maximum(o_gap_ch, float(oracle_eps)), np.nan)
        y_sb_a = np.where(o_gap_sb > float(oracle_eps), a_gap_sb / np.maximum(o_gap_sb, float(oracle_eps)), np.nan)
        y_sb_b = np.where(o_gap_sb > float(oracle_eps), b_gap_sb / np.maximum(o_gap_sb, float(oracle_eps)), np.nan)
        ylabel_extra = "|cov-target| / |oracle_cov-target|"
        ref_line = 1.0
        ref_label = "oracle baseline"
        hist_xlabel = "per-subject abs coverage gap rate"
        vals = np.concatenate([y_sb_a[np.isfinite(y_sb_a)], y_sb_b[np.isfinite(y_sb_b)]]) if I_max > 0 else np.array([])
        if vals.size > 0:
            hi = float(np.nanquantile(vals, 0.95))
            hi = max(hi, 1.5)
            hist_bins = np.linspace(0.0, hi, 31)
        else:
            hist_bins = np.linspace(0.0, 2.0, 31)

    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax_ch = axes2[0]
    xs = np.arange(J_max)
    line_a_ch, = ax_ch.plot(xs, y_ch_a, marker="o", label=label_a)
    line_b_ch, = ax_ch.plot(xs, y_ch_b, marker="o", label=label_b)
    color_a_ch = line_a_ch.get_color()
    color_b_ch = line_b_ch.get_color()
    if ref_line is not None:
        ax_ch.axhline(float(ref_line), color="black", linestyle="--", linewidth=1, alpha=0.6, label=ref_label)
    ax_ch.set_xlabel("channel j")
    ax_ch.set_ylabel(ylabel_extra)
    if mode == "coverage":
        ax_ch.set_ylim(0.0, 1.0)
        ax_ch.set_title("Coverage by channel")
    elif mode == "gap":
        ax_ch.set_title(f"Delta abs coverage gap by channel vs {label_oracle}")
    else:
        ax_ch.set_title(f"Abs coverage gap rate by channel vs {label_oracle}")
    ax_ch.grid(True, alpha=0.25)
    ax_ch.legend(loc="best")
    for j in range(J_max):
        if n_by_ch[j] > 0 and np.isfinite(y_ch_a[j]):
            ax_ch.annotate(f"n={n_by_ch[j]}", (j, y_ch_a[j]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.6)
            if show_point_values:
                ax_ch.annotate(
                    f"{y_ch_a[j]:.2f}",
                    (j, y_ch_a[j]),
                    textcoords="offset points",
                    xytext=(-5, -12),
                    ha="right",
                    fontsize=annotation_fontsize,
                    alpha=0.85,
                    color=color_a_ch,
                )
        if show_point_values and n_by_ch[j] > 0 and np.isfinite(y_ch_b[j]):
            ax_ch.annotate(
                f"{y_ch_b[j]:.2f}",
                (j, y_ch_b[j]),
                textcoords="offset points",
                xytext=(5, -12),
                ha="left",
                fontsize=annotation_fontsize,
                alpha=0.85,
                color=color_b_ch,
            )

    ax_sb = axes2[1]
    a_vals = y_sb_a[np.isfinite(y_sb_a)]
    b_vals = y_sb_b[np.isfinite(y_sb_b)]
    ax_sb.hist(a_vals, bins=hist_bins, alpha=0.5, label=label_a)
    ax_sb.hist(b_vals, bins=hist_bins, alpha=0.5, label=label_b)
    if ref_line is not None:
        ax_sb.axvline(float(ref_line), color="black", linestyle="--", linewidth=1, alpha=0.6, label=ref_label)
    ax_sb.set_xlabel(hist_xlabel)
    ax_sb.set_ylabel("count of subjects")
    if mode == "coverage":
        ax_sb.set_title("Distribution of per-subject coverage")
    elif mode == "gap":
        ax_sb.set_title(f"Distribution of per-subject delta abs coverage gap vs {label_oracle}")
    else:
        ax_sb.set_title(f"Distribution of per-subject abs coverage gap rate vs {label_oracle}")
    ax_sb.grid(True, alpha=0.25)
    ax_sb.legend(loc="best")

    if title is not None:
        fig2.suptitle(title + " (channel/subject diagnostics)")
        fig2.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig2.tight_layout()

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig2.savefig(os.path.join(save_dir, f"{base}_channel_subject.png"), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig2)

    # --- Extra heatmap: coverage by (subject, channel) ---
    if not show_heatmap_subject_channel:
        return

    heatmap_mode_use = str(heatmap_mode).lower().strip()
    # Backward-compatible alias
    if heatmap_mode_use == "subject_timebin":
        heatmap_mode_use = "timebin_subject"

    allowed_modes = {"channel_subject", "subject_channel", "timebin_subject", "timebin_channel"}
    if heatmap_mode_use not in allowed_modes:
        raise ValueError(f"heatmap_mode must be one of {sorted(allowed_modes)}")

    # time binning helper (equal-width bins)
    def _time_bin_ids(t: np.ndarray, *, n_bins: int, t_min_: float | None, t_max_: float | None) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float).ravel()
        if t.size == 0:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return np.zeros(0, dtype=int), centers
        if t_min_ is None:
            t_min_ = float(np.nanmin(t))
        if t_max_ is None:
            t_max_ = float(np.nanmax(t))
        if not np.isfinite(t_min_) or not np.isfinite(t_max_) or t_max_ <= t_min_:
            raise ValueError("invalid t_min/t_max for time binning")
        edges = np.linspace(t_min_, t_max_, int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        b = np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, int(n_bins) - 1)
        return b.astype(int), centers

    # Dimensions
    I_max = int(np.max(sb)) + 1 if sb.size > 0 else 0
    J_max = int(np.max(ch)) + 1 if ch.size > 0 else 0

    # If time-bin modes are requested, require time_ids
    if heatmap_mode_use in {"timebin_subject", "timebin_channel"}:
        if time_ids is None:
            raise ValueError("time_ids is required when heatmap_mode uses time bins")
        tt = np.asarray(time_ids)
        if tt.ndim != 1 or tt.size != sb.size:
            raise ValueError("time_ids must be a 1D array with the same length as covered arrays")
        tb, t_centers = _time_bin_ids(tt, n_bins=int(n_time_bins), t_min_=t_min, t_max_=t_max)
        T_bins = int(n_time_bins)
    else:
        tb = None
        t_centers = None
        T_bins = 0

    # Build coverage matrices according to mode
    if heatmap_mode_use == "channel_subject":
        cov_mat_a = np.full((J_max, I_max), np.nan, dtype=float)  # rows=channel, cols=subject
        cov_mat_b = np.full((J_max, I_max), np.nan, dtype=float)
        count_mat = np.zeros((J_max, I_max), dtype=int)
        for i in range(I_max):
            mi = sb == i
            if not np.any(mi):
                continue
            for j in range(J_max):
                m = mi & (ch == j)
                c = int(np.sum(m))
                count_mat[j, i] = c
                if c > 0:
                    cov_mat_a[j, i] = float(np.mean(ca[m]))
                    cov_mat_b[j, i] = float(np.mean(cb[m]))
        xlab, ylab = "subject", "channel"
        xticks = np.arange(I_max)
        yticks = np.arange(J_max)
    elif heatmap_mode_use == "subject_channel":
        cov_mat_a = np.full((I_max, J_max), np.nan, dtype=float)  # rows=subject, cols=channel
        cov_mat_b = np.full((I_max, J_max), np.nan, dtype=float)
        count_mat = np.zeros((I_max, J_max), dtype=int)
        for i in range(I_max):
            mi = sb == i
            if not np.any(mi):
                continue
            for j in range(J_max):
                m = mi & (ch == j)
                c = int(np.sum(m))
                count_mat[i, j] = c
                if c > 0:
                    cov_mat_a[i, j] = float(np.mean(ca[m]))
                    cov_mat_b[i, j] = float(np.mean(cb[m]))
        xlab, ylab = "channel", "subject"
        xticks = np.arange(J_max)
        yticks = np.arange(I_max)
    elif heatmap_mode_use == "timebin_subject":
        cov_mat_a = np.full((T_bins, I_max), np.nan, dtype=float)  # rows=time bins, cols=subject
        cov_mat_b = np.full((T_bins, I_max), np.nan, dtype=float)
        count_mat = np.zeros((T_bins, I_max), dtype=int)
        for b in range(T_bins):
            mb = tb == b
            if not np.any(mb):
                continue
            for i in range(I_max):
                m = mb & (sb == i)
                c = int(np.sum(m))
                count_mat[b, i] = c
                if c > 0:
                    cov_mat_a[b, i] = float(np.mean(ca[m]))
                    cov_mat_b[b, i] = float(np.mean(cb[m]))
        xlab, ylab = "subject", "time bin"
        xticks = np.arange(I_max)
        yticks = np.arange(T_bins)
    else:  # "timebin_channel"
        cov_mat_a = np.full((T_bins, J_max), np.nan, dtype=float)  # rows=time bins, cols=channel
        cov_mat_b = np.full((T_bins, J_max), np.nan, dtype=float)
        count_mat = np.zeros((T_bins, J_max), dtype=int)
        for b in range(T_bins):
            mb = tb == b
            if not np.any(mb):
                continue
            for j in range(J_max):
                m = mb & (ch == j)
                c = int(np.sum(m))
                count_mat[b, j] = c
                if c > 0:
                    cov_mat_a[b, j] = float(np.mean(ca[m]))
                    cov_mat_b[b, j] = float(np.mean(cb[m]))
        xlab, ylab = "channel", "time bin"
        xticks = np.arange(J_max)
        yticks = np.arange(T_bins)

    if mode == "coverage":
        target_hm = (1.0 - float(alpha_target)) if alpha_target is not None else 0.9
        Z_a = cov_mat_a
        Z_b = cov_mat_b
        center = target_hm
        ref_text = f"target={target_hm:.2f}"
        all_vals = np.concatenate([Z_a[np.isfinite(Z_a)], Z_b[np.isfinite(Z_b)]])
        if all_vals.size == 0:
            return
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        vmin = min(vmin, center)
        vmax = max(vmax, center)
        if np.isclose(vmin, vmax):
            vmin = max(0.0, center - 0.1)
            vmax = min(1.0, center + 0.1)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")
        cbar_label = "coverage"
    else:
        if target is None:
            raise ValueError("alpha_target must be provided when coverage_mode is 'gap' or 'gap_rate'")
        # Build oracle coverage matrix in the same shape as cov_mat_a/cov_mat_b using co + ids.
        # We'll reuse the already-computed sb/ch arrays.
        # cov_mat_o is aligned with the selected heatmap_mode.
        if heatmap_mode_use == "channel_subject":
            cov_mat_o = np.full((J_max, I_max), np.nan, dtype=float)
            for i in range(I_max):
                mi = sb == i
                if not np.any(mi):
                    continue
                for j in range(J_max):
                    m = mi & (ch == j)
                    if np.any(m):
                        cov_mat_o[j, i] = float(np.mean(co[m]))
        elif heatmap_mode_use == "subject_channel":
            cov_mat_o = np.full((I_max, J_max), np.nan, dtype=float)
            for i in range(I_max):
                mi = sb == i
                if not np.any(mi):
                    continue
                for j in range(J_max):
                    m = mi & (ch == j)
                    if np.any(m):
                        cov_mat_o[i, j] = float(np.mean(co[m]))
        elif heatmap_mode_use == "timebin_subject":
            cov_mat_o = np.full((T_bins, I_max), np.nan, dtype=float)
            for b in range(T_bins):
                mb = tb == b
                if not np.any(mb):
                    continue
                for i in range(I_max):
                    m = mb & (sb == i)
                    if np.any(m):
                        cov_mat_o[b, i] = float(np.mean(co[m]))
        else:  # timebin_channel
            cov_mat_o = np.full((T_bins, J_max), np.nan, dtype=float)
            for b in range(T_bins):
                mb = tb == b
                if not np.any(mb):
                    continue
                for j in range(J_max):
                    m = mb & (ch == j)
                    if np.any(m):
                        cov_mat_o[b, j] = float(np.mean(co[m]))

        if mode == "gap":
            Z_a = np.abs(cov_mat_a - target) - np.abs(cov_mat_o - target)
            Z_b = np.abs(cov_mat_b - target) - np.abs(cov_mat_o - target)
            center = max(np.nanmedian(Z_a), np.nanmedian(Z_b))
            ref_text = f"oracle={label_oracle}, target={target:.2f}"
            cbar_label = "|cov-target| - |oracle_cov-target|"
        else:  # gap_rate
            o_gap = np.abs(cov_mat_o - target)
            a_gap = np.abs(cov_mat_a - target)
            b_gap = np.abs(cov_mat_b - target)
            Z_a = np.where(o_gap > float(oracle_eps), a_gap / np.maximum(o_gap, float(oracle_eps)), np.nan)
            Z_b = np.where(o_gap > float(oracle_eps), b_gap / np.maximum(o_gap, float(oracle_eps)), np.nan)
            # center = 1.0
            center = max(np.nanmedian(Z_a), np.nanmedian(Z_b))
            ref_text = f"oracle={label_oracle}, target={target:.2f}"
            cbar_label = "|cov-target| / |oracle_cov-target|"

        all_vals = np.concatenate([Z_a[np.isfinite(Z_a)], Z_b[np.isfinite(Z_b)]])
        if all_vals.size == 0:
            return
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        vmin = min(vmin, center)
        vmax = max(vmax, center)
        print(vmax)
        print(vmin)
        if mode == 'gap':
            vmin = max(vmin, 0)
            vmax = min(vmax, 10)
        else:
            vmin = max(vmin, 1)
            vmax = min(vmax, 10)
        if np.isclose(vmin, vmax):
            vmin = center - 0.1
            vmax = center + 0.1
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")

    heatmap_style_use = str(heatmap_style).lower().strip()
    if heatmap_style_use not in {"imshow", "contourf"}:
        raise ValueError("heatmap_style must be one of {'imshow','contourf'}")
    levels = int(heatmap_levels)
    if levels < 3:
        raise ValueError("heatmap_levels must be >= 3")
    up = int(max(1, heatmap_upsample))

    fig3, axes3 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, cov_mat_a.shape[1] * 0.35), max(4.5, cov_mat_a.shape[0] * 0.6) * 2),
        constrained_layout=True,
    )
    if heatmap_style_use == "imshow":
        im0 = axes3[0].imshow(Z_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes3[1].imshow(Z_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        # Filled contours on an (x=subject, y=channel) grid.
        # Upsample using scipy.ndimage.zoom (cubic interpolation) to make contours smoother.
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            # Fill NaNs with center so interpolation doesn't create holes.
            Zp = np.where(np.isfinite(Zp), Zp, float(center))
            if up > 1:
                # zoom factors: (rows=channel, cols=subject)
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(Z_a)
        Zb = prep(Z_b)
        # Build a coordinate grid consistent with the chosen heatmap_mode.
        # cov_mat_* is always shaped as (rows=y-axis, cols=x-axis).
        if heatmap_mode_use in {"channel_subject"}:
            y_end = float(J_max - 1)
            x_end = float(I_max - 1)
        elif heatmap_mode_use in {"subject_channel"}:
            y_end = float(I_max - 1)
            x_end = float(J_max - 1)
        elif heatmap_mode_use in {"timebin_subject"}:
            y_end = float(T_bins - 1)
            x_end = float(I_max - 1)
        elif heatmap_mode_use in {"timebin_channel"}:
            y_end = float(T_bins - 1)
            x_end = float(J_max - 1)
        else:  # should be impossible due to validation above
            y_end = float(Za.shape[0] - 1)
            x_end = float(Za.shape[1] - 1)

        y = np.linspace(0.0, y_end, Za.shape[0])
        x = np.linspace(0.0, x_end, Za.shape[1])
        X, Y = np.meshgrid(x, y)

        im0 = axes3[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes3[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)
    if mode == "coverage":
        axes3[0].set_title(f"{label_a}: coverage ({heatmap_mode_use}) ({ref_text})")
    elif mode == "gap":
        axes3[0].set_title(f"{label_a}: delta abs gap ({heatmap_mode_use}) ({ref_text})")
    else:
        axes3[0].set_title(f"{label_a}: abs gap rate ({heatmap_mode_use}) ({ref_text})")
    axes3[0].set_ylabel(ylab)
    axes3[0].set_xlabel(xlab)

    if mode == "coverage":
        axes3[1].set_title(f"{label_b}: coverage ({heatmap_mode_use}) ({ref_text})")
    elif mode == "gap":
        axes3[1].set_title(f"{label_b}: delta abs gap ({heatmap_mode_use}) ({ref_text})")
    else:
        axes3[1].set_title(f"{label_b}: abs gap rate ({heatmap_mode_use}) ({ref_text})")
    axes3[1].set_ylabel(ylab)
    axes3[1].set_xlabel(xlab)

    # Ticks (keep light to avoid clutter)
    if xticks.size <= 30:
        axes3[0].set_xticks(xticks)
        axes3[1].set_xticks(xticks)
    if yticks.size <= 30:
        axes3[0].set_yticks(yticks)
        axes3[1].set_yticks(yticks)

    # Shared colorbar
    cbar = fig3.colorbar(im1, ax=axes3.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label)

    if title is not None:
        fig3.suptitle(title + " (subject×channel heatmaps)")

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig3.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_channel.png"), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig3)
        

def plot_compare_conditional_reports_aggregated(
    artifacts: list[dict],
    *,
    # Which stored artifacts to compare
    report_key_a: str = "report_tmfv",
    report_key_b: str = "report_scp",
    covered_key_a: str = "covered_tmfv",
    covered_key_b: str = "covered_scp",
    lower_key_a: str | None = None,
    upper_key_a: str | None = None,
    lower_key_b: str | None = None,
    upper_key_b: str | None = None,
    t_key: str = "t_test",
    channel_key: str = "chan_test",
    subject_key: str = "subj_test",
    # Labels / plotting
    label_a: str = "TWFV",
    label_b: str = "SCP",
    key: str = "X",  # time bins in your notebook
    n_time_bins: int = 10,
    alpha_target: float = 0.1,
    title: str | None = None,
    show: bool = True,
    save_plots: bool = False,
    save_dir: str = "result_img",
    save_basename: str | None = None,
    dpi: int = 200,
    # Heatmap style
    heatmap_style: str = "imshow",  # "imshow" | "contourf"
    heatmap_levels: int = 15,
    heatmap_upsample: int = 1,
    # Oracle-relative length heatmap options (requires oracle bounds in artifacts)
    oracle_lower_key: str = "lower_oracle",
    oracle_upper_key: str = "upper_oracle",
    oracle_shorter_coef: float = 1.0,
):
    """
    Aggregated version of `plot_compare_conditional_reports` for batch runs.

    Instead of plotting per-run conditional coverage, it plots the *absolute coverage gap*:
        gap := | coverage - (1 - alpha_target) |
    and summarizes across runs with mean ± sd.

    Produces 10 visuals:
      1) abs coverage gap by time bins (mean±sd across runs) for A/B
      2) avg interval length by time bins (mean±sd across runs) for A/B
      2b) interval score by time bins (mean±sd across runs) for A/B
      3) abs coverage gap by channel (mean±sd across runs) for A/B
      4) distribution (histogram) of per-subject abs coverage gap (pooled), with per-run mean(|gap|) stats
      5) subject×channel abs coverage gap heatmap (mean across runs) for A/B
      6) subject×timebin abs coverage gap heatmap (mean across runs) for A/B
      7) subject×channel mean interval length heatmap (mean across runs) for A/B
      8) subject×timebin mean interval length heatmap (mean across runs) for A/B
      9) subject×channel mean interval score heatmap (mean across runs) for A/B
      10) timebin×subject mean interval score heatmap (mean across runs) for A/B
    """
    if len(artifacts) == 0:
        raise ValueError("artifacts list is empty")

    target = 1.0 - float(alpha_target)
    alpha = float(alpha_target)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha_target must be in (0, 1)")

    def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        Winkler / interval score (smaller is better):
          S = (hi-lo) + (2/alpha)*(lo-y)*1[y<lo] + (2/alpha)*(y-hi)*1[y>hi]
        """
        y = np.asarray(y, dtype=float).ravel()
        lo = np.asarray(lo, dtype=float).ravel()
        hi = np.asarray(hi, dtype=float).ravel()
        if not (y.size == lo.size == hi.size):
            raise ValueError("interval_score: y, lo, hi must have the same length")
        width = hi - lo
        below = y < lo
        above = y > hi
        pen = np.zeros_like(width, dtype=float)
        if np.any(below):
            pen[below] = (2.0 / alpha) * (lo[below] - y[below])
        if np.any(above):
            pen[above] = (2.0 / alpha) * (y[above] - hi[above])
        return width + pen

    def _infer_bounds_keys(covered_key: str) -> tuple[str, str]:
        # Convention used by run_batch.py artifacts: covered_X, lower_X, upper_X.
        suffix = covered_key.split("covered_", 1)[-1]
        return f"lower_{suffix}", f"upper_{suffix}"

    def _sanitize(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
        return s[:120] if len(s) > 120 else s

    base = save_basename or (title if title is not None else f"{label_a}_vs_{label_b}_agg")
    base = _sanitize(base)

    # --- (1)&(2) bins: abs coverage gap and interval length ---
    if lower_key_a is None or upper_key_a is None:
        lower_key_a, upper_key_a = _infer_bounds_keys(covered_key_a)
    if lower_key_b is None or upper_key_b is None:
        lower_key_b, upper_key_b = _infer_bounds_keys(covered_key_b)

    cov_a_runs = []
    cov_b_runs = []
    len_a_runs = []
    len_b_runs = []
    score_a_runs = []
    score_b_runs = []
    n_runs = len(artifacts)
    centers_ref = None

    if key != "X":
        # Fallback: use stored quantile reports (note: centers may differ across runs).
        # Kept mainly for completeness; for time ("X") use fixed bins below.
        for art in artifacts:
            rep_a = art[report_key_a]
            rep_b = art[report_key_b]
            if key not in rep_a or key not in rep_b:
                raise KeyError(f"Key '{key}' not found in reports for one artifact")
            rows_a = rep_a[key]["rows"]
            rows_b = rep_b[key]["rows"]
            nb = min(len(rows_a), len(rows_b))
            cov_a_runs.append(np.array([rows_a[i]["coverage"] for i in range(nb)], dtype=float))
            cov_b_runs.append(np.array([rows_b[i]["coverage"] for i in range(nb)], dtype=float))
            len_a_runs.append(np.array([rows_a[i]["avg_length"] for i in range(nb)], dtype=float))
            len_b_runs.append(np.array([rows_b[i]["avg_length"] for i in range(nb)], dtype=float))
            centers = np.array([(rows_a[i]["left"] + rows_a[i]["right"]) / 2.0 for i in range(nb)], dtype=float)
            centers_ref = centers if centers_ref is None else centers_ref

            # Interval score by the same (per-run) quantile bins.
            # Note: edges may differ across runs; this branch does not enforce alignment.
            edges = np.asarray(rep_a[key]["edges"], dtype=float).ravel()
            if edges.size < 2:
                score_a_runs.append(np.full(nb, np.nan, dtype=float))
                score_b_runs.append(np.full(nb, np.nan, dtype=float))
                continue
            y = np.asarray(art["Y_test_true"], dtype=float).ravel()
            yhat = np.asarray(art.get("Y_test_fit", np.zeros_like(y)), dtype=float).ravel()
            if key == "Y":
                v = y
            elif key == "abs_resid":
                v = np.abs(y - yhat)
            else:
                raise ValueError("For key != 'X', key must be one of {'Y','abs_resid'} when computing interval score.")
            lo_a = np.asarray(art[lower_key_a], dtype=float).ravel()
            hi_a = np.asarray(art[upper_key_a], dtype=float).ravel()
            lo_b = np.asarray(art[lower_key_b], dtype=float).ravel()
            hi_b = np.asarray(art[upper_key_b], dtype=float).ravel()
            if not (v.size == y.size == lo_a.size == hi_a.size == lo_b.size == hi_b.size):
                raise ValueError("Per-run test arrays must have the same length when computing interval score.")
            b = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, (edges.size - 2))
            sc_a = interval_score(y, lo_a, hi_a)
            sc_b = interval_score(y, lo_b, hi_b)
            sbin_a = np.full(nb, np.nan, dtype=float)
            sbin_b = np.full(nb, np.nan, dtype=float)
            for k_bin in range(nb):
                m = b == k_bin
                if np.any(m):
                    sbin_a[k_bin] = float(np.mean(sc_a[m]))
                    sbin_b[k_bin] = float(np.mean(sc_b[m]))
            score_a_runs.append(sbin_a)
            score_b_runs.append(sbin_b)
        centers = np.asarray(centers_ref, dtype=float)
        C_a = np.stack(cov_a_runs, axis=0)
        C_b = np.stack(cov_b_runs, axis=0)
        L_a = np.stack(len_a_runs, axis=0)
        L_b = np.stack(len_b_runs, axis=0)
        S_a = np.stack(score_a_runs, axis=0)
        S_b = np.stack(score_b_runs, axis=0)
    else:
        # Time bins: use a fixed, equal-width grid on [0, T-1] so bins align across seeds.
        nb = int(n_time_bins)
        if nb <= 1:
            raise ValueError("n_time_bins must be >= 2")

        # Determine a shared T across artifacts.
        T_ref = int(artifacts[0].get("T", 0))
        if T_ref <= 1:
            raise ValueError("Invalid T in artifacts[0]")
        for art in artifacts[1:]:
            if int(art.get("T", T_ref)) != T_ref:
                raise RuntimeError("All artifacts must share the same T for fixed time binning.")

        edges = np.linspace(0.0, float(T_ref - 1), nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def bin_id(t: np.ndarray) -> np.ndarray:
            # Like diagnosis.bin_mean_by_time: last edge inclusive.
            return np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, nb - 1)

        for art in artifacts:
            t = np.asarray(art[t_key], dtype=float).ravel()
            ca = np.asarray(art[covered_key_a], dtype=bool).ravel()
            cb = np.asarray(art[covered_key_b], dtype=bool).ravel()
            y = np.asarray(art["Y_test_true"], dtype=float).ravel()
            lo_a = np.asarray(art[lower_key_a], dtype=float).ravel()
            hi_a = np.asarray(art[upper_key_a], dtype=float).ravel()
            lo_b = np.asarray(art[lower_key_b], dtype=float).ravel()
            hi_b = np.asarray(art[upper_key_b], dtype=float).ravel()
            if not (t.size == ca.size == cb.size == y.size == lo_a.size == hi_a.size == lo_b.size == hi_b.size):
                raise ValueError("Per-run test arrays must have the same length for fixed time binning.")

            b = bin_id(t)
            cov_a = np.full(nb, np.nan, dtype=float)
            cov_b = np.full(nb, np.nan, dtype=float)
            len_a = np.full(nb, np.nan, dtype=float)
            len_b = np.full(nb, np.nan, dtype=float)
            scr_a = np.full(nb, np.nan, dtype=float)
            scr_b = np.full(nb, np.nan, dtype=float)
            sc_a = interval_score(y, lo_a, hi_a)
            sc_b = interval_score(y, lo_b, hi_b)
            for k_bin in range(nb):
                m = b == k_bin
                if not np.any(m):
                    continue
                cov_a[k_bin] = float(np.mean(ca[m]))
                cov_b[k_bin] = float(np.mean(cb[m]))
                len_a[k_bin] = float(np.mean((hi_a - lo_a)[m]))
                len_b[k_bin] = float(np.mean((hi_b - lo_b)[m]))
                scr_a[k_bin] = float(np.mean(sc_a[m]))
                scr_b[k_bin] = float(np.mean(sc_b[m]))

            cov_a_runs.append(cov_a)
            cov_b_runs.append(cov_b)
            len_a_runs.append(len_a)
            len_b_runs.append(len_b)
            score_a_runs.append(scr_a)
            score_b_runs.append(scr_b)

        C_a = np.stack(cov_a_runs, axis=0)
        C_b = np.stack(cov_b_runs, axis=0)
        L_a = np.stack(len_a_runs, axis=0)
        L_b = np.stack(len_b_runs, axis=0)
        S_a = np.stack(score_a_runs, axis=0)
        S_b = np.stack(score_b_runs, axis=0)

    G_a = np.abs(C_a - target)
    G_b = np.abs(C_b - target)
    G_a_mean = np.nanmean(G_a, axis=0)
    G_a_sd = np.nanstd(G_a, axis=0)
    G_b_mean = np.nanmean(G_b, axis=0)
    G_b_sd = np.nanstd(G_b, axis=0)

    L_a_mean = np.nanmean(L_a, axis=0)
    L_a_sd = np.nanstd(L_a, axis=0)
    L_b_mean = np.nanmean(L_b, axis=0)
    L_b_sd = np.nanstd(L_b, axis=0)

    S_a_mean = np.nanmean(S_a, axis=0)
    S_a_sd = np.nanstd(S_a, axis=0)
    S_b_mean = np.nanmean(S_b, axis=0)
    S_b_sd = np.nanstd(S_b, axis=0)

    fig1, ax1 = plt.subplots(figsize=(10.5, 3.6))
    ax1.plot(centers, G_a_mean, marker="o", label=f"{label_a}: mean |cov-target|")
    ax1.fill_between(centers, G_a_mean - G_a_sd, G_a_mean + G_a_sd, alpha=0.20)
    ax1.plot(centers, G_b_mean, marker="o", label=f"{label_b}: mean |cov-target|")
    ax1.fill_between(centers, G_b_mean - G_b_sd, G_b_mean + G_b_sd, alpha=0.20)
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel(f"{key} (bin centers)")
    ax1.set_ylabel("|coverage - target|")
    ax1.set_title((title + " - " if title else "") + f"{key}: abs conditional coverage gap (target={target:.2f}), mean±sd over {n_runs} runs")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    fig1.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, f"{base}_bins_gap.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10.5, 3.6))
    ax2.plot(centers, L_a_mean, marker="o", label=f"{label_a}: mean length")
    ax2.fill_between(centers, L_a_mean - L_a_sd, L_a_mean + L_a_sd, alpha=0.20)
    ax2.plot(centers, L_b_mean, marker="o", label=f"{label_b}: mean length")
    ax2.fill_between(centers, L_b_mean - L_b_sd, L_b_mean + L_b_sd, alpha=0.20)
    ax2.set_xlabel(f"{key} (bin centers)")
    ax2.set_ylabel("avg interval length")
    ax2.set_title((title + " - " if title else "") + f"{key}: interval length, mean±sd over {n_runs} runs")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")
    fig2.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig2.savefig(os.path.join(save_dir, f"{base}_bins_length.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    # --- (2b) interval score by bins (mean±sd across runs) ---
    fig2b, ax2b = plt.subplots(figsize=(10.5, 3.6))
    ax2b.plot(centers, S_a_mean, marker="o", label=f"{label_a}: mean interval score")
    ax2b.fill_between(centers, S_a_mean - S_a_sd, S_a_mean + S_a_sd, alpha=0.20)
    ax2b.plot(centers, S_b_mean, marker="o", label=f"{label_b}: mean interval score")
    ax2b.fill_between(centers, S_b_mean - S_b_sd, S_b_mean + S_b_sd, alpha=0.20)
    ax2b.set_xlabel(f"{key} (bin centers)")
    ax2b.set_ylabel("interval score (smaller is better)")
    ax2b.set_title((title + " - " if title else "") + f"{key}: interval score, mean±sd over {n_runs} runs")
    ax2b.grid(True, alpha=0.25)
    ax2b.legend(loc="best")
    fig2b.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig2b.savefig(os.path.join(save_dir, f"{base}_bins_interval_score.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2b)

    # --- (3) abs coverage gap by channel (mean±sd across runs) ---
    ch_gap_a_runs = []
    ch_gap_b_runs = []
    ch_ref = None
    for art in artifacts:
        ch = np.asarray(art[channel_key], dtype=int)
        ca = np.asarray(art[covered_key_a], dtype=bool)
        cb = np.asarray(art[covered_key_b], dtype=bool)
        if not (ch.ndim == ca.ndim == cb.ndim == 1 and ch.size == ca.size == cb.size):
            raise ValueError("channel/covered arrays must be 1D and aligned per artifact")
        J_max = int(np.max(ch)) + 1 if ch.size > 0 else 0
        cov_ch_a = np.full(J_max, np.nan, dtype=float)
        cov_ch_b = np.full(J_max, np.nan, dtype=float)
        for j in range(J_max):
            m = ch == j
            if np.any(m):
                cov_ch_a[j] = float(np.mean(ca[m]))
                cov_ch_b[j] = float(np.mean(cb[m]))
        gap_ch_a = np.abs(cov_ch_a - target)
        gap_ch_b = np.abs(cov_ch_b - target)
        if ch_ref is None:
            ch_ref = J_max
        else:
            ch_ref = min(ch_ref, J_max)
            gap_ch_a = gap_ch_a[:ch_ref]
            gap_ch_b = gap_ch_b[:ch_ref]
        ch_gap_a_runs.append(gap_ch_a)
        ch_gap_b_runs.append(gap_ch_b)

    Gch_a = np.stack(ch_gap_a_runs, axis=0)
    Gch_b = np.stack(ch_gap_b_runs, axis=0)
    xs = np.arange(Gch_a.shape[1])
    m_a = np.nanmean(Gch_a, axis=0)
    s_a = np.nanstd(Gch_a, axis=0)
    m_b = np.nanmean(Gch_b, axis=0)
    s_b = np.nanstd(Gch_b, axis=0)

    fig3, ax3 = plt.subplots(figsize=(10.5, 3.6))
    ax3.errorbar(xs - 0.05, m_a, yerr=s_a, fmt="o-", capsize=3, label=f"{label_a}: mean±sd |cov-target|")
    ax3.errorbar(xs + 0.05, m_b, yerr=s_b, fmt="o-", capsize=3, label=f"{label_b}: mean±sd |cov-target|")
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax3.set_xlabel("channel j")
    ax3.set_ylabel("|coverage - target|")
    ax3.set_title((title + " - " if title else "") + f"abs conditional coverage gap by channel (target={target:.2f}), mean±sd over {n_runs} runs")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")
    fig3.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig3.savefig(os.path.join(save_dir, f"{base}_channel_gap.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig3)

    # --- (4) distribution of per-subject abs coverage gap ---
    subj_gap_a_all = []
    subj_gap_b_all = []
    per_run_mean_a = []
    per_run_mean_b = []
    for art in artifacts:
        sb = np.asarray(art[subject_key], dtype=int)
        ca = np.asarray(art[covered_key_a], dtype=bool)
        cb = np.asarray(art[covered_key_b], dtype=bool)
        I_max = int(np.max(sb)) + 1 if sb.size > 0 else 0
        cov_sb_a = np.full(I_max, np.nan, dtype=float)
        cov_sb_b = np.full(I_max, np.nan, dtype=float)
        for i in range(I_max):
            m = sb == i
            if np.any(m):
                cov_sb_a[i] = float(np.mean(ca[m]))
                cov_sb_b[i] = float(np.mean(cb[m]))
        gap_sb_a = np.abs(cov_sb_a - target)
        gap_sb_b = np.abs(cov_sb_b - target)
        subj_gap_a_all.append(gap_sb_a[np.isfinite(gap_sb_a)])
        subj_gap_b_all.append(gap_sb_b[np.isfinite(gap_sb_b)])
        per_run_mean_a.append(float(np.nanmean(gap_sb_a)))
        per_run_mean_b.append(float(np.nanmean(gap_sb_b)))

    gap_a_pool = np.concatenate(subj_gap_a_all) if subj_gap_a_all else np.array([])
    gap_b_pool = np.concatenate(subj_gap_b_all) if subj_gap_b_all else np.array([])

    fig4, ax4 = plt.subplots(figsize=(10.5, 3.6))
    bins = np.linspace(0.0, 0.5, 31)
    ax4.hist(gap_a_pool, bins=bins, alpha=0.5, label=f"{label_a}")
    ax4.hist(gap_b_pool, bins=bins, alpha=0.5, label=f"{label_b}")
    ax4.set_xlabel("|per-subject coverage - target|")
    ax4.set_ylabel("count (subjects pooled across runs)")
    ax4.set_title((title + " - " if title else "") + "distribution of per-subject abs coverage gap")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best")
    ax4.text(
        0.99,
        0.95,
        f"{label_a}: mean(per-run mean)={np.mean(per_run_mean_a):.3f} ± {np.std(per_run_mean_a):.3f}\n"
        f"{label_b}: mean(per-run mean)={np.mean(per_run_mean_b):.3f} ± {np.std(per_run_mean_b):.3f}",
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig4.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig4.savefig(os.path.join(save_dir, f"{base}_subject_gap_hist.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig4)

    # --- (5) subject×channel abs coverage gap heatmap (mean across runs) ---
    # Build matrices per run then average.
    J_max = int(np.max(np.concatenate([np.asarray(a[channel_key], dtype=int) for a in artifacts])) + 1)
    I_max = int(np.max(np.concatenate([np.asarray(a[subject_key], dtype=int) for a in artifacts])) + 1)

    mats_a = []
    mats_b = []
    for art in artifacts:
        ch = np.asarray(art[channel_key], dtype=int)
        sb = np.asarray(art[subject_key], dtype=int)
        ca = np.asarray(art[covered_key_a], dtype=bool)
        cb = np.asarray(art[covered_key_b], dtype=bool)
        cov_mat_a = np.full((J_max, I_max), np.nan, dtype=float)
        cov_mat_b = np.full((J_max, I_max), np.nan, dtype=float)
        for i in range(I_max):
            mi = sb == i
            if not np.any(mi):
                continue
            for j in range(J_max):
                m = mi & (ch == j)
                if np.any(m):
                    cov_mat_a[j, i] = float(np.mean(ca[m]))
                    cov_mat_b[j, i] = float(np.mean(cb[m]))
        mats_a.append(np.abs(cov_mat_a - target))
        mats_b.append(np.abs(cov_mat_b - target))

    mean_mat_a = np.nanmean(np.stack(mats_a, axis=0), axis=0)
    mean_mat_b = np.nanmean(np.stack(mats_b, axis=0), axis=0)
    vmax = float(np.nanmax(np.concatenate([mean_mat_a[np.isfinite(mean_mat_a)], mean_mat_b[np.isfinite(mean_mat_b)]])))
    vmax = max(vmax, 1e-6)

    heatmap_style_use = str(heatmap_style).lower().strip()
    if heatmap_style_use not in {"imshow", "contourf"}:
        raise ValueError("heatmap_style must be one of {'imshow','contourf'}")
    levels = int(heatmap_levels)
    if levels < 3:
        raise ValueError("heatmap_levels must be >= 3")
    up = int(max(1, heatmap_upsample))

    fig5, axes5 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, I_max * 0.35), max(4.5, J_max * 0.6) * 2),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    if heatmap_style_use == "imshow":
        im0 = axes5[0].imshow(mean_mat_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes5[1].imshow(mean_mat_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            Zp = np.where(np.isfinite(Zp), Zp, 0.0)
            if up > 1:
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(mean_mat_a)
        Zb = prep(mean_mat_b)
        y = np.linspace(0, J_max - 1, Za.shape[0])
        x = np.linspace(0, I_max - 1, Za.shape[1])
        X, Y = np.meshgrid(x, y)
        im0 = axes5[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes5[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)

    axes5[0].set_title(f"{label_a}: mean abs gap |cov-target| by (channel, subject)")
    axes5[1].set_title(f"{label_b}: mean abs gap |cov-target| by (channel, subject)")
    for ax in axes5:
        ax.set_ylabel("channel")
        ax.set_xlabel("subject")
        if I_max <= 30:
            ax.set_xticks(np.arange(I_max))
        if J_max <= 30:
            ax.set_yticks(np.arange(J_max))

    cbar = fig5.colorbar(im1, ax=axes5.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("|coverage - target|")

    if title is not None:
        fig5.suptitle(title + " (subject×channel abs-gap heatmaps)")
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig5.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_channel_gap.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig5)

    # --- (6) timebin×subject abs coverage gap heatmap (mean across runs) ---
    # Use the same fixed, equal-width time bins on [0, T-1] so bins align across runs.
    nb = int(n_time_bins)
    if nb <= 1:
        raise ValueError("n_time_bins must be >= 2")

    # Determine a shared T across artifacts.
    T_ref = int(artifacts[0].get("T", 0))
    if T_ref <= 1:
        raise ValueError("Invalid T in artifacts[0]")
    for art in artifacts[1:]:
        if int(art.get("T", T_ref)) != T_ref:
            raise RuntimeError("All artifacts must share the same T for fixed time binning.")

    edges = np.linspace(0.0, float(T_ref - 1), nb + 1)

    def bin_id(t: np.ndarray) -> np.ndarray:
        # Last edge inclusive.
        return np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, nb - 1)

    I_max = int(np.max(np.concatenate([np.asarray(a[subject_key], dtype=int) for a in artifacts])) + 1)
    mats_a = []
    mats_b = []
    for art in artifacts:
        t = np.asarray(art[t_key], dtype=float).ravel()
        sb = np.asarray(art[subject_key], dtype=int).ravel()
        ca = np.asarray(art[covered_key_a], dtype=bool).ravel()
        cb = np.asarray(art[covered_key_b], dtype=bool).ravel()
        if not (t.size == sb.size == ca.size == cb.size):
            raise ValueError("t/subject/covered arrays must have the same length within each artifact.")

        b = bin_id(t)
        cov_mat_a = np.full((nb, I_max), np.nan, dtype=float)  # rows=timebin, cols=subject
        cov_mat_b = np.full((nb, I_max), np.nan, dtype=float)
        for k_bin in range(nb):
            mk = b == k_bin
            if not np.any(mk):
                continue
            for i in range(I_max):
                m = mk & (sb == i)
                if np.any(m):
                    cov_mat_a[k_bin, i] = float(np.mean(ca[m]))
                    cov_mat_b[k_bin, i] = float(np.mean(cb[m]))

        mats_a.append(np.abs(cov_mat_a - target))
        mats_b.append(np.abs(cov_mat_b - target))

    mean_mat_a = np.nanmean(np.stack(mats_a, axis=0), axis=0)
    mean_mat_b = np.nanmean(np.stack(mats_b, axis=0), axis=0)
    vmax = float(
        np.nanmax(
            np.concatenate(
                [mean_mat_a[np.isfinite(mean_mat_a)], mean_mat_b[np.isfinite(mean_mat_b)]]
            )
        )
    )
    vmax = max(vmax, 1e-6)

    heatmap_style_use = str(heatmap_style).lower().strip()
    if heatmap_style_use not in {"imshow", "contourf"}:
        raise ValueError("heatmap_style must be one of {'imshow','contourf'}")
    levels = int(heatmap_levels)
    if levels < 3:
        raise ValueError("heatmap_levels must be >= 3")
    up = int(max(1, heatmap_upsample))

    fig6, axes6 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, nb * 0.7), max(4.5, I_max * 0.25) * 2),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    if heatmap_style_use == "imshow":
        im0 = axes6[0].imshow(mean_mat_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes6[1].imshow(mean_mat_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            Zp = np.where(np.isfinite(Zp), Zp, 0.0)
            if up > 1:
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(mean_mat_a)
        Zb = prep(mean_mat_b)
        y = np.linspace(0, nb - 1, Za.shape[0])
        x = np.linspace(0, I_max - 1, Za.shape[1])
        X, Y = np.meshgrid(x, y)
        im0 = axes6[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes6[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)

    axes6[0].set_title(f"{label_a}: mean abs gap |cov-target| by (timebin, subject)")
    axes6[1].set_title(f"{label_b}: mean abs gap |cov-target| by (timebin, subject)")
    for ax in axes6:
        ax.set_ylabel("time bin")
        ax.set_xlabel("subject")
        if I_max <= 30:
            ax.set_xticks(np.arange(I_max))
        if nb <= 30:
            ax.set_yticks(np.arange(nb))

    cbar = fig6.colorbar(im1, ax=axes6.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("|coverage - target|")

    if title is not None:
        fig6.suptitle(title + " (timebin×subject abs-gap heatmaps)")
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig6.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_timebin_gap.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig6)

    # --- (7) subject×channel mean interval length heatmap (mean across runs) ---
    # Here we use the axis order requested: rows=subject, cols=channel.
    #
    # Requested transform relative to ORACLE length:
    #   if len > len_oracle: show len/len_oracle
    #   else:               show oracle_shorter_coef * (len_oracle/len)
    if oracle_shorter_coef <= 0:
        raise ValueError("oracle_shorter_coef must be > 0")
    if oracle_lower_key not in artifacts[0] or oracle_upper_key not in artifacts[0]:
        raise KeyError(
            "Oracle-relative length heatmaps require oracle bounds in artifacts. "
            f"Missing '{oracle_lower_key}'/'{oracle_upper_key}' in artifacts[0]."
        )
    J_max = int(np.max(np.concatenate([np.asarray(a[channel_key], dtype=int) for a in artifacts])) + 1)
    I_max = int(np.max(np.concatenate([np.asarray(a[subject_key], dtype=int) for a in artifacts])) + 1)

    mats_a = []
    mats_b = []
    for art in artifacts:
        ch = np.asarray(art[channel_key], dtype=int)
        sb = np.asarray(art[subject_key], dtype=int)
        lo_a = np.asarray(art[lower_key_a], dtype=float)
        hi_a = np.asarray(art[upper_key_a], dtype=float)
        lo_b = np.asarray(art[lower_key_b], dtype=float)
        hi_b = np.asarray(art[upper_key_b], dtype=float)
        lo_o = np.asarray(art[oracle_lower_key], dtype=float)
        hi_o = np.asarray(art[oracle_upper_key], dtype=float)
        if not (ch.ndim == sb.ndim == lo_a.ndim == hi_a.ndim == lo_b.ndim == hi_b.ndim == 1):
            raise ValueError("channel/subject/lower/upper arrays must be 1D per artifact")
        if not (ch.size == sb.size == lo_a.size == hi_a.size == lo_b.size == hi_b.size == lo_o.size == hi_o.size):
            raise ValueError("channel/subject/lower/upper arrays must be aligned per artifact")

        len_mat_a = np.full((I_max, J_max), np.nan, dtype=float)  # rows=subject, cols=channel
        len_mat_b = np.full((I_max, J_max), np.nan, dtype=float)
        len_a = hi_a - lo_a
        len_b = hi_b - lo_b
        len_o = hi_o - lo_o
        for i in range(I_max):
            mi = sb == i
            if not np.any(mi):
                continue
            for j in range(J_max):
                m = mi & (ch == j)
                if np.any(m):
                    la = float(np.mean(len_a[m]))
                    lb = float(np.mean(len_b[m]))
                    lo = float(np.mean(len_o[m]))
                    # Oracle-relative transform (always >= 1 when oracle_shorter_coef >= 1)
                    if np.isfinite(la) and np.isfinite(lo) and la > 0 and lo > 0:
                        len_mat_a[i, j] = (la / lo) if (la >= lo) else (oracle_shorter_coef * (lo / la))
                    if np.isfinite(lb) and np.isfinite(lo) and lb > 0 and lo > 0:
                        len_mat_b[i, j] = (lb / lo) if (lb >= lo) else (oracle_shorter_coef * (lo / lb))
        mats_a.append(len_mat_a)
        mats_b.append(len_mat_b)

    mean_len_a = np.nanmean(np.stack(mats_a, axis=0), axis=0)
    mean_len_b = np.nanmean(np.stack(mats_b, axis=0), axis=0)
    vmax = float(
        np.nanmax(
            np.concatenate([mean_len_a[np.isfinite(mean_len_a)], mean_len_b[np.isfinite(mean_len_b)]])
        )
    )
    vmax = max(vmax, 1e-6)

    fig7, axes7 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, J_max * 0.6), max(4.5, I_max * 0.25) * 2),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=1.0, vmax=vmax)

    if heatmap_style_use == "imshow":
        im0 = axes7[0].imshow(mean_len_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes7[1].imshow(mean_len_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            Zp = np.where(np.isfinite(Zp), Zp, 0.0)
            if up > 1:
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(mean_len_a)
        Zb = prep(mean_len_b)
        y = np.linspace(0, I_max - 1, Za.shape[0])
        x = np.linspace(0, J_max - 1, Za.shape[1])
        X, Y = np.meshgrid(x, y)
        im0 = axes7[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes7[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)

    axes7[0].set_title(f"{label_a}: oracle-relative length ratio by (subject, channel)")
    axes7[1].set_title(f"{label_b}: oracle-relative length ratio by (subject, channel)")
    for ax in axes7:
        ax.set_ylabel("subject")
        ax.set_xlabel("channel")
        if J_max <= 30:
            ax.set_xticks(np.arange(J_max))
        if I_max <= 30:
            ax.set_yticks(np.arange(I_max))

    cbar = fig7.colorbar(im1, ax=axes7.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("oracle-relative length ratio (>=1 if oracle_shorter_coef>=1)")

    if title is not None:
        fig7.suptitle(title + " (subject×channel oracle-relative length heatmaps)")
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig7.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_channel_length.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig7)

    # --- (8) timebin×subject mean interval length heatmap (mean across runs) ---
    # Use the same fixed time bins on [0, T-1] as in (6), and keep axes as rows=timebin, cols=subject.
    nb = int(n_time_bins)
    edges = np.linspace(0.0, float(T_ref - 1), nb + 1)

    def bin_id(t: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, nb - 1)

    I_max = int(np.max(np.concatenate([np.asarray(a[subject_key], dtype=int) for a in artifacts])) + 1)
    mats_a = []
    mats_b = []
    for art in artifacts:
        t = np.asarray(art[t_key], dtype=float).ravel()
        sb = np.asarray(art[subject_key], dtype=int).ravel()
        lo_a = np.asarray(art[lower_key_a], dtype=float).ravel()
        hi_a = np.asarray(art[upper_key_a], dtype=float).ravel()
        lo_b = np.asarray(art[lower_key_b], dtype=float).ravel()
        hi_b = np.asarray(art[upper_key_b], dtype=float).ravel()
        lo_o = np.asarray(art[oracle_lower_key], dtype=float).ravel()
        hi_o = np.asarray(art[oracle_upper_key], dtype=float).ravel()
        if not (t.size == sb.size == lo_a.size == hi_a.size == lo_b.size == hi_b.size == lo_o.size == hi_o.size):
            raise ValueError("t/subject/lower/upper arrays must have the same length within each artifact.")

        b = bin_id(t)
        len_mat_a = np.full((nb, I_max), np.nan, dtype=float)  # rows=timebin, cols=subject
        len_mat_b = np.full((nb, I_max), np.nan, dtype=float)
        len_a = hi_a - lo_a
        len_b = hi_b - lo_b
        len_o = hi_o - lo_o
        for k_bin in range(nb):
            mk = b == k_bin
            if not np.any(mk):
                continue
            for i in range(I_max):
                m = mk & (sb == i)
                if np.any(m):
                    la = float(np.mean(len_a[m]))
                    lb = float(np.mean(len_b[m]))
                    lo = float(np.mean(len_o[m]))
                    if np.isfinite(la) and np.isfinite(lo) and la > 0 and lo > 0:
                        len_mat_a[k_bin, i] = (la / lo) if (la >= lo) else (oracle_shorter_coef * (lo / la))
                    if np.isfinite(lb) and np.isfinite(lo) and lb > 0 and lo > 0:
                        len_mat_b[k_bin, i] = (lb / lo) if (lb >= lo) else (oracle_shorter_coef * (lo / lb))

        mats_a.append(len_mat_a)
        mats_b.append(len_mat_b)

    mean_len_a = np.nanmean(np.stack(mats_a, axis=0), axis=0)
    mean_len_b = np.nanmean(np.stack(mats_b, axis=0), axis=0)
    vmax = float(
        np.nanmax(
            np.concatenate([mean_len_a[np.isfinite(mean_len_a)], mean_len_b[np.isfinite(mean_len_b)]])
        )
    )
    vmax = max(vmax, 1e-6)

    fig8, axes8 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, nb * 0.7), max(4.5, I_max * 0.25) * 2),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=1.0, vmax=vmax)

    if heatmap_style_use == "imshow":
        im0 = axes8[0].imshow(mean_len_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes8[1].imshow(mean_len_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            Zp = np.where(np.isfinite(Zp), Zp, 0.0)
            if up > 1:
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(mean_len_a)
        Zb = prep(mean_len_b)
        y = np.linspace(0, nb - 1, Za.shape[0])
        x = np.linspace(0, I_max - 1, Za.shape[1])
        X, Y = np.meshgrid(x, y)
        im0 = axes8[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes8[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)

    axes8[0].set_title(f"{label_a}: oracle-relative length ratio by (timebin, subject)")
    axes8[1].set_title(f"{label_b}: oracle-relative length ratio by (timebin, subject)")
    for ax in axes8:
        ax.set_ylabel("time bin")
        ax.set_xlabel("subject")
        if I_max <= 30:
            ax.set_xticks(np.arange(I_max))
        if nb <= 30:
            ax.set_yticks(np.arange(nb))

    cbar = fig8.colorbar(im1, ax=axes8.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("oracle-relative length ratio (>=1 if oracle_shorter_coef>=1)")

    if title is not None:
        fig8.suptitle(title + " (timebin×subject oracle-relative length heatmaps)")
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig8.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_timebin_length.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig8)

    # --- (9) subject×channel mean interval score heatmap (mean across runs) ---
    J_max = int(np.max(np.concatenate([np.asarray(a[channel_key], dtype=int) for a in artifacts])) + 1)
    I_max = int(np.max(np.concatenate([np.asarray(a[subject_key], dtype=int) for a in artifacts])) + 1)

    mats_a = []
    mats_b = []
    for art in artifacts:
        ch = np.asarray(art[channel_key], dtype=int)
        sb = np.asarray(art[subject_key], dtype=int)
        y = np.asarray(art["Y_test_true"], dtype=float).ravel()
        lo_a = np.asarray(art[lower_key_a], dtype=float).ravel()
        hi_a = np.asarray(art[upper_key_a], dtype=float).ravel()
        lo_b = np.asarray(art[lower_key_b], dtype=float).ravel()
        hi_b = np.asarray(art[upper_key_b], dtype=float).ravel()
        if not (ch.ndim == sb.ndim == 1):
            raise ValueError("channel/subject arrays must be 1D per artifact")
        if not (ch.size == sb.size == y.size == lo_a.size == hi_a.size == lo_b.size == hi_b.size):
            raise ValueError("Per-run test arrays must be aligned for interval-score heatmaps.")

        sc_a = interval_score(y, lo_a, hi_a)
        sc_b = interval_score(y, lo_b, hi_b)
        sc_mat_a = np.full((I_max, J_max), np.nan, dtype=float)  # rows=subject, cols=channel
        sc_mat_b = np.full((I_max, J_max), np.nan, dtype=float)
        for i in range(I_max):
            mi = sb == i
            if not np.any(mi):
                continue
            for j in range(J_max):
                m = mi & (ch == j)
                if np.any(m):
                    sc_mat_a[i, j] = float(np.mean(sc_a[m]))
                    sc_mat_b[i, j] = float(np.mean(sc_b[m]))
        mats_a.append(sc_mat_a)
        mats_b.append(sc_mat_b)

    mean_sc_a = np.nanmean(np.stack(mats_a, axis=0), axis=0)
    mean_sc_b = np.nanmean(np.stack(mats_b, axis=0), axis=0)
    vmax = float(
        np.nanmax(np.concatenate([mean_sc_a[np.isfinite(mean_sc_a)], mean_sc_b[np.isfinite(mean_sc_b)]]))
    )
    vmax = max(vmax, 1e-6)

    fig9, axes9 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, J_max * 0.6), max(4.5, I_max * 0.25) * 2),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Purples")
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    if heatmap_style_use == "imshow":
        im0 = axes9[0].imshow(mean_sc_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes9[1].imshow(mean_sc_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            Zp = np.where(np.isfinite(Zp), Zp, 0.0)
            if up > 1:
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(mean_sc_a)
        Zb = prep(mean_sc_b)
        yv = np.linspace(0, I_max - 1, Za.shape[0])
        xv = np.linspace(0, J_max - 1, Za.shape[1])
        X, Y = np.meshgrid(xv, yv)
        im0 = axes9[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes9[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)

    axes9[0].set_title(f"{label_a}: mean interval score by (subject, channel)")
    axes9[1].set_title(f"{label_b}: mean interval score by (subject, channel)")
    for ax in axes9:
        ax.set_ylabel("subject")
        ax.set_xlabel("channel")
        if J_max <= 30:
            ax.set_xticks(np.arange(J_max))
        if I_max <= 30:
            ax.set_yticks(np.arange(I_max))
    cbar = fig9.colorbar(im1, ax=axes9.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("interval score (smaller is better)")
    if title is not None:
        fig9.suptitle(title + " (subject×channel interval-score heatmaps)")
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig9.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_channel_interval_score.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig9)

    # --- (10) timebin×subject mean interval score heatmap (mean across runs) ---
    nb = int(n_time_bins)
    edges_sc = np.linspace(0.0, float(T_ref - 1), nb + 1)

    def bin_id_sc(t: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(edges_sc[1:-1], t, side="right"), 0, nb - 1)

    I_max = int(np.max(np.concatenate([np.asarray(a[subject_key], dtype=int) for a in artifacts])) + 1)
    mats_a = []
    mats_b = []
    for art in artifacts:
        t = np.asarray(art[t_key], dtype=float).ravel()
        sb = np.asarray(art[subject_key], dtype=int).ravel()
        y = np.asarray(art["Y_test_true"], dtype=float).ravel()
        lo_a = np.asarray(art[lower_key_a], dtype=float).ravel()
        hi_a = np.asarray(art[upper_key_a], dtype=float).ravel()
        lo_b = np.asarray(art[lower_key_b], dtype=float).ravel()
        hi_b = np.asarray(art[upper_key_b], dtype=float).ravel()
        if not (t.size == sb.size == y.size == lo_a.size == hi_a.size == lo_b.size == hi_b.size):
            raise ValueError("Per-run test arrays must have the same length for interval-score timebin heatmap.")

        sc_a = interval_score(y, lo_a, hi_a)
        sc_b = interval_score(y, lo_b, hi_b)
        b = bin_id_sc(t)
        sc_mat_a = np.full((nb, I_max), np.nan, dtype=float)  # rows=timebin, cols=subject
        sc_mat_b = np.full((nb, I_max), np.nan, dtype=float)
        for k_bin in range(nb):
            mk = b == k_bin
            if not np.any(mk):
                continue
            for i in range(I_max):
                m = mk & (sb == i)
                if np.any(m):
                    sc_mat_a[k_bin, i] = float(np.mean(sc_a[m]))
                    sc_mat_b[k_bin, i] = float(np.mean(sc_b[m]))
        mats_a.append(sc_mat_a)
        mats_b.append(sc_mat_b)

    mean_sc_a = np.nanmean(np.stack(mats_a, axis=0), axis=0)
    mean_sc_b = np.nanmean(np.stack(mats_b, axis=0), axis=0)
    vmax = float(
        np.nanmax(np.concatenate([mean_sc_a[np.isfinite(mean_sc_a)], mean_sc_b[np.isfinite(mean_sc_b)]]))
    )
    vmax = max(vmax, 1e-6)

    fig10, axes10 = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, nb * 0.7), max(4.5, I_max * 0.25) * 2),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Purples")
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    if heatmap_style_use == "imshow":
        im0 = axes10[0].imshow(mean_sc_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes10[1].imshow(mean_sc_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "heatmap_style='contourf' requires SciPy. Install it (e.g. `pip install scipy`) "
                "or use heatmap_style='imshow'."
            ) from e

        def prep(Z: np.ndarray) -> np.ndarray:
            Zp = np.asarray(Z, dtype=float)
            Zp = np.where(np.isfinite(Zp), Zp, 0.0)
            if up > 1:
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(mean_sc_a)
        Zb = prep(mean_sc_b)
        yv = np.linspace(0, nb - 1, Za.shape[0])
        xv = np.linspace(0, I_max - 1, Za.shape[1])
        X, Y = np.meshgrid(xv, yv)
        im0 = axes10[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes10[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)

    axes10[0].set_title(f"{label_a}: mean interval score by (timebin, subject)")
    axes10[1].set_title(f"{label_b}: mean interval score by (timebin, subject)")
    for ax in axes10:
        ax.set_ylabel("time bin")
        ax.set_xlabel("subject")
        if I_max <= 30:
            ax.set_xticks(np.arange(I_max))
        if nb <= 30:
            ax.set_yticks(np.arange(nb))
    cbar = fig10.colorbar(im1, ax=axes10.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("interval score (smaller is better)")
    if title is not None:
        fig10.suptitle(title + " (timebin×subject interval-score heatmaps)")
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig10.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_timebin_interval_score.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig10)


def plot_methods_marginal_coverage_and_interval_score(
    artifacts: list[dict],
    *,
    # If not provided, infer methods from keys `covered_*` that have matching `lower_*`/`upper_*`.
    method_specs: Optional[Sequence[Dict[str, str]]] = None,
    method_order: Optional[Sequence[str]] = None,
    method_labels: Optional[Dict[str, str]] = None,
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
    # Scoring uses alpha; if None, uses per-artifact `art["alpha"]` (falls back to 0.1).
    alpha_target: Optional[float] = None,
    # Plot options
    title: Optional[str] = None,
    show: bool = True,
    save_plots: bool = False,
    save_dir: str = "result_img",
    save_basename: Optional[str] = None,
    dpi: int = 200,
    figsize: Tuple[float, float] = (11.5, 4.2),
):
    """
    Compare *all* methods in `artifacts` by:
      - marginal coverage (mean±sd across runs)
      - mean interval score (mean±sd across runs; smaller is better)

    Produces one plot with dual y-axes:
      - left y-axis: coverage
      - right y-axis: interval score

    Method discovery (default):
      For each key `covered_<name>` in artifacts[0], include it as a method iff
      `lower_<name>` and `upper_<name>` also exist (needed for interval score).
    """
    if len(artifacts) == 0:
        raise ValueError("artifacts list is empty")

    def _sanitize(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
        return s[:120] if len(s) > 120 else s

    def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        lo = np.asarray(lo, dtype=float).ravel()
        hi = np.asarray(hi, dtype=float).ravel()
        if not (y.size == lo.size == hi.size):
            raise ValueError("interval_score: y, lo, hi must have the same length")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("interval_score: alpha must be in (0,1)")
        width = hi - lo
        below = y < lo
        above = y > hi
        pen = np.zeros_like(width, dtype=float)
        if np.any(below):
            pen[below] = (2.0 / float(alpha)) * (lo[below] - y[below])
        if np.any(above):
            pen[above] = (2.0 / float(alpha)) * (y[above] - hi[above])
        return width + pen

    # Build method specs.
    if method_specs is None:
        specs: List[Dict[str, str]] = []
        a0 = artifacts[0]
        for k in sorted(a0.keys()):
            if not (isinstance(k, str) and k.startswith("covered_")):
                continue
            name = k.split("covered_", 1)[-1]
            lo_k = f"lower_{name}"
            hi_k = f"upper_{name}"
            if lo_k in a0 and hi_k in a0:
                specs.append({"name": name, "covered_key": k, "lower_key": lo_k, "upper_key": hi_k})
        method_specs = specs

    # Apply include/exclude filters.
    specs = list(method_specs)
    if include_methods is not None:
        include_set = set(map(str, include_methods))
        specs = [s for s in specs if str(s["name"]) in include_set]
    if exclude_methods is not None:
        exclude_set = set(map(str, exclude_methods))
        specs = [s for s in specs if str(s["name"]) not in exclude_set]

    if len(specs) == 0:
        raise ValueError("No methods to plot (after inference/filters).")

    # Apply order (by method name).
    if method_order is not None:
        order = [str(x) for x in method_order]
        pos = {name: i for i, name in enumerate(order)}
        specs = sorted(specs, key=lambda s: pos.get(str(s["name"]), 10**9))

    names = [str(s["name"]) for s in specs]
    labels = [(method_labels.get(n, n) if method_labels else n) for n in names]

    # Per-run aggregates
    cov_runs: Dict[str, List[float]] = {n: [] for n in names}
    score_runs: Dict[str, List[float]] = {n: [] for n in names}

    for art in artifacts:
        y = np.asarray(art.get("Y_test_true", []), dtype=float).ravel()
        alpha_use = float(alpha_target) if alpha_target is not None else float(art.get("alpha", 0.1))
        for s in specs:
            n = str(s["name"])
            covered = np.asarray(art.get(s["covered_key"], []), dtype=bool).ravel()
            lo = np.asarray(art.get(s["lower_key"], []), dtype=float).ravel()
            hi = np.asarray(art.get(s["upper_key"], []), dtype=float).ravel()
            if not (y.size == covered.size == lo.size == hi.size):
                raise ValueError(
                    f"Artifact arrays misaligned for method '{n}': "
                    f"Y_test_true={y.size}, covered={covered.size}, lower={lo.size}, upper={hi.size}"
                )
            cov_runs[n].append(float(np.mean(covered)) if covered.size else float("nan"))
            if y.size:
                sc = interval_score(y, lo, hi, alpha=alpha_use)
                score_runs[n].append(float(np.mean(sc)))
            else:
                score_runs[n].append(float("nan"))

    cov_mean = np.array([np.nanmean(cov_runs[n]) for n in names], dtype=float)
    cov_sd = np.array([np.nanstd(cov_runs[n]) for n in names], dtype=float)
    sc_mean = np.array([np.nanmean(score_runs[n]) for n in names], dtype=float)
    sc_sd = np.array([np.nanstd(score_runs[n]) for n in names], dtype=float)

    x = np.arange(len(names), dtype=float)
    fig, ax_cov = plt.subplots(figsize=figsize)
    ax_sc = ax_cov.twinx()

    # Slight horizontal offsets so points don't overlap.
    ax_cov.errorbar(x - 0.06, cov_mean, yerr=cov_sd, fmt="o", capsize=3, color="C0", label="coverage (mean±sd)")
    ax_sc.errorbar(
        x + 0.06,
        sc_mean,
        yerr=sc_sd,
        fmt="s",
        capsize=3,
        color="C3",
        label="interval score (mean±sd)",
    )

    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(labels, rotation=30, ha="right")
    ax_cov.set_xlabel("method")
    ax_cov.set_ylabel("marginal coverage")
    ax_sc.set_ylabel("interval score (smaller is better)")

    # Target line (if alpha is available)
    alpha_line = float(alpha_target) if alpha_target is not None else float(artifacts[0].get("alpha", 0.1))
    if 0.0 < alpha_line < 1.0:
        ax_cov.axhline(1.0 - alpha_line, color="black", linestyle="--", linewidth=1, alpha=0.6)

    if title is not None:
        ax_cov.set_title(title)
    else:
        ax_cov.set_title("Marginal coverage and interval score (mean±sd across runs)")

    ax_cov.grid(True, axis="y", alpha=0.25)

    # Combined legend
    h1, l1 = ax_cov.get_legend_handles_labels()
    h2, l2 = ax_sc.get_legend_handles_labels()
    ax_cov.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()

    base = save_basename or (title if title is not None else "methods_marginal_summary")
    base = _sanitize(base)
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{base}_coverage_and_interval_score.png"), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    summary = {
        "methods": names,
        "labels": labels,
        "coverage_mean": cov_mean,
        "coverage_sd": cov_sd,
        "interval_score_mean": sc_mean,
        "interval_score_sd": sc_sd,
        "n_runs": len(artifacts),
    }
    return fig, (ax_cov, ax_sc), summary


def build_methods_macro_table_from_artifacts(
    artifacts: list[dict],
    *,
    # Method discovery / selection (same convention as plot_methods_marginal_coverage_and_interval_score)
    method_specs: Optional[Sequence[Dict[str, str]]] = None,
    method_order: Optional[Sequence[str]] = None,
    method_labels: Optional[Dict[str, str]] = None,
    include_methods: Optional[Sequence[str]] = None,
    exclude_methods: Optional[Sequence[str]] = None,
    # Partition config
    n_time_bins: int = 10,
    min_group_size: int = 20,
    # Coverage/score config
    alpha_target: Optional[float] = None,
    coverage_tol: float = 0.005,
    panel_a_mode: str = "coverage",  # "coverage" | "coverage_gap"
    # Output formatting
    digits_cov: int = 3,
    digits_score: int = 2,
    latex: bool = True,
    latex_caption: Optional[str] = None,
    latex_label: str = "tab:main_comparison",
):
    """
    Build the "main table" described in the prompt from `artifacts`.

    Two stacked panels:
      - Panel A: macro-averaged coverage (and Δ) or coverage gap (|coverage-target|), per panel_a_mode
      - Panel B: macro-averaged interval score (lower is better) with mean±sd over runs

    Rows (partition types):
      1) Overall (no conditioning)
      2) Time bins
      3) Channel
      4) Subject
      5) Time × Subject
      6) Channel × Subject

    Columns = methods inferred from artifacts.
    """
    if len(artifacts) == 0:
        raise ValueError("artifacts list is empty")
    if method_labels is None or len(method_labels) == 0:
        raise ValueError("method_labels must be provided and non-empty; it defines which methods to include/display.")
    if int(n_time_bins) < 2:
        raise ValueError("n_time_bins must be >= 2")
    if int(min_group_size) < 1:
        raise ValueError("min_group_size must be >= 1")
    panel_a_mode_use = str(panel_a_mode).strip().lower()
    if panel_a_mode_use not in ("coverage", "coverage_gap"):
        raise ValueError("panel_a_mode must be 'coverage' or 'coverage_gap'")

    def _sanitize(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
        return s[:120] if len(s) > 120 else s

    def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        lo = np.asarray(lo, dtype=float).ravel()
        hi = np.asarray(hi, dtype=float).ravel()
        if not (y.size == lo.size == hi.size):
            raise ValueError("interval_score: y, lo, hi must have the same length")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("interval_score: alpha must be in (0,1)")
        width = hi - lo
        below = y < lo
        above = y > hi
        pen = np.zeros_like(width, dtype=float)
        if np.any(below):
            pen[below] = (2.0 / float(alpha)) * (lo[below] - y[below])
        if np.any(above):
            pen[above] = (2.0 / float(alpha)) * (y[above] - hi[above])
        return width + pen

    def macro_mean_by_group(values: np.ndarray, group: np.ndarray, *, min_n: int) -> float:
        v = np.asarray(values, dtype=float).ravel()
        g = np.asarray(group, dtype=int).ravel()
        if v.size != g.size:
            raise ValueError("macro_mean_by_group: values and group must have same length")
        if v.size == 0:
            return float("nan")
        K = int(np.max(g)) + 1 if g.size else 0
        if K <= 0:
            return float("nan")
        cnt = np.bincount(g, minlength=K).astype(float)
        s = np.bincount(g, weights=v, minlength=K).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_g = s / cnt
        keep = (cnt >= float(min_n)) & np.isfinite(mean_g)
        if not np.any(keep):
            return float("nan")
        return float(np.mean(mean_g[keep]))

    def macro_mean_gap_by_group(
        covered: np.ndarray, group: np.ndarray, *, target: float, min_n: int
    ) -> float:
        """Macro-avg of |per-group coverage - target| over groups with n>=min_n."""
        cov = np.asarray(covered, dtype=float).ravel()
        g = np.asarray(group, dtype=int).ravel()
        if cov.size != g.size:
            raise ValueError("macro_mean_gap_by_group: covered and group must have same length")
        if cov.size == 0:
            return float("nan")
        K = int(np.max(g)) + 1 if g.size else 0
        if K <= 0:
            return float("nan")
        cnt = np.bincount(g, minlength=K).astype(float)
        s = np.bincount(g, weights=cov, minlength=K).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov_g = s / cnt
        gap_g = np.abs(cov_g - float(target))
        keep = (cnt >= float(min_n)) & np.isfinite(gap_g)
        if not np.any(keep):
            return float("nan")
        return float(np.mean(gap_g[keep]))

    # --- infer methods ---
    if method_specs is None:
        specs: List[Dict[str, str]] = []
        a0 = artifacts[0]
        for k in sorted(a0.keys()):
            if not (isinstance(k, str) and k.startswith("covered_")):
                continue
            name = k.split("covered_", 1)[-1]
            lo_k = f"lower_{name}"
            hi_k = f"upper_{name}"
            if lo_k in a0 and hi_k in a0:
                specs.append({"name": name, "covered_key": k, "lower_key": lo_k, "upper_key": hi_k})
        method_specs = specs

    specs = list(method_specs)
    # Only include methods explicitly listed in method_labels.
    allowed = set(map(str, method_labels.keys()))
    specs = [s for s in specs if str(s["name"]) in allowed]
    if include_methods is not None:
        include_set = set(map(str, include_methods))
        specs = [s for s in specs if str(s["name"]) in include_set]
    if exclude_methods is not None:
        exclude_set = set(map(str, exclude_methods))
        specs = [s for s in specs if str(s["name"]) not in exclude_set]
    if len(specs) == 0:
        raise ValueError("No methods to tabulate (after filtering by method_labels/include/exclude).")
    if method_order is not None:
        order = [str(x) for x in method_order]
        pos = {name: i for i, name in enumerate(order)}
        specs = sorted(specs, key=lambda s: pos.get(str(s["name"]), 10**9))

    methods = [str(s["name"]) for s in specs]
    labels = [str(method_labels[m]) for m in methods]

    # Rows (partitions)
    row_names = [
        "Overall",
        "Time bins",
        "Channel",
        "Subject",
        "Time × Subject",
        "Channel × Subject",
    ]

    # Per-run macro aggregates: dict[row][method] -> list[float] over runs
    cov_runs: Dict[str, Dict[str, List[float]]] = {rn: {m: [] for m in methods} for rn in row_names}
    gap_runs: Dict[str, Dict[str, List[float]]] = {rn: {m: [] for m in methods} for rn in row_names}
    sc_runs: Dict[str, Dict[str, List[float]]] = {rn: {m: [] for m in methods} for rn in row_names}

    for art in artifacts:
        y = np.asarray(art.get("Y_test_true", []), dtype=float).ravel()
        t = np.asarray(art.get("t_test", []), dtype=int).ravel()
        subj = np.asarray(art.get("subj_test", []), dtype=int).ravel()
        chan = np.asarray(art.get("chan_test", []), dtype=int).ravel()

        if not (y.size == t.size == subj.size == chan.size):
            raise ValueError("Artifact test arrays misaligned: Y_test_true/t_test/subj_test/chan_test must match length.")

        alpha_use = float(alpha_target) if alpha_target is not None else float(art.get("alpha", 0.1))
        target = 1.0 - float(alpha_use)

        # Time bins group id for this run
        T_ref = int(art.get("T", int(np.max(t) + 1 if t.size else 0)))
        nb = int(n_time_bins)
        edges = np.linspace(0.0, float(max(T_ref - 1, 0)), nb + 1) if nb >= 2 else np.array([0.0, 1.0])

        def bin_id(tt: np.ndarray) -> np.ndarray:
            return np.clip(np.searchsorted(edges[1:-1], tt.astype(float), side="right"), 0, nb - 1)

        tb = bin_id(t) if t.size else np.asarray([], dtype=int)

        I_max = int(np.max(subj)) + 1 if subj.size else 0
        J_max = int(np.max(chan)) + 1 if chan.size else 0
        g_time_subject = (tb * max(I_max, 1) + subj) if subj.size else np.asarray([], dtype=int)
        g_chan_subject = (chan * max(I_max, 1) + subj) if subj.size else np.asarray([], dtype=int)

        for s in specs:
            m = str(s["name"])
            covered = np.asarray(art.get(s["covered_key"], []), dtype=bool).ravel()
            lo = np.asarray(art.get(s["lower_key"], []), dtype=float).ravel()
            hi = np.asarray(art.get(s["upper_key"], []), dtype=float).ravel()
            if not (y.size == covered.size == lo.size == hi.size):
                raise ValueError(f"Artifact arrays misaligned for method '{m}'.")

            cov_point = covered.astype(float)
            sc_point = interval_score(y, lo, hi, alpha=alpha_use) if y.size else np.asarray([], dtype=float)

            # Overall (macro==micro)
            cov_runs["Overall"][m].append(float(np.mean(cov_point)) if cov_point.size else float("nan"))
            cov_mean_overall = float(np.mean(cov_point)) if cov_point.size else float("nan")
            gap_runs["Overall"][m].append(float(np.abs(cov_mean_overall - target)) if np.isfinite(cov_mean_overall) else float("nan"))
            sc_runs["Overall"][m].append(float(np.mean(sc_point)) if sc_point.size else float("nan"))

            # Macro across groups
            cov_runs["Time bins"][m].append(macro_mean_by_group(cov_point, tb, min_n=min_group_size))
            gap_runs["Time bins"][m].append(macro_mean_gap_by_group(cov_point, tb, target=target, min_n=min_group_size))
            sc_runs["Time bins"][m].append(macro_mean_by_group(sc_point, tb, min_n=min_group_size))

            cov_runs["Channel"][m].append(macro_mean_by_group(cov_point, chan, min_n=min_group_size))
            gap_runs["Channel"][m].append(macro_mean_gap_by_group(cov_point, chan, target=target, min_n=min_group_size))
            sc_runs["Channel"][m].append(macro_mean_by_group(sc_point, chan, min_n=min_group_size))

            cov_runs["Subject"][m].append(macro_mean_by_group(cov_point, subj, min_n=min_group_size))
            gap_runs["Subject"][m].append(macro_mean_gap_by_group(cov_point, subj, target=target, min_n=min_group_size))
            sc_runs["Subject"][m].append(macro_mean_by_group(sc_point, subj, min_n=min_group_size))

            cov_runs["Time × Subject"][m].append(macro_mean_by_group(cov_point, g_time_subject, min_n=min_group_size))
            gap_runs["Time × Subject"][m].append(macro_mean_gap_by_group(cov_point, g_time_subject, target=target, min_n=min_group_size))
            sc_runs["Time × Subject"][m].append(macro_mean_by_group(sc_point, g_time_subject, min_n=min_group_size))

            cov_runs["Channel × Subject"][m].append(macro_mean_by_group(cov_point, g_chan_subject, min_n=min_group_size))
            gap_runs["Channel × Subject"][m].append(macro_mean_gap_by_group(cov_point, g_chan_subject, target=target, min_n=min_group_size))
            sc_runs["Channel × Subject"][m].append(macro_mean_by_group(sc_point, g_chan_subject, min_n=min_group_size))

    # Aggregate across runs
    def mean_sd(xs: Iterable[float]) -> tuple[float, float]:
        a = np.asarray(list(xs), dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    alpha_line = float(alpha_target) if alpha_target is not None else float(artifacts[0].get("alpha", 0.1))
    target_line = 1.0 - float(alpha_line) if (0.0 < alpha_line < 1.0) else float("nan")

    cov_stats: Dict[str, Dict[str, Dict[str, float]]] = {rn: {} for rn in row_names}
    gap_stats: Dict[str, Dict[str, Dict[str, float]]] = {rn: {} for rn in row_names}
    sc_stats: Dict[str, Dict[str, Dict[str, float]]] = {rn: {} for rn in row_names}
    for rn in row_names:
        for m in methods:
            cm, csd = mean_sd(cov_runs[rn][m])
            gm, gsd = mean_sd(gap_runs[rn][m])
            sm, ssd = mean_sd(sc_runs[rn][m])
            cov_stats[rn][m] = {"mean": cm, "sd": csd, "delta": cm - target_line}
            gap_stats[rn][m] = {"mean": gm, "sd": gsd}
            sc_stats[rn][m] = {"mean": sm, "sd": ssd}

    # Build formatted strings + dagger/bold/underline
    def fmt_cov(cm: float, csd: float, delta: float, gm: float, gsd: float, *, dagger: bool, bold: bool, underline: bool) -> str:
        if panel_a_mode_use == "coverage_gap":
            if not np.isfinite(gm):
                return "--"
            s = 100.0
            core = f"{(s * gm):.{digits_cov}f} ± {(s * gsd):.{digits_cov}f}"
        else:
            if not np.isfinite(cm):
                return "--"
            s = 100.0
            d = delta
            core = f"{(s * cm):.{digits_cov}f} ({(s * d):+.{digits_cov}f}) ± {(s * csd):.{digits_cov}f}"
        core = core + (r"$^{\dagger}$" if dagger else "")
        if bold:
            return r"\textbf{" + core + "}"
        if underline:
            return r"\underline{" + core + "}"
        return core

    def fmt_sc(sm: float, ssd: float, *, bold: bool, underline: bool) -> str:
        if not np.isfinite(sm):
            return "--"
        core = f"{sm:.{digits_score}f} ± {ssd:.{digits_score}f}"
        if bold:
            return r"\textbf{" + core + "}"
        if underline:
            return r"\underline{" + core + "}"
        return core

    panelA: Dict[str, Dict[str, str]] = {rn: {} for rn in row_names}
    panelB: Dict[str, Dict[str, str]] = {rn: {} for rn in row_names}
    undercovers: Dict[str, Dict[str, bool]] = {rn: {} for rn in row_names}

    for rn in row_names:
        # dagger rule: mean coverage < target - tol
        for m in methods:
            cm = cov_stats[rn][m]["mean"]
            csd = cov_stats[rn][m]["sd"]
            delta = cov_stats[rn][m]["delta"]
            gm = gap_stats[rn][m]["mean"]
            gsd = gap_stats[rn][m]["sd"]
            dagger = bool(np.isfinite(cm) and np.isfinite(target_line) and (cm < target_line - float(coverage_tol)))
            undercovers[rn][m] = dagger

        # Panel A: best/second-best = smallest/second-smallest gap among non-dagger methods
        eligible_a = [m for m in methods if not undercovers[rn][m] and np.isfinite(gap_stats[rn][m]["mean"])]
        best_a = None
        second_a = None
        if len(eligible_a) >= 1:
            eligible_a_sorted = sorted(eligible_a, key=lambda mm: gap_stats[rn][mm]["mean"])
            best_a = eligible_a_sorted[0]
            if len(eligible_a_sorted) >= 2:
                second_a = eligible_a_sorted[1]

        for m in methods:
            cm = cov_stats[rn][m]["mean"]
            csd = cov_stats[rn][m]["sd"]
            delta = cov_stats[rn][m]["delta"]
            gm = gap_stats[rn][m]["mean"]
            gsd = gap_stats[rn][m]["sd"]
            dagger = undercovers[rn][m]
            panelA[rn][m] = fmt_cov(cm, csd, delta, gm, gsd, dagger=dagger, bold=(best_a == m), underline=(second_a == m))

        # Panel B: best/second-best (lowest) interval score among methods that are NOT dagger
        eligible = [m for m in methods if not undercovers[rn][m] and np.isfinite(sc_stats[rn][m]["mean"])]
        best_m = None
        second_m = None
        if len(eligible) >= 1:
            eligible_sorted = sorted(eligible, key=lambda mm: sc_stats[rn][mm]["mean"])
            best_m = eligible_sorted[0]
            if len(eligible_sorted) >= 2:
                second_m = eligible_sorted[1]
        for m in methods:
            sm = sc_stats[rn][m]["mean"]
            ssd = sc_stats[rn][m]["sd"]
            panelB[rn][m] = fmt_sc(sm, ssd, bold=(best_m == m), underline=(second_m == m))

    # Optional LaTeX (two panels)
    latex_str = None
    if latex:
        if panel_a_mode_use == "coverage_gap":
            cap_a = r"\textbf{Panel A: Coverage gap}"
            cap_text = (
                "Coverage gap and interval score across conditioning partitions. "
                "Values are macro-averaged over groups within each partition and reported as mean$\\pm$sd over runs. "
                "In Panel A, coverage gap $=|$empirical coverage $-$ target$|$ (in \\%); bold/underline = best/second-best (smallest gap) among methods that satisfy coverage (no dagger). "
                "In Panel B, lower is better; bold/underline = best/second-best (lowest) interval score among methods that satisfy coverage."
            )
        else:
            cap_a = r"\textbf{Panel A: Coverage}"
            cap_text = (
                "Coverage and interval score across conditioning partitions. "
                "Values are macro-averaged over groups within each partition and reported as mean$\\pm$sd over runs. "
                "In Panel A, $\\Delta$ denotes coverage minus target $(1-\\alpha)$; bold/underline = best/second-best (smallest $|\\Delta|$) among methods that satisfy coverage (no dagger). "
                "In Panel B, lower is better; bold/underline = best/second-best (lowest) interval score among methods that satisfy coverage."
            )
        cap = latex_caption or cap_text
        cols = "l" + "c" * len(methods)
        header = "Condition & " + " & ".join(labels) + r" \\"

        def rows_from(panel: Dict[str, Dict[str, str]]) -> str:
            lines = []
            for rn in row_names:
                vals = [panel[rn][m] for m in methods]
                lines.append(rn + " & " + " & ".join(vals) + r" \\")
            return "\n".join(lines)

        latex_str = (
            r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            rf"\caption{{{cap}}}" "\n"
            rf"\label{{{latex_label}}}" "\n"
            r"\small" "\n"
            r"\setlength{\tabcolsep}{6pt}" "\n"
            r"\begin{minipage}{\textwidth}" "\n"
            rf"\subcaption*{{{cap_a}}}" "\n"
            rf"\begin{{tabular}}{{{cols}}}" "\n"
            r"\toprule" "\n"
            + header + "\n"
            r"\midrule" "\n"
            + rows_from(panelA) + "\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n\n"
            r"\vspace{0.6em}" "\n"
            r"\subcaption*{\textbf{Panel B: Interval score (lower is better)}}" "\n"
            rf"\begin{{tabular}}{{{cols}}}" "\n"
            r"\toprule" "\n"
            + header + "\n"
            r"\midrule" "\n"
            + rows_from(panelB) + "\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n\n"
            r"\begin{flushleft}" "\n"
            r"\footnotesize" "\n"
            rf"$^{{\dagger}}$ Coverage below target $-\,$tolerance (target$-{float(coverage_tol):.3f}$). "
            r"Interval score is the (negatively oriented) interval scoring rule." "\n"
            r"\end{flushleft}" "\n"
            r"\end{minipage}" "\n"
            r"\end{table*}" "\n"
        )

    out = {
        "methods": methods,
        "labels": labels,
        "rows": row_names,
        "target": target_line,
        "coverage_tol": float(coverage_tol),
        "min_group_size": int(min_group_size),
        "panel_a_mode": panel_a_mode_use,
        "panelA": panelA,
        "panelB": panelB,
        "coverage_stats": cov_stats,
        "gap_stats": gap_stats,
        "score_stats": sc_stats,
        "latex": latex_str,
    }
    return out


def plot_methods_by_time_and_channel_coverage_and_interval_score_aggregated(
    artifacts: list[dict],
    *,
    method_labels: Dict[str, str],
    method_order: Optional[Sequence[str]] = None,
    # Time binning (for the "time" plots)
    n_time_bins: int = 10,
    alpha_target: Optional[float] = None,
    # Artifact keys
    t_key: str = "t_test",
    channel_key: str = "chan_test",
    y_key: str = "Y_test_true",
    # Plot options
    title: Optional[str] = None,
    show_gap: bool = False,  # if True, plot |coverage-target| instead of coverage
    show: bool = True,
    save_plots: bool = False,
    save_dir: str = "result_img",
    save_basename: Optional[str] = None,
    dpi: int = 200,
    figsize: Tuple[float, float] = (11.0, 3.8),
    capsize: float = 3.0,
    x_jitter_frac: float = 0.12,
    bar_lw: float = 3,
):
    """
    Compare *all specified methods* by time bins and by channel using `artifacts`.

    Produces four figures:
      1) coverage vs time-bin center (mean±sd over runs), with optional gap-to-target display
      2) interval score vs time-bin center (mean±sd over runs)
      3) coverage by channel j (mean±sd over runs), with optional gap-to-target display
      4) interval score by channel j (mean±sd over runs)

    Error display:
      - Uses error bars (±sd over runs), not bands.
      - At each x location, different methods are slightly x-shifted so bars/markers don't overlap.

    Method inclusion:
      - Methods are taken from `method_labels` keys. For a method name `m`, this function expects
        keys in each artifact: `covered_{m}`, `lower_{m}`, `upper_{m}`.

    Notes:
      - Time bins are fixed equal-width on [0, T-1] (aligned across runs).
    """
    if len(artifacts) == 0:
        raise ValueError("artifacts list is empty")
    if method_labels is None or len(method_labels) == 0:
        raise ValueError("method_labels must be provided and non-empty")
    nb = int(n_time_bins)
    if nb < 2:
        raise ValueError("n_time_bins must be >= 2")

    # Order methods
    methods = [str(m) for m in method_labels.keys()]
    if method_order is not None:
        order = [str(x) for x in method_order]
        pos = {m: i for i, m in enumerate(order)}
        methods = sorted(methods, key=lambda m: pos.get(m, 10**9))
    labels = [str(method_labels[m]) for m in methods]

    def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        lo = np.asarray(lo, dtype=float).ravel()
        hi = np.asarray(hi, dtype=float).ravel()
        if not (y.size == lo.size == hi.size):
            raise ValueError("interval_score: y, lo, hi must have the same length")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("interval_score: alpha must be in (0,1)")
        width = hi - lo
        below = y < lo
        above = y > hi
        pen = np.zeros_like(width, dtype=float)
        if np.any(below):
            pen[below] = (2.0 / float(alpha)) * (lo[below] - y[below])
        if np.any(above):
            pen[above] = (2.0 / float(alpha)) * (y[above] - hi[above])
        return width + pen

    T_ref = int(artifacts[0].get("T", 0))
    if T_ref <= 1:
        raise ValueError("Invalid T in artifacts[0]; required for fixed time bins.")
    for art in artifacts[1:]:
        if int(art.get("T", T_ref)) != T_ref:
            raise RuntimeError("All artifacts must share the same T for fixed time binning.")

    edges = np.linspace(0.0, float(T_ref - 1), nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def bin_id(t: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, nb - 1)

    # Per-method per-run arrays (n_runs, nb)
    cov_runs: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    score_runs: Dict[str, List[np.ndarray]] = {m: [] for m in methods}

    for art in artifacts:
        t = np.asarray(art.get(t_key, []), dtype=float).ravel()
        y = np.asarray(art.get(y_key, []), dtype=float).ravel()
        if not (t.size == y.size):
            raise ValueError(f"{t_key} and {y_key} must be aligned per artifact")
        b = bin_id(t)
        alpha_use = float(alpha_target) if alpha_target is not None else float(art.get("alpha", 0.1))
        target = 1.0 - float(alpha_use)

        for m in methods:
            covered = np.asarray(art.get(f"covered_{m}", []), dtype=bool).ravel()
            lo = np.asarray(art.get(f"lower_{m}", []), dtype=float).ravel()
            hi = np.asarray(art.get(f"upper_{m}", []), dtype=float).ravel()
            if not (covered.size == lo.size == hi.size == y.size):
                raise ValueError(f"Artifact arrays misaligned for method '{m}'.")

            cov_bin = np.full(nb, np.nan, dtype=float)
            sc_bin = np.full(nb, np.nan, dtype=float)
            sc_pt = interval_score(y, lo, hi, alpha=alpha_use) if y.size else np.asarray([], dtype=float)
            for k in range(nb):
                mk = b == k
                if not np.any(mk):
                    continue
                cov_val = float(np.mean(covered[mk]))
                cov_bin[k] = abs(cov_val - target) if bool(show_gap) else cov_val
                sc_bin[k] = float(np.mean(sc_pt[mk]))

            cov_runs[m].append(cov_bin)
            score_runs[m].append(sc_bin)

    cov_mean = {m: np.nanmean(np.stack(cov_runs[m], axis=0), axis=0) for m in methods}
    cov_sd = {m: np.nanstd(np.stack(cov_runs[m], axis=0), axis=0) for m in methods}
    sc_mean = {m: np.nanmean(np.stack(score_runs[m], axis=0), axis=0) for m in methods}
    sc_sd = {m: np.nanstd(np.stack(score_runs[m], axis=0), axis=0) for m in methods}

    # Grouped-bar geometry for time bins
    if nb >= 2:
        group_w_time = float(centers[1] - centers[0])
    else:
        group_w_time = 1.0
    group_w_time = max(group_w_time, 1e-9)
    total_bar_span_time = float(x_jitter_frac) * group_w_time  # fraction of bin spacing used by all bars
    bar_w_time = total_bar_span_time / max(len(methods), 1)
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * bar_w_time

    base = save_basename or (title if title is not None else "methods_by_X")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base.strip())[:120]

    # --- Figure 1: coverage (or abs gap) ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    for idx, (m, lab) in enumerate(zip(methods, labels)):
        x = centers + offsets[idx]
        is_tmfv = str(m).lower() == "tmfv"
        ax1.bar(
            x,
            cov_mean[m],
            width=bar_w_time * 0.95,
            yerr=cov_sd[m],
            capsize=capsize,
            alpha=0.85,
            edgecolor="black" if is_tmfv else None,
            linewidth=bar_lw if is_tmfv else 0.0,
            label=f"{lab}",
        )
    alpha_line = float(alpha_target) if alpha_target is not None else float(artifacts[0].get("alpha", 0.1))
    if not show_gap and 0.0 < alpha_line < 1.0:
        ax1.axhline(1.0 - alpha_line, color="black", linestyle="--", linewidth=1, alpha=0.6)
    if show_gap:
        ax1.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xlabel("time (bin centers)")
    ax1.set_ylabel("|coverage - target|" if show_gap else "coverage")
    ax1.set_title((title + " - " if title else "") + ("time: abs conditional coverage gap" if show_gap else "time: conditional coverage"))
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best", ncol=min(3, max(1, len(methods))))
    fig1.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, f"{base}_coverage_by_time.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # --- Figure 2: interval score ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    for idx, (m, lab) in enumerate(zip(methods, labels)):
        x = centers + offsets[idx]
        is_tmfv = str(m).lower() == "tmfv"
        ax2.bar(
            x,
            sc_mean[m],
            width=bar_w_time * 0.95,
            yerr=sc_sd[m],
            capsize=capsize,
            alpha=0.85,
            edgecolor="black" if is_tmfv else None,
            linewidth=bar_lw if is_tmfv else 0.0,
            label=f"{lab}",
        )
    # Start y-axis from (min score across all methods/bins) - 1
    min_sc_time = float(np.nanmin(np.concatenate([np.asarray(sc_mean[m], dtype=float).ravel() for m in methods])))
    if np.isfinite(min_sc_time):
        ax2.set_ylim(bottom=min_sc_time - 1.0)
    ax2.set_xlabel("time (bin centers)")
    ax2.set_ylabel("interval score (smaller is better)")
    ax2.set_title((title + " - " if title else "") + "time: interval score")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="best", ncol=min(3, max(1, len(methods))))
    fig2.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig2.savefig(os.path.join(save_dir, f"{base}_interval_score_by_time.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    # --- Channel plots (coverage and interval score by channel j) ---
    J_max = int(
        np.max(np.concatenate([np.asarray(a.get(channel_key, []), dtype=int).ravel() for a in artifacts if np.asarray(a.get(channel_key, [])).size > 0]))
        + 1
    )
    if J_max <= 0:
        raise ValueError("Could not determine J_max from artifacts for channel plots.")

    cov_ch_runs: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    sc_ch_runs: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    for art in artifacts:
        ch = np.asarray(art.get(channel_key, []), dtype=int).ravel()
        y = np.asarray(art.get(y_key, []), dtype=float).ravel()
        if not (ch.size == y.size):
            raise ValueError(f"{channel_key} and {y_key} must be aligned per artifact")
        alpha_use = float(alpha_target) if alpha_target is not None else float(art.get("alpha", 0.1))
        target = 1.0 - float(alpha_use)
        for m in methods:
            covered = np.asarray(art.get(f"covered_{m}", []), dtype=bool).ravel()
            lo = np.asarray(art.get(f"lower_{m}", []), dtype=float).ravel()
            hi = np.asarray(art.get(f"upper_{m}", []), dtype=float).ravel()
            if not (covered.size == lo.size == hi.size == y.size):
                raise ValueError(f"Artifact arrays misaligned for method '{m}'.")
            cov_j = np.full(J_max, np.nan, dtype=float)
            sc_j = np.full(J_max, np.nan, dtype=float)
            sc_pt = interval_score(y, lo, hi, alpha=alpha_use) if y.size else np.asarray([], dtype=float)
            for j in range(J_max):
                mj = ch == j
                if not np.any(mj):
                    continue
                cov_val = float(np.mean(covered[mj]))
                cov_j[j] = abs(cov_val - target) if bool(show_gap) else cov_val
                sc_j[j] = float(np.mean(sc_pt[mj]))
            cov_ch_runs[m].append(cov_j)
            sc_ch_runs[m].append(sc_j)

    cov_ch_mean = {m: np.nanmean(np.stack(cov_ch_runs[m], axis=0), axis=0) for m in methods}
    cov_ch_sd = {m: np.nanstd(np.stack(cov_ch_runs[m], axis=0), axis=0) for m in methods}
    sc_ch_mean = {m: np.nanmean(np.stack(sc_ch_runs[m], axis=0), axis=0) for m in methods}
    sc_ch_sd = {m: np.nanstd(np.stack(sc_ch_runs[m], axis=0), axis=0) for m in methods}

    xs = np.arange(J_max, dtype=float)
    # offsets in channel space
    group_w_ch = 1.0
    total_bar_span_ch = float(x_jitter_frac) * group_w_ch
    bar_w_ch = total_bar_span_ch / max(len(methods), 1)
    offsets_ch = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * bar_w_ch

    fig3, ax3 = plt.subplots(figsize=figsize)
    for idx, (m, lab) in enumerate(zip(methods, labels)):
        x = xs + offsets_ch[idx]
        is_tmfv = str(m).lower() == "tmfv"
        ax3.bar(
            x,
            cov_ch_mean[m],
            width=bar_w_ch * 0.95,
            yerr=cov_ch_sd[m],
            capsize=capsize,
            alpha=0.85,
            edgecolor="black" if is_tmfv else None,
            linewidth=bar_lw if is_tmfv else 0.0,
            label=f"{lab}",
        )
    if not show_gap and 0.0 < alpha_line < 1.0:
        ax3.axhline(1.0 - alpha_line, color="black", linestyle="--", linewidth=1, alpha=0.6)
    if show_gap:
        ax3.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax3.set_xlabel("channel j")
    ax3.set_ylabel("|coverage - target|" if show_gap else "coverage")
    ax3.set_title((title + " - " if title else "") + ("coverage gap by channel" if show_gap else "coverage by channel"))
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(loc="best", ncol=min(3, max(1, len(methods))))
    fig3.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig3.savefig(os.path.join(save_dir, f"{base}_coverage_by_channel.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=figsize)
    for idx, (m, lab) in enumerate(zip(methods, labels)):
        x = xs + offsets_ch[idx]
        is_tmfv = str(m).lower() == "tmfv"
        ax4.bar(
            x,
            sc_ch_mean[m],
            width=bar_w_ch * 0.95,
            yerr=sc_ch_sd[m],
            capsize=capsize,
            alpha=0.85,
            edgecolor="black" if is_tmfv else None,
            linewidth=bar_lw if is_tmfv else 0.0,
            label=f"{lab}",
        )
    # Start y-axis from (min score across all methods/channels) - 1
    min_sc_ch = float(np.nanmin(np.concatenate([np.asarray(sc_ch_mean[m], dtype=float).ravel() for m in methods])))
    if np.isfinite(min_sc_ch):
        ax4.set_ylim(bottom=min_sc_ch - 1.0)
    ax4.set_xlabel("channel j")
    ax4.set_ylabel("interval score (smaller is better)")
    ax4.set_title((title + " - " if title else "") + "interval score by channel")
    ax4.grid(True, axis="y", alpha=0.25)
    ax4.legend(loc="best", ncol=min(3, max(1, len(methods))))
    fig4.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig4.savefig(os.path.join(save_dir, f"{base}_interval_score_by_channel.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig4)

    summary = {
        "methods": methods,
        "labels": labels,
        "time_centers": centers,
        "time_cov_mean": cov_mean,
        "time_cov_sd": cov_sd,
        "time_score_mean": sc_mean,
        "time_score_sd": sc_sd,
        "channel_xs": xs,
        "channel_cov_mean": cov_ch_mean,
        "channel_cov_sd": cov_ch_sd,
        "channel_score_mean": sc_ch_mean,
        "channel_score_sd": sc_ch_sd,
        "n_runs": len(artifacts),
        "n_time_bins": nb,
        "time_edges": edges,
    }
    return (fig1, fig2, fig3, fig4), summary


# Backward-compatible alias (older name)
def plot_methods_by_X_coverage_and_interval_score_aggregated(*args, **kwargs):
    return plot_methods_by_time_and_channel_coverage_and_interval_score_aggregated(*args, **kwargs)
