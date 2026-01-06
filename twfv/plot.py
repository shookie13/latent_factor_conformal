import numpy as np
import matplotlib.pyplot as plt
from .cp import unflatten

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
    keys: tuple[str, ...] = ("X", "Y", "abs_resid"),
    title: str | None = None,
    alpha_target: float | None = None,
):
    """
    Compare two conditional interval reports produced by twfv.metrics.conditional_interval_report.

    Each report is a dict mapping keys (e.g. "X","Y","abs_resid") -> {"edges":..., "rows":[...]}
    where each row has fields: bin, left, right, n, coverage, avg_length.

    Produces a grid of plots: one row per key, two columns (coverage, avg interval length).
    """
    # Filter keys to those available in both reports
    keys_use = [k for k in keys if (k in report_a and k in report_b)]
    if len(keys_use) == 0:
        raise ValueError("No overlapping keys found in both reports.")

    nrows = len(keys_use)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3.2 * nrows), squeeze=False)

    for row_i, k in enumerate(keys_use):
        rows_a = report_a[k]["rows"]
        rows_b = report_b[k]["rows"]
        nb = min(len(rows_a), len(rows_b))

        cov_a = np.array([rows_a[i]["coverage"] for i in range(nb)], dtype=float)
        cov_b = np.array([rows_b[i]["coverage"] for i in range(nb)], dtype=float)
        len_a = np.array([rows_a[i]["avg_length"] for i in range(nb)], dtype=float)
        len_b = np.array([rows_b[i]["avg_length"] for i in range(nb)], dtype=float)
        n_a = np.array([rows_a[i]["n"] for i in range(nb)], dtype=float)
        n_b = np.array([rows_b[i]["n"] for i in range(nb)], dtype=float)

        # Bin x-axis: use bin centers for readability
        centers = np.array([(rows_a[i]["left"] + rows_a[i]["right"]) / 2.0 for i in range(nb)], dtype=float)

        ax_cov = axes[row_i, 0]
        ax_len = axes[row_i, 1]

        ax_cov.plot(centers, cov_a, marker="o", label=f"{label_a}")
        ax_cov.plot(centers, cov_b, marker="o", label=f"{label_b}")
        if alpha_target is not None:
            ax_cov.axhline(1.0 - float(alpha_target), color="black", linestyle="--", linewidth=1, alpha=0.6, label="target")
        ax_cov.set_title(f"{k}: coverage by quantile bins")
        ax_cov.set_xlabel(f"{k} (bin centers)")
        ax_cov.set_ylabel("coverage")
        ax_cov.set_ylim(0.0, 1.0)
        ax_cov.grid(True, alpha=0.25)
        ax_cov.legend(loc="best")

        ax_len.plot(centers, len_a, marker="o", label=f"{label_a}")
        ax_len.plot(centers, len_b, marker="o", label=f"{label_b}")
        ax_len.set_title(f"{k}: avg interval length by quantile bins")
        ax_len.set_xlabel(f"{k} (bin centers)")
        ax_len.set_ylabel("avg length")
        ax_len.grid(True, alpha=0.25)
        ax_len.legend(loc="best")

        # Annotate effective sample sizes lightly (use min of the two for compactness)
        for i in range(nb):
            if np.isfinite(centers[i]):
                ax_cov.annotate(f"n={int(min(n_a[i], n_b[i]))}", (centers[i], cov_a[i] if np.isfinite(cov_a[i]) else 0.0),
                                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.6)

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        plt.tight_layout()
    plt.show()