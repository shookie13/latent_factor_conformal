import numpy as np
import matplotlib.pyplot as plt
import os
import re
import matplotlib.colors as mcolors
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
    # Optional: if provided, add two extra graphs:
    #  (a) coverage by channel j over all (i,t) points
    #  (b) histogram of per-subject coverage over (t,j)
    channel_ids: np.ndarray | None = None,     # (n_points,) int in [0, J-1]
    subject_ids: np.ndarray | None = None,     # (n_points,) int in [0, I-1]
    covered_a: np.ndarray | None = None,       # (n_points,) bool
    covered_b: np.ndarray | None = None,       # (n_points,) bool
    show_point_values: bool = True,
    save_plots: bool = False,
    save_dir: str = "result_img",
    save_basename: str | None = None,
    show: bool = True,
    dpi: int = 200,
    annotation_fontsize: int = 10,
    show_heatmap_subject_channel: bool = True,
    heatmap_style: str = "imshow",  # "imshow" | "contourf"
    heatmap_levels: int = 15,
    heatmap_upsample: int = 1,
):
    """
    Compare two conditional interval reports produced by twfv.metrics.conditional_interval_report.

    Each report is a dict mapping keys (e.g. "X","Y","abs_resid") -> {"edges":..., "rows":[...]}
    where each row has fields: bin, left, right, n, coverage, avg_length.

    Produces a grid of plots: one row per key, two columns (coverage, avg interval length).

    If channel_ids/subject_ids and covered_* are provided, also produces:
      1) coverage by channel j (aggregated over all provided points)
      2) histogram of per-subject coverage (aggregated over all provided points)
      3) coverage by (subject, channel) as either a heatmap ("imshow") or filled contour ("contourf").

    Notes on smoothing:
      - For `heatmap_style="contourf"`, the matrix is upsampled using `scipy.ndimage.zoom`
        with cubic interpolation to make contours smoother.
    """
    # Filter keys to those available in both reports
    keys_use = [k for k in keys if (k in report_a and k in report_b)]
    if len(keys_use) == 0:
        raise ValueError("No overlapping keys found in both reports.")

    nrows = len(keys_use)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 5 * nrows), squeeze=False)

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

        line_a_cov, = ax_cov.plot(centers, cov_a, marker="o", label=f"{label_a}")
        line_b_cov, = ax_cov.plot(centers, cov_b, marker="o", label=f"{label_b}")
        color_a_cov = line_a_cov.get_color()
        color_b_cov = line_b_cov.get_color()
        if alpha_target is not None:
            ax_cov.axhline(1.0 - float(alpha_target), color="black", linestyle="--", linewidth=1, alpha=0.6, label="target")
        ax_cov.set_title(f"{k}: coverage by quantile bins")
        ax_cov.set_xlabel(f"{k} (bin centers)")
        ax_cov.set_ylabel("coverage")
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
                if np.isfinite(centers[i]) and np.isfinite(cov_a[i]):
                    ax_cov.annotate(
                        f"{cov_a[i]:.2f}",
                        (centers[i], cov_a[i]),
                        textcoords="offset points",
                        xytext=(-5, -12),
                        ha="right",
                        fontsize=annotation_fontsize,
                        alpha=0.85,
                        color=color_a_cov,
                    )
                if np.isfinite(centers[i]) and np.isfinite(cov_b[i]):
                    ax_cov.annotate(
                        f"{cov_b[i]:.2f}",
                        (centers[i], cov_b[i]),
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
                ax_cov.annotate(f"n={int(min(n_a[i], n_b[i]))}", (centers[i], cov_a[i] if np.isfinite(cov_a[i]) else 0.0),
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

    if not (ch.ndim == sb.ndim == ca.ndim == cb.ndim == 1):
        raise ValueError("channel_ids, subject_ids, covered_a, covered_b must be 1D arrays")
    if not (ch.size == sb.size == ca.size == cb.size):
        raise ValueError("channel_ids/subject_ids/covered_a/covered_b must have the same length")

    # coverage by channel
    J_max = int(np.max(ch)) + 1 if ch.size > 0 else 0
    cov_by_ch_a = np.full(J_max, np.nan, dtype=float)
    cov_by_ch_b = np.full(J_max, np.nan, dtype=float)
    n_by_ch = np.zeros(J_max, dtype=int)
    for j in range(J_max):
        m = ch == j
        n_by_ch[j] = int(np.sum(m))
        if n_by_ch[j] > 0:
            cov_by_ch_a[j] = float(np.mean(ca[m]))
            cov_by_ch_b[j] = float(np.mean(cb[m]))

    # per-subject coverage distribution
    I_max = int(np.max(sb)) + 1 if sb.size > 0 else 0
    cov_by_sb_a = np.full(I_max, np.nan, dtype=float)
    cov_by_sb_b = np.full(I_max, np.nan, dtype=float)
    n_by_sb = np.zeros(I_max, dtype=int)
    for i in range(I_max):
        m = sb == i
        n_by_sb[i] = int(np.sum(m))
        if n_by_sb[i] > 0:
            cov_by_sb_a[i] = float(np.mean(ca[m]))
            cov_by_sb_b[i] = float(np.mean(cb[m]))

    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax_ch = axes2[0]
    xs = np.arange(J_max)
    line_a_ch, = ax_ch.plot(xs, cov_by_ch_a, marker="o", label=label_a)
    line_b_ch, = ax_ch.plot(xs, cov_by_ch_b, marker="o", label=label_b)
    color_a_ch = line_a_ch.get_color()
    color_b_ch = line_b_ch.get_color()
    if alpha_target is not None:
        ax_ch.axhline(1.0 - float(alpha_target), color="black", linestyle="--", linewidth=1, alpha=0.6, label="target")
    ax_ch.set_xlabel("channel j")
    ax_ch.set_ylabel("coverage (over provided points)")
    ax_ch.set_ylim(0.0, 1.0)
    ax_ch.set_title("Coverage by channel")
    ax_ch.grid(True, alpha=0.25)
    ax_ch.legend(loc="best")
    for j in range(J_max):
        if n_by_ch[j] > 0 and np.isfinite(cov_by_ch_a[j]):
            ax_ch.annotate(f"n={n_by_ch[j]}", (j, cov_by_ch_a[j]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.6)
            if show_point_values:
                ax_ch.annotate(
                    f"{cov_by_ch_a[j]:.2f}",
                    (j, cov_by_ch_a[j]),
                    textcoords="offset points",
                    xytext=(-5, -12),
                    ha="right",
                    fontsize=annotation_fontsize,
                    alpha=0.85,
                    color=color_a_ch,
                )
        if show_point_values and n_by_ch[j] > 0 and np.isfinite(cov_by_ch_b[j]):
            ax_ch.annotate(
                f"{cov_by_ch_b[j]:.2f}",
                (j, cov_by_ch_b[j]),
                textcoords="offset points",
                xytext=(5, -12),
                ha="left",
                fontsize=annotation_fontsize,
                alpha=0.85,
                color=color_b_ch,
            )

    ax_sb = axes2[1]
    a_vals = cov_by_sb_a[np.isfinite(cov_by_sb_a)]
    b_vals = cov_by_sb_b[np.isfinite(cov_by_sb_b)]
    bins = np.linspace(0.5, 1.0, 21)
    ax_sb.hist(a_vals, bins=bins, alpha=0.5, label=label_a)
    ax_sb.hist(b_vals, bins=bins, alpha=0.5, label=label_b)
    if alpha_target is not None:
        ax_sb.axvline(1.0 - float(alpha_target), color="black", linestyle="--", linewidth=1, alpha=0.6, label="target")
    ax_sb.set_xlabel("per-subject coverage")
    ax_sb.set_ylabel("count of subjects")
    ax_sb.set_title("Distribution of per-subject coverage")
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

    # Compute per (channel, subject) coverage matrices
    I_max = int(np.max(sb)) + 1 if sb.size > 0 else 0
    J_max = int(np.max(ch)) + 1 if ch.size > 0 else 0

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

    target = (1.0 - float(alpha_target)) if alpha_target is not None else 0.9
    # Choose symmetric bounds around target for a diverging normalization
    all_vals = np.concatenate([cov_mat_a[np.isfinite(cov_mat_a)], cov_mat_b[np.isfinite(cov_mat_b)]])
    if all_vals.size == 0:
        return
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    # Ensure bounds include target and have non-zero range
    vmin = min(vmin, target)
    vmax = max(vmax, target)
    if np.isclose(vmin, vmax):
        vmin = max(0.0, target - 0.1)
        vmax = min(1.0, target + 0.1)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=target, vmax=vmax)
    # Use non-reversed RdBu so *lower* values map to red and *higher* values map to blue.
    # (The center is controlled by TwoSlopeNorm(vcenter=target).)
    cmap = plt.get_cmap("RdBu_r")  # white-ish around center, red/blue for deviations

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
        figsize=(max(10, I_max * 0.35), max(4.5, J_max * 0.6) * 2),
        constrained_layout=True,
    )
    if heatmap_style_use == "imshow":
        im0 = axes3[0].imshow(cov_mat_a, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        im1 = axes3[1].imshow(cov_mat_b, aspect="auto", origin="lower", cmap=cmap, norm=norm)
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
            # Fill NaNs with target so interpolation doesn't create holes.
            Zp = np.where(np.isfinite(Zp), Zp, float(target))
            if up > 1:
                # zoom factors: (rows=channel, cols=subject)
                Zp = _zoom(Zp, zoom=(up, up), order=3, mode="nearest", prefilter=True)
            return Zp

        Za = prep(cov_mat_a)
        Zb = prep(cov_mat_b)
        y = np.linspace(0, J_max - 1, Za.shape[0])
        x = np.linspace(0, I_max - 1, Za.shape[1])
        X, Y = np.meshgrid(x, y)

        im0 = axes3[0].contourf(X, Y, Za, levels=levels, cmap=cmap, norm=norm)
        im1 = axes3[1].contourf(X, Y, Zb, levels=levels, cmap=cmap, norm=norm)
    axes3[0].set_title(f"{label_a}: coverage by (channel, subject) (target={target:.2f})")
    axes3[0].set_ylabel("channel")
    axes3[0].set_xlabel("subject")

    axes3[1].set_title(f"{label_b}: coverage by (channel, subject) (target={target:.2f})")
    axes3[1].set_ylabel("channel")
    axes3[1].set_xlabel("subject")

    # Ticks (keep light to avoid clutter)
    if I_max <= 30:
        axes3[0].set_xticks(np.arange(I_max))
        axes3[1].set_xticks(np.arange(I_max))
    if J_max <= 30:
        axes3[0].set_yticks(np.arange(J_max))
        axes3[1].set_yticks(np.arange(J_max))

    # Shared colorbar
    cbar = fig3.colorbar(im1, ax=axes3.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("coverage")

    if title is not None:
        fig3.suptitle(title + " (subject×channel heatmaps)")

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        fig3.savefig(os.path.join(save_dir, f"{base}_heatmap_subject_channel.png"), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig3)
        