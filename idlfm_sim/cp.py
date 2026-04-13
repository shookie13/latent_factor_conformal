from typing import Protocol, Callable, Optional, List, Tuple
import numpy as np
from scipy.stats import norm, uniform  # for kernels
import functools

Shape3 = Tuple[int, int, int]  # (I, J, T)

def unflatten(k: int, I: int, J: int, T: int) -> Tuple[int, int, int]:
    """Inverse of flatten for 0-based indices."""
    ij, t = divmod(k, T)
    i, j = divmod(ij, J)
    return i, j, t

class CPMethod(Protocol):
    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None: ...
    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None: ...
    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray: ...
    # Optional: calibrate/predict using externally provided fitted means
    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None: ...  # type: ignore[override]
    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray: ...  # type: ignore[override]


class NaiveAbsoluteResidualCP:
    """
    Baseline split conformal using absolute residuals per stream.
    This implementation does NOT fit a built-in mean model; it expects
    externally provided fitted means for calibration and prediction.
    """
    def __init__(self, shape: Shape3, alpha: float = 0.1):
        self.I, self.J, self.T = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.alpha = alpha
        self.q_abs = np.full(self.J, np.nan, dtype=float)

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("This CP method does not fit an internal model. Calibrate with calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("This CP method requires fitted means. Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("This CP method does not center on an internal model. Use predict_interval_from_fit(fit).")

    # New: calibration and prediction using externally provided fitted means
    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        """
        Calibrate a *global* absolute residual quantile using provided fitted values (no internal fit required).
        """
        # Compute all residuals globally (do not split by stream)
        res = np.abs(np.asarray(Y_cal_true) - np.asarray(Y_cal_fit))
        m = res.size
        if m == 0:
            self.q_abs[:] = np.nan
        else:
            q_level = min(1.0, np.ceil((m + 1) * (1 - self.alpha)) / m)
            try:
                q_val = float(np.quantile(res, q_level, method="higher"))
            except TypeError:
                # for older numpy
                q_val = float(np.quantile(res, q_level, interpolation="higher"))
            self.q_abs[:] = q_val  # set the same quantile for all streams

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        """
        Produce intervals centered at provided fitted means using calibrated per-stream quantiles.
        """
        out = np.zeros((X_test_idx.size, 2), dtype=float)
        for idx, (k, y_fit) in enumerate(zip(X_test_idx.tolist(), Y_test_fit.tolist())):
            _, j, _ = unflatten(int(k), self.I, self.J, self.T)
            q = self.q_abs[j]
            if not np.isfinite(q):
                q = 0.0
            out[idx, 0] = y_fit - q
            out[idx, 1] = y_fit + q
        return out


class WeightedSplitConformalCP:
    def __init__(
        self,
        shape: Shape3,
        alpha: float = 0.1,
        p_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        test_w_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.I, self.J, self.T = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.alpha = alpha
        self._p_fn = p_fn or (lambda idx: np.full(idx.size, 0.5, dtype=float))
        # Optional test-sampling weights p_i^{test}. We allow unnormalized weights; the
        # conformalization step normalizes them together with the ghost weight.
        self._test_w_fn = test_w_fn or (lambda idx: np.ones(idx.size, dtype=float))
        self._S_sorted = None
        self._w_cal_sorted = None
    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("This CP method does not fit an internal model. Calibrate with calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("This CP method requires fitted means. Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("This CP method does not center on an internal model. Use predict_interval_from_fit(fit).")
    def calibrate_from_fit(self, X_cal_idx, Y_cal_true, Y_cal_fit):
        S_cal = np.abs(Y_cal_true - Y_cal_fit)
        X_cal_idx = np.asarray(X_cal_idx)
        p_cal = np.clip(self._p_fn(X_cal_idx), 1e-6, np.inf)
        r_cal = (1.0 - p_cal) / p_cal           # r_i
        test_w_cal = np.clip(self._test_w_fn(X_cal_idx), 0.0, np.inf)
        w_cal = test_w_cal * r_cal                      # p_i^{test} * r_i
        order = np.argsort(S_cal)
        self._S_sorted = S_cal[order]
        self._w_cal_sorted = w_cal[order]

    def predict_interval_from_fit(self, X_test_idx, Y_test_fit, alpha: float):
        if self._S_sorted is None or self._w_cal_sorted is None:
            raise RuntimeError("Call calibrate_from_fit before prediction.")
        X_test_idx = np.asarray(X_test_idx)
        S_sorted = self._S_sorted
        w_cal_sorted = self._w_cal_sorted
        n_test = X_test_idx.size
        out = np.zeros((n_test, 2), dtype=float)

        p_test = np.clip(self._p_fn(X_test_idx), 1e-6, np.inf)
        r_ghost = (1.0 - p_test) / p_test       # r_*
        test_w_test = np.clip(self._test_w_fn(X_test_idx), 0.0, np.inf)
        w_ghost = test_w_test * r_ghost                 # p_*^{test} * r_*

        for j in range(n_test):
            all_w = np.concatenate([w_cal_sorted, np.array([w_ghost[j]])])
            w = all_w / np.sum(all_w)
            w_sorted = w[:-1]                           # weights of finite scores
            cumw = np.cumsum(w_sorted)
            idx = np.searchsorted(cumw, 1 - alpha, side="left")
            if idx < S_sorted.shape[0]:
                q_alpha = S_sorted[idx]
            else:
                q_alpha = np.inf
            yhat = float(Y_test_fit[j])
            out[j, 0] = yhat - q_alpha
            out[j, 1] = yhat + q_alpha
        return out

class LocalizedSplitConformalCP:
    """
    Localized weighted split conformal CP with i,j-specific time localization.

    For each test point (i,j,t), only calibrates using calibration points with the same
    (i,j) and time within the bandwidth (via kernel).

    H_samp: callable taking X_center (time index) and returning (tilde_X, H) where
            H(X, X_center=tilde_X) returns localization weights for array X (times).
    p_fn: callable p_fn(flat_indices) -> probabilities in (0,1] for missingness.
    """
    def __init__(
        self,
        shape: Shape3,
        alpha: float = 0.1,
        p_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        H_samp: Optional[Callable[..., tuple]] = None,
        bandwidth: float = 10,
        test_w_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        repr_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.I, self.J, self.T = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.alpha = alpha
        # default sampler: gaussian kernel bandwidth=10
        self._bandwidth = float(bandwidth)
        self._H_samp = H_samp or (lambda X_center: kernel_sampler(X_center=X_center, type="gaussian", bandwidth=self._bandwidth))
        self._test_w_fn = test_w_fn or (lambda idx: np.ones(idx.size, dtype=float))
        # Optional learned representation g(X). If provided, localization is performed by
        # Gaussian kernel distance in representation space instead of raw time.
        self._repr_fn = repr_fn
        # Cached calibration
        self._S_cal: Optional[np.ndarray] = None
        self._t_cal: Optional[np.ndarray] = None
        self._i_cal: Optional[np.ndarray] = None  # <-- New for i index
        self._j_cal: Optional[np.ndarray] = None  # <-- New for j index
        self._X_cal_idx: Optional[np.ndarray] = None
        self._repr_cal: Optional[np.ndarray] = None

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("This CP method does not fit an internal model. Calibrate with calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("This CP method requires fitted means. Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("This CP method does not center on an internal model. Use predict_interval_from_fit(fit).")

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        S_cal = np.abs(Y_cal_true - Y_cal_fit)  # (n_cal,)
        # Cache calibration indices for optional test-sampling weights
        self._X_cal_idx = np.asarray(X_cal_idx, dtype=int)
        if self._repr_fn is not None:
            repr_cal = np.asarray(self._repr_fn(self._X_cal_idx), dtype=float)
            if repr_cal.ndim == 1:
                repr_cal = repr_cal[:, None]
            self._repr_cal = repr_cal
        else:
            self._repr_cal = None
        n_samples = X_cal_idx.size
        t_cal = np.empty(n_samples, dtype=int)
        i_cal = np.empty(n_samples, dtype=int)
        j_cal = np.empty(n_samples, dtype=int)

        for n, k in enumerate(X_cal_idx.tolist()):
            i, j, t = unflatten(int(k), self.I, self.J, self.T)
            t_cal[n] = t
            i_cal[n] = i
            j_cal[n] = j
        self._S_cal = S_cal
        self._t_cal = t_cal
        self._i_cal = i_cal
        self._j_cal = j_cal

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        if self._S_cal is None or self._t_cal is None or self._i_cal is None or self._j_cal is None:
            raise RuntimeError("Call calibrate_from_fit before prediction.")
        X_cal_idx = getattr(self, "_X_cal_idx", None)
        if X_cal_idx is None:
            raise RuntimeError("Missing cached calibration indices. Call calibrate_from_fit before prediction.")

        n_test = int(X_test_idx.size)
        out = np.zeros((n_test, 2), dtype=float)
        # pre-extract test indices
        t_test = np.empty(n_test, dtype=int)
        i_test = np.empty(n_test, dtype=int)
        j_test = np.empty(n_test, dtype=int)
        for n, k in enumerate(X_test_idx.tolist()):
            i, j, t = unflatten(int(k), self.I, self.J, self.T)
            t_test[n] = t
            i_test[n] = i
            j_test[n] = j

        S_cal = self._S_cal
        t_cal = self._t_cal
        i_cal = self._i_cal
        j_cal = self._j_cal
        repr_test = None
        if self._repr_fn is not None:
            repr_test = np.asarray(self._repr_fn(np.asarray(X_test_idx, dtype=int)), dtype=float)
            if repr_test.ndim == 1:
                repr_test = repr_test[:, None]

        for j in range(n_test):
            # Select calibration points with same i and j as the test point
            mask = (i_cal == i_test[j]) & (j_cal == j_test[j])
            if not np.any(mask):
                # If there are no calibration points for this (i,j), set interval to NaN
                out[j, 0] = np.nan
                out[j, 1] = np.nan
                continue

            S_cal_sel = S_cal[mask]
            t_cal_sel = t_cal[mask]
            X_cal_idx_sel = X_cal_idx[mask]
            n_cal_sel = S_cal_sel.shape[0]

            if self._repr_fn is not None and self._repr_cal is not None and repr_test is not None:
                z_cal = self._repr_cal[mask]
                z_test = repr_test[j]
                bw = float(self._bandwidth)
                if bw <= 0.0 or not np.isfinite(bw):
                    w_loc_cal = np.ones(n_cal_sel, dtype=float)
                else:
                    diff = z_cal - z_test[None, :]
                    d2 = np.sum(diff * diff, axis=1)
                    w_loc_cal = np.exp(-0.5 * d2 / (bw * bw))
                w_loc_ghost = 1.0
            else:
                tilde_X, H = self._H_samp(X_center=t_test[j])
                w_loc_cal = H(X=t_cal_sel, X_center=tilde_X)  # (n_cal_sel,)
                w_loc_ghost = H(X=t_test[j], X_center=tilde_X)  # scalar

            # test-sampling weights (allow unnormalized)
            test_w_cal = np.clip(self._test_w_fn(X_cal_idx_sel), 0.0, np.inf)
            test_w_ghost = float(np.clip(self._test_w_fn(np.array([X_test_idx[j]], dtype=int)), 0.0, np.inf)[0])

            w_tilde_cal = w_loc_cal * test_w_cal
            w_tilde_ghost = w_loc_ghost * test_w_ghost
            all_w = np.concatenate([w_tilde_cal, np.array([w_tilde_ghost])])

            # Avoid division by zero
            if np.sum(all_w) == 0:
                out[j, 0] = np.nan
                out[j, 1] = np.nan
                continue

            w_norm = all_w / np.sum(all_w)
            order = np.argsort(S_cal_sel)
            S_sorted = S_cal_sel[order]
            w_sorted = w_norm[order]
            cumw = np.cumsum(w_sorted)
            idx = np.searchsorted(cumw, 1 - alpha, side="left")
            if idx < n_cal_sel:
                q_alpha = S_sorted[idx]
            else:
                q_alpha = S_sorted[-1]
            yhat = float(Y_test_fit[j])
            out[j, 0] = yhat - q_alpha
            out[j, 1] = yhat + q_alpha
        return out

class LocalizedWeightedSplitConformalCP:
    """
    Localized weighted split conformal CP combining missingness-odds weights with
    deterministic time-localization via a kernel (optionally supplied via H_samp).

    H_samp: callable taking X_center (time index) and returning (tilde_X, H) where
            H(X, X_center=...) returns localization weights for array X (times).
            Any sampled tilde_X is ignored in this implementation; we always use X_center
            as the kernel center (no RLCP sampling).
    p_fn: callable p_fn(flat_indices) -> probabilities in (0,1] for missingness.
    """
    def __init__(
        self,
        shape: Shape3,
        alpha: float = 0.1,
        p_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        H_samp: Optional[Callable[..., tuple]] = None,
        bandwidth: float = 10,
        test_w_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        diagnostics: bool = False,
        lhs_use_test_weight: bool = False,
        repr_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.I, self.J, self.T = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.alpha = alpha
        self._p_fn = p_fn or (lambda idx: np.full(idx.size, 0.5, dtype=float))
        # Default sampler: Gaussian kernel with given bandwidth.
        # If H_samp is not provided, we can fast-path Gaussian kernel computations (avoid SciPy + Python call overhead).
        self._fast_kernel = H_samp is None
        self._kernel_type = "gaussian"
        self._bandwidth = float(bandwidth)
        # Default: deterministic Gaussian kernel (no sampling).
        self._H_samp = H_samp or (lambda X_center: (X_center, functools.partial(kernel, type="gaussian", bandwidth=self._bandwidth)))
        self._test_w_fn = test_w_fn or (lambda idx: np.ones(idx.size, dtype=float))
        self._diagnostics = bool(diagnostics)
        self._lhs_use_test_weight = bool(lhs_use_test_weight)
        self._repr_fn = repr_fn
        self.last_diagnostics: Optional[dict] = None
        # Cached calibration: grouped by (i,j), sorted by score once per group.
        # groups[gid] where gid=i*J+j = (S_sorted, t_sorted, r_sorted, test_w_sorted, repr_sorted_or_none)
        self._groups: Optional[List[Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]]] = None
        # Cached localization kernel for each group (score-sorted order).
        # For gid=i*J+j: H_cache[gid] is (m,m) kernel between localized representations.
        self._H_cache: Optional[List[Optional[np.ndarray]]] = None
        self._X_cal_idx: Optional[np.ndarray] = None

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("This CP method does not fit an internal model. Calibrate with calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("This CP method requires fitted means. Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("This CP method does not center on an internal model. Use predict_interval_from_fit(fit).")

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        # Vectorized unflatten for row-major flatten: k = ((i*J)+j)*T + t
        X_cal_idx = np.asarray(X_cal_idx, dtype=np.int64)
        self._X_cal_idx = X_cal_idx
        S_cal = np.abs(np.asarray(Y_cal_true, dtype=float) - np.asarray(Y_cal_fit, dtype=float))  # (n_cal,)

        ij = X_cal_idx // self.T
        t_cal = (X_cal_idx - ij * self.T).astype(np.int64, copy=False)
        i_cal = (ij // self.J).astype(np.int64, copy=False)
        j_cal = (ij - i_cal * self.J).astype(np.int64, copy=False)
        gid = (i_cal * self.J + j_cal).astype(np.int64, copy=False)  # group id in [0, I*J)

        p_cal = np.clip(self._p_fn(X_cal_idx.astype(int, copy=False)), 1e-6, 1.0)
        r_cal = (1.0 - p_cal) / p_cal

        # Precompute test-sampling weights for calibration indices once.
        test_w_cal = np.clip(self._test_w_fn(X_cal_idx.astype(int, copy=False)), 0.0, np.inf).astype(float, copy=False)
        repr_cal = None
        if self._repr_fn is not None:
            repr_cal = np.asarray(self._repr_fn(X_cal_idx.astype(int, copy=False)), dtype=float)
            if repr_cal.ndim == 1:
                repr_cal = repr_cal[:, None]

        # Build contiguous groups by sorting once by gid (faster than per-test boolean masks).
        order_gid = np.argsort(gid, kind="mergesort")
        gid_sorted = gid[order_gid]
        # group boundaries in the sorted-by-gid array
        boundaries = np.flatnonzero(np.diff(gid_sorted)) + 1
        starts = np.concatenate([np.array([0], dtype=int), boundaries])
        ends = np.concatenate([boundaries, np.array([gid_sorted.size], dtype=int)])

        groups: List[Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]] = [None] * (self.I * self.J)
        H_cache: List[Optional[np.ndarray]] = [None] * (self.I * self.J)
        bw = float(self._bandwidth)
        for st, en in zip(starts.tolist(), ends.tolist()):
            if en <= st:
                continue
            g = int(gid_sorted[st])
            idx_g = order_gid[st:en]

            # Sort by score ONCE per group; reuse in every prediction.
            S_g = S_cal[idx_g]
            ord_s = np.argsort(S_g, kind="mergesort")
            idx_s = idx_g[ord_s]

            repr_sorted_g = None if repr_cal is None else repr_cal[idx_s].astype(float, copy=False)
            groups[g] = (
                S_cal[idx_s].astype(float, copy=False),
                t_cal[idx_s].astype(np.int64, copy=False),
                r_cal[idx_s].astype(float, copy=False),
                test_w_cal[idx_s].astype(float, copy=False),
                repr_sorted_g,
            )

            # Cache deterministic localization kernel for this group (centers=t_sorted, points=t_sorted),
            # in score-sorted order so it lines up with S_sorted columns during prediction.
            t_sorted_g = t_cal[idx_s].astype(float, copy=False)  # (m,)
            m = int(t_sorted_g.size)
            if m > 0:
                if (bw <= 0.0) or (not np.isfinite(bw)):
                    H_cache[g] = np.ones((m, m), dtype=float)
                elif repr_sorted_g is not None:
                    diff = repr_sorted_g[:, None, :] - repr_sorted_g[None, :, :]
                    d2 = np.sum(diff * diff, axis=2)
                    H_cache[g] = np.exp(-0.5 * d2 / (bw * bw)).astype(np.float32, copy=False)
                elif self._fast_kernel and self._kernel_type == "gaussian":
                    u = (t_sorted_g[:, None] - t_sorted_g[None, :]) / bw
                    # Use float32 to reduce memory; downstream promotes as needed.
                    H_cache[g] = np.exp(-0.5 * u * u).astype(np.float32, copy=False)
                else:
                    # General deterministic path: ignore any sampling from H_samp and use X_center=center.
                    H_mat_g = np.empty((m, m), dtype=float)
                    for ic in range(m):
                        _, H = self._H_samp(X_center=int(t_sorted_g[ic]))
                        H_mat_g[ic, :] = np.asarray(H(X=t_sorted_g, X_center=float(t_sorted_g[ic])), dtype=float).reshape(-1)
                    H_cache[g] = np.clip(H_mat_g, 0.0, np.inf)

        self._groups = groups
        self._H_cache = H_cache

    @staticmethod
    def _lhs_indicator_rate(
        S_sorted: np.ndarray,      # (m,) scores in ascending order
        cum: np.ndarray,           # (m+1, m+1) row-wise cumulative probs over [finite..., ghost]
        r_sorted: np.ndarray,      # (m,) missingness odds ratios aligned with S_sorted
        r_ghost: float,            # missingness odds ratio for the test (ghost) point
        beta: float,               # candidate tilde-alpha
        test_w_sorted: Optional[np.ndarray] = None,  # (m,) optional test-sampling weights aligned with S_sorted
        test_w_test: Optional[float] = None,         # optional test-sampling weight for the test (ghost) point
        p_test: Optional[float] = None,              # optional missingness probability for test point; overrides r_ghost
        use_test_weight: bool = False,               # if True, use weights r_i * test_w_i in the LHS average
    ) -> float:
        """
        Compute weighted LHS(beta):

          Default:
            LHS(beta) = (sum_i r_i * 1{V_i <= Q(beta; F_i)}) / (sum_i r_i)

          Optional (use_test_weight=True):
            LHS(beta) = (sum_i r_i * test_w_i * 1{V_i <= Q(beta; F_i)}) / (sum_i r_i * test_w_i)

        with V_{m+1} treated as +inf (ghost), since score at test is unknown.

        Columns: 0..m-1 are finite scores S_sorted; column m is ghost at +inf.
        Rows:    0..m-1 are calibration centers; row m is the test center.
        """
        m = int(S_sorted.size)
        r_sorted = np.asarray(r_sorted, dtype=float).reshape(-1)
        if r_sorted.size != m:
            raise ValueError(f"r_sorted must have shape (m,), got {r_sorted.shape} for m={m}")

        # Allow passing p_test instead of r_ghost (more direct from caller).
        if p_test is not None and np.isfinite(float(p_test)):
            p = float(np.clip(float(p_test), 1e-12, 1.0))
            r_ghost_eff = (1.0 - p) / p
        else:
            r_ghost_eff = float(r_ghost)

        r_cal = np.clip(r_sorted, 0.0, np.inf)
        r_test = float(np.clip(r_ghost_eff, 0.0, np.inf))

        if use_test_weight:
            if test_w_sorted is None:
                w_cal = np.ones(m, dtype=float)
            else:
                w_cal = np.asarray(test_w_sorted, dtype=float).reshape(-1)
                if w_cal.size != m:
                    raise ValueError(f"test_w_sorted must have shape (m,), got {w_cal.shape} for m={m}")
            w_cal = np.clip(w_cal, 0.0, np.inf)
            w_test = 1.0 if (test_w_test is None) else float(np.clip(float(test_w_test), 0.0, np.inf))
            w_lhs = np.concatenate([r_cal * w_cal, np.array([r_test * w_test], dtype=float)])
        else:
            w_lhs = np.concatenate([r_cal, np.array([r_test], dtype=float)])

        denom = float(w_lhs.sum())

        # For each row i, find smallest k with cum[i,k] >= beta
        mask = (cum >= beta)
        any_true = mask.any(axis=1)
        q_idx = np.where(any_true, mask.argmax(axis=1), m)  # safety; beta<=1 should always be true

        # Quantile threshold per row: S_sorted[q_idx] if q_idx < m else +inf
        idx_clip = np.clip(q_idx, 0, max(m - 1, 0))
        thr = S_sorted[idx_clip] if m > 0 else np.array([], dtype=float)
        thr = np.where(q_idx < m, thr, np.inf)

        # For cal rows i=0..m-1: V_i = S_sorted[i]
        ind_cal = (S_sorted <= thr[:m]) if m > 0 else np.array([], dtype=bool)

        # For test row i=m: V_{m+1} = +inf, so indicator is true iff thr_test is +inf (q_idx == m)
        ind_test = bool(q_idx[m] == m)

        num = float(np.dot(w_lhs[:-1], ind_cal.astype(float, copy=False))) + (float(w_lhs[-1]) if ind_test else 0.0)
        if denom <= 0.0:
            # All odds are zero: fall back to the original uniform average.
            return (float(ind_cal.sum()) + (1.0 if ind_test else 0.0)) / float(m + 1)
        return num / denom

    @staticmethod
    def _binary_search_smallest_candidate(
        candidates: np.ndarray,    # sorted increasing, shape (L,)
        S_sorted: np.ndarray,      # (m,)
        cum: np.ndarray,           # (m+1, m+1)
        r_sorted: np.ndarray,      # (m,)
        r_ghost: float,            # scalar
        target: float,             # desired coverage level, e.g. 1 - alpha
        test_w_sorted: Optional[np.ndarray] = None,
        test_w_test: Optional[float] = None,
        p_test: Optional[float] = None,
        use_test_weight: bool = False,
    ) -> int:
        """
        Return index of the smallest candidates[idx] such that LHS(candidates[idx]) >= target.
        Assumes monotonicity in beta (true since quantiles are nondecreasing in beta).
        """
        lo, hi = 0, int(candidates.size) - 1
        best = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            beta = float(candidates[mid])
            lhs = LocalizedWeightedSplitConformalCP._lhs_indicator_rate(
                S_sorted,
                cum,
                r_sorted,
                r_ghost,
                beta,
                test_w_sorted=test_w_sorted,
                test_w_test=test_w_test,
                p_test=p_test,
                use_test_weight=use_test_weight,
            )
            if lhs >= target:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        if self._groups is None or self._X_cal_idx is None:
            raise RuntimeError("Call calibrate_from_fit before prediction.")

        X_test_idx = np.asarray(X_test_idx, dtype=np.int64)
        n_test = int(X_test_idx.size)
        out = np.zeros((n_test, 2), dtype=float)

        # Unflatten test indices
        ij = X_test_idx // self.T
        t_test = (X_test_idx - ij * self.T).astype(np.int64, copy=False)
        i_test = (ij // self.J).astype(np.int64, copy=False)
        j_test = (ij - i_test * self.J).astype(np.int64, copy=False)
        gid_test = (i_test * self.J + j_test).astype(np.int64, copy=False)
        repr_test = None
        if self._repr_fn is not None:
            repr_test = np.asarray(self._repr_fn(X_test_idx.astype(int, copy=False)), dtype=float)
            if repr_test.ndim == 1:
                repr_test = repr_test[:, None]

        # Missingness odds for test + test-sampling weights
        p_test = np.clip(self._p_fn(X_test_idx.astype(int, copy=False)), 1e-6, 1.0)
        r_ghost = (1.0 - p_test) / p_test
        test_w_test = np.clip(self._test_w_fn(X_test_idx.astype(int, copy=False)), 0.0, np.inf).astype(float, copy=False)

        # Optional diagnostics
        diag = None
        if self._diagnostics:
            diag = {
                "ghost_norm": np.full(n_test, np.nan, dtype=float),
                "tilde_alpha": np.full(n_test, np.nan, dtype=float),
                "tilde_alpha_idx": np.full(n_test, -1, dtype=int),
            }

        bw = float(self._bandwidth)
        groups = self._groups
        H_cache = self._H_cache

        for n in range(n_test):
            g = int(gid_test[n])
            grp = groups[g] if (0 <= g < len(groups)) else None
            if grp is None:
                out[n, :] = np.nan
                continue

            S_sorted, t_sorted, r_sorted, test_w_sorted, repr_sorted = grp
            m = int(S_sorted.size)
            if m == 0:
                out[n, :] = np.nan
                continue

            # base weights w_j = (missingness odds) * (test weight)
            w_base = np.clip(r_sorted * test_w_sorted, 0.0, np.inf)          # (m,)
            w_ghost_base = float(np.clip(r_ghost[n] * test_w_test[n], 0.0, np.inf))

            # Points (columns): finite cal points + ghost at +inf (located at t_test[n])
            w_points = np.concatenate([w_base.astype(float, copy=False), np.array([w_ghost_base], dtype=float)])

            # Build H_{i,j} for all centers i and points j
            if bw <= 0.0 or not np.isfinite(bw):
                # no localization
                H_mat = np.ones((m + 1, m + 1), dtype=float)
            elif repr_sorted is not None and repr_test is not None and H_cache is not None and (H_cache[g] is not None):
                H_mat = np.empty((m + 1, m + 1), dtype=float)
                H_mat[:-1, :-1] = H_cache[g]
                z_test = repr_test[n]
                diff = repr_sorted - z_test[None, :]
                d2 = np.sum(diff * diff, axis=1)
                h_col = np.exp(-0.5 * d2 / (bw * bw)).astype(float, copy=False)
                H_mat[:-1, -1] = h_col
                H_mat[-1, :-1] = h_col
                H_mat[-1, -1] = 1.0
            elif self._fast_kernel and self._kernel_type == "gaussian" and H_cache is not None and (H_cache[g] is not None):
                # Deterministic Gaussian localization with cached cal-cal kernel block.
                H_mat = np.empty((m + 1, m + 1), dtype=float)
                H_mat[:-1, :-1] = H_cache[g]  # (m,m)

                # Cal centers -> ghost point at t_test[n]
                u_col = (float(t_test[n]) - t_sorted.astype(float, copy=False)) / bw
                h_col = np.exp(-0.5 * u_col * u_col).astype(float, copy=False)
                H_mat[:-1, -1] = h_col

                # Test center at t_test[n] -> cal points and ghost at t_test[n]
                # (For Gaussian kernel this is symmetric, so row to cal equals h_col.)
                H_mat[-1, :-1] = h_col
                H_mat[-1, -1] = 1.0
            else:
                # General deterministic path (slower): ignore any sampling from H_samp and use X_center=center.
                t_points = np.concatenate([t_sorted.astype(float, copy=False), np.array([float(t_test[n])], dtype=float)])
                t_centers = np.concatenate([t_sorted.astype(float, copy=False), np.array([float(t_test[n])], dtype=float)])
                H_mat = np.empty((m + 1, m + 1), dtype=float)
                for ic in range(m + 1):
                    _, H = self._H_samp(X_center=int(t_centers[ic]))
                    H_row = np.asarray(H(X=t_points[:-1], X_center=float(t_centers[ic])), dtype=float).reshape(-1)
                    H_mat[ic, :-1] = H_row
                    H_mat[ic, -1] = float(np.asarray(H(X=float(t_test[n]), X_center=float(t_centers[ic])), dtype=float))

            H_mat = np.clip(H_mat, 0.0, np.inf)

            # Unnormalized p_{i,j}^H = H_{i,j} * w_j
            P_unnorm = H_mat * w_points[None, :]
            row_sums = P_unnorm.sum(axis=1)

            # Normalize rows; handle degenerate rows (all-zero)
            P = np.zeros_like(P_unnorm)
            good = row_sums > 0
            P[good] = P_unnorm[good] / row_sums[good, None]
            if np.any(~good):
                # fallback: ignore localization, use base weights over finite part; no ghost
                wb = w_base
                s = float(wb.sum())
                if s > 0:
                    P[~good, :-1] = (wb / s)[None, :]
                    P[~good, -1] = 0.0
                else:
                    P[~good, :-1] = 1.0 / m
                    P[~good, -1] = 0.0

            # Row-wise cumulative probs over sorted support [S_sorted..., +inf]
            cum = np.cumsum(P, axis=1)  # (m+1, m+1), last col always 1

            # Candidate list Γ = cumulative weights of F_{n+1} (test row), ordered
            candidates = cum[-1].copy()  # includes last element == 1.0

            # We want coverage = 1 - alpha (your alpha is miscoverage)
            target = float(1.0 - alpha)

            # Binary search for smallest candidate achieving the inequality
            best_idx = self._binary_search_smallest_candidate(
                candidates,
                S_sorted,
                cum,
                r_sorted,
                float(r_ghost[n]),
                target,
                test_w_sorted=test_w_sorted,
                test_w_test=float(test_w_test[n]),
                p_test=float(p_test[n]),
                use_test_weight=self._lhs_use_test_weight,
            )
            tilde_alpha = float(candidates[best_idx])

            # Final radius: q = Q(tilde_alpha; F_{n+1})
            test_mask = (cum[-1] >= tilde_alpha)
            k = int(test_mask.argmax())  # first k where cum >= tilde_alpha
            if k >= m:
                # quantile hits the ghost at +inf -> cap to max finite score
                q = float(S_sorted[-1])
            else:
                q = float(S_sorted[k])

            yhat = float(Y_test_fit[n])
            out[n, 0] = yhat - q
            out[n, 1] = yhat + q

            if diag is not None:
                diag["ghost_norm"][n] = float(P[-1, -1])
                diag["tilde_alpha"][n] = tilde_alpha
                diag["tilde_alpha_idx"][n] = int(best_idx)

        if self._diagnostics:
            self.last_diagnostics = diag

        return out

    def plot_weight_diagnostics(
        self,
        X_test_idx: np.ndarray,
        *,
        test_point: Optional[int] = None,
        x_axis: str = "t",
        scatter_time_score: bool = False,
        color_by: str = "all",
        alpha: Optional[float] = None,
        window_mult: float = 2.0,
        use_sampled_tilde: bool = False,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Diagnostic plot of weight components for a randomly chosen (or specified) test point.

        For the selected test point, we take calibration points in the same (i,j) group and
        restrict to a time window |t_cal - t_test| <= window_mult * bandwidth. Then we plot:
          1) missingness weight      r_cal
          2) localization weight     w_loc
          3) test-sampling weight    test_w_cal
          4) combined weight         r_cal * w_loc * test_w_cal

        Notes:
        - This is intended for debugging weight concentration / ghost dominance.
        - Localization is computed deterministically with center=t_test (no RLCP sampling).
        - x_axis controls the plot x-axis:
            * "t" (default): calibration time t (within the selected time window)
            * "S_cal": calibration score S_cal (for the same points; windowing still uses time)
        - scatter_time_score=True makes a scatter plot with:
            * x = time t
            * y = score S_cal
            * point color = one or more weight components (see color_by)
          This is useful to see whether high scores get high weight.
        - color_by (for scatter_time_score): one of {"missingness","localization","test","combined","all"}.
            * "all" plots 4 subplots (missingness/localization/test/combined) at once.
        - alpha (optional): miscoverage level used to compute and draw the diagnostic q_alpha line.
        """
        if self._groups is None:
            raise RuntimeError("Call calibrate_from_fit before calling plot_weight_diagnostics().")

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            raise RuntimeError(f"matplotlib is required for plotting diagnostics: {e}")

        X_test_idx = np.asarray(X_test_idx, dtype=np.int64).ravel()
        if X_test_idx.size == 0:
            raise ValueError("X_test_idx is empty.")

        rng = np.random.default_rng(seed)
        if test_point is None:
            k_test = int(rng.choice(X_test_idx))
        else:
            k_test = int(test_point)

        # Vectorized unflatten for a single index k = ((i*J)+j)*T + t
        ij = k_test // self.T
        t_test = int(k_test - ij * self.T)
        i_test = int(ij // self.J)
        j_test = int(ij - i_test * self.J)
        gid = int(i_test * self.J + j_test)

        grp = self._groups[gid] if (0 <= gid < len(self._groups)) else None
        if grp is None:
            raise RuntimeError(f"No calibration group found for test point (i={i_test}, j={j_test}) (gid={gid}).")

        S_sorted, t_sorted, r_sorted, test_w_sorted, _ = grp
        if t_sorted.size == 0:
            raise RuntimeError(f"Empty calibration group for (i={i_test}, j={j_test}).")

        bw = float(self._bandwidth)
        if not np.isfinite(bw) or bw <= 0:
            bw = 1.0

        # Restrict to window around t_test (as requested)
        half_window = float(window_mult) * bw
        win_mask = np.abs(t_sorted.astype(float) - float(t_test)) <= half_window
        if not np.any(win_mask):
            # fallback: use all points if none in window
            win_mask = np.ones_like(t_sorted, dtype=bool)

        t_win = t_sorted[win_mask].astype(float, copy=False)
        S_win = S_sorted[win_mask].astype(float, copy=False)
        r_win = r_sorted[win_mask].astype(float, copy=False)
        tw_win = test_w_sorted[win_mask].astype(float, copy=False)

        # Localization weights: either deterministic tilde=t_test, or sample tilde~N(t_test,bw)
        # (use_sampled_tilde is kept for API compatibility but ignored; no sampling.)
        tilde = float(t_test)
        u = (t_win - tilde) / bw
        wloc_win = np.exp(-0.5 * u * u)

        # Combined finite weights (not normalized)
        w_comb = np.clip(r_win, 0.0, np.inf) * np.clip(tw_win, 0.0, np.inf) * np.clip(wloc_win, 0.0, np.inf)

        # Compute diagnostic q_alpha values for this test point using the plotted subset.
        # (Uses the same "cap to max score when ghost_norm>=alpha" convention as prediction.)
        alpha_eff = float(self.alpha if alpha is None else alpha)
        # test-point weights needed for ghost
        p_test_pt = float(np.clip(self._p_fn(np.array([k_test], dtype=int))[0], 1e-6, 1.0))
        r_ghost_pt = (1.0 - p_test_pt) / p_test_pt
        test_w_ghost_pt = float(np.clip(self._test_w_fn(np.array([k_test], dtype=int))[0], 0.0, np.inf))
        # localization ghost weight
        u0 = (float(t_test) - tilde) / bw
        wloc_ghost_pt = float(np.exp(-0.5 * u0 * u0))
        w_ghost_pt = float(np.clip(wloc_ghost_pt * r_ghost_pt * test_w_ghost_pt, 0.0, np.inf))

        def _weighted_qalpha(S_vals: np.ndarray, w_fin_vals: np.ndarray, w_ghost_val: float) -> float:
            """Compute capped weighted (1-alpha) quantile with ghost-at-infinity convention on a subset."""
            w_fin_vals = np.clip(np.asarray(w_fin_vals, dtype=float), 0.0, np.inf)
            nz = w_fin_vals > 0
            if not np.any(nz):
                return float("nan")
            S_fin = np.asarray(S_vals, dtype=float)[nz]
            w_fin_nz = w_fin_vals[nz]
            total_w = float(np.sum(w_fin_nz) + float(np.clip(w_ghost_val, 0.0, np.inf)))
            if total_w <= 0.0 or not np.isfinite(total_w):
                return float("nan")
            ghost_norm = float(np.clip(w_ghost_val, 0.0, np.inf)) / total_w
            if ghost_norm >= alpha_eff:
                return float(np.max(S_fin))
            ord_s = np.argsort(S_fin)
            S_s = S_fin[ord_s]
            w_s = (w_fin_nz[ord_s] / total_w)
            cumw = np.cumsum(w_s)
            idx = int(np.searchsorted(cumw, 1.0 - alpha_eff, side="left"))
            return float(S_s[idx]) if idx < int(S_s.size) else float(S_s[-1])

        # Combined-weights quantile (r * kernel * test_w)
        q_alpha_all = _weighted_qalpha(S_win, w_comb, w_ghost_pt)
        # Localization-only quantile (kernel weights only)
        q_alpha_loc = _weighted_qalpha(S_win, wloc_win, wloc_ghost_pt)

        # Choose x-axis and sort for readable plotting
        x_axis_norm = str(x_axis).strip().lower()
        if x_axis_norm in ("t", "time"):
            x_vals = t_win
            x_label = "calibration time t (within window)"
            ord_x = np.argsort(x_vals)
        elif x_axis_norm in ("s_cal", "s", "score", "residual", "residual_score"):
            x_vals = S_win
            x_label = "calibration score S_cal (for points in window)"
            ord_x = np.argsort(x_vals)
        else:
            raise ValueError("x_axis must be 't' or 'S_cal'")

        x_vals = x_vals[ord_x]
        t_win = t_win[ord_x]
        S_win = S_win[ord_x]
        r_win = r_win[ord_x]
        tw_win = tw_win[ord_x]
        wloc_win = wloc_win[ord_x]
        w_comb = w_comb[ord_x]

        if scatter_time_score:
            cb = str(color_by).strip().lower()
            comp = {
                "missingness": (r_win, "missingness weight r"),
                "localization": (wloc_win, "localization weight (kernel)"),
                "test": (tw_win, "test-sampling weight"),
                "combined": (w_comb, "combined weight"),
            }

            if cb in ("all", "4", "both"):
                # 2x2 grid showing all four components
                fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True, sharey=True)
                axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
                keys = ["missingness", "localization", "test", "combined"]
                for ax, k in zip(axes_list, keys):
                    cvals, ctitle = comp[k]
                    sc = ax.scatter(
                        t_win,
                        S_win,
                        c=cvals,
                        cmap="viridis",
                        s=26,
                        edgecolors="none",
                    )
                    q_line = q_alpha_loc if k == "localization" else q_alpha_all
                    if np.isfinite(q_line):
                        ax.axhline(q_line, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
                    ax.set_title(ctitle)
                    ax.grid(True, alpha=0.25)
                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("weight")

                for ax in axes[1, :]:
                    ax.set_xlabel("calibration time t (within window)")
                for ax in axes[:, 0]:
                    ax.set_ylabel("calibration score S_cal")

                fig.suptitle(
                    f"Weight scatter for test k={k_test} (i={i_test}, j={j_test}, t={t_test}), "
                    f"bw={self._bandwidth:g}, window=±{half_window:g}, tilde={tilde:.2f}, "
                    f"q_all={q_alpha_all:.3f}, q_loc={q_alpha_loc:.3f}, n={int(t_win.size)}",
                    y=0.98,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.96))

                if save_path:
                    fig.savefig(save_path, dpi=160, bbox_inches="tight")
                if show:
                    plt.show()
                return fig, list(axes_list)

            # Single-component scatter
            if cb in ("missingness", "r"):
                cvals, ctitle = comp["missingness"]
            elif cb in ("localization", "kernel", "w_loc"):
                cvals, ctitle = comp["localization"]
            elif cb in ("test", "test_w", "testsampling"):
                cvals, ctitle = comp["test"]
            elif cb in ("combined",):
                cvals, ctitle = comp["combined"]
            else:
                raise ValueError("color_by must be one of {'missingness','localization','test','combined','all'}")

            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            sc = ax.scatter(t_win, S_win, c=cvals, cmap="viridis", s=28, edgecolors="none")
            q_line = q_alpha_loc if cb in ("localization", "kernel", "w_loc") else q_alpha_all
            if np.isfinite(q_line):
                ax.axhline(q_line, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.set_xlabel("calibration time t (within window)")
            ax.set_ylabel("calibration score S_cal")
            ax.grid(True, alpha=0.25)
            ax.set_title(f"S_cal vs time, colored by {ctitle}")
            fig.colorbar(sc, ax=ax).set_label("weight")
            fig.suptitle(
                f"Weight scatter for test k={k_test} (i={i_test}, j={j_test}, t={t_test}), "
                f"bw={self._bandwidth:g}, window=±{half_window:g}, tilde={tilde:.2f}, "
                f"q_all={q_alpha_all:.3f}, q_loc={q_alpha_loc:.3f}, n={int(t_win.size)}",
                y=1.02,
            )
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=160, bbox_inches="tight")
            if show:
                plt.show()
            return fig, [ax]

        titles = [
            "Missingness weight r",
            "Localization weight (kernel)",
            "Test-sampling weight",
            "Combined weight r × kernel × test_w",
        ]
        ys = [r_win, wloc_win, tw_win, w_comb]

        fig, axes = plt.subplots(nrows=len(ys), ncols=1, figsize=(9.5, 2.2 * len(ys)), sharex=True)
        if len(ys) == 1:
            axes = [axes]

        for ax, y, ttl in zip(axes, ys, titles):
            ax.plot(x_vals, y, marker="o", linestyle="-", linewidth=1.2, markersize=3.5)
            ax.set_ylabel("weight")
            ax.set_title(ttl)
            ax.grid(True, axis="y", alpha=0.25)

        # If plotting against S_cal, add a vertical marker at q_alpha
        if x_axis_norm in ("s_cal", "s", "score", "residual", "residual_score"):
            for ax, ttl in zip(axes, titles):
                q_line = q_alpha_loc if "Localization" in str(ttl) else q_alpha_all
                if np.isfinite(q_line):
                    ax.axvline(q_line, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

        axes[-1].set_xlabel(x_label)

        fig.suptitle(
            f"Weight diagnostics for test k={k_test} (i={i_test}, j={j_test}, t={t_test}), "
            f"bw={self._bandwidth:g}, window=±{half_window:g}, tilde={tilde:.2f}, "
            f"q_all={q_alpha_all:.3f}, q_loc={q_alpha_loc:.3f}, n={int(t_win.size)}",
            y=1.02,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes

def weighted_split_conformal_prediction(
    Y_cal,
    Y_cal_hat,
    Y_test,
    Y_test_hat,
    test_mask_indices,
    p_cal,
    p_test,
    alpha: float = 0.05,
    use_numpy: bool = False,
):
    """
    Weighted split conformal prediction for the test set.
    Reweights absolute residuals by missingness odds and includes a test-specific ghost mass.
    """
    if use_numpy:
        S_cal = np.abs(np.asarray(Y_cal) - np.asarray(Y_cal_hat))  # (n_cal,)
        y_test_vals = np.asarray(Y_test)
        y_test_hat_vals = np.asarray(Y_test_hat)
    else:
        S_cal = np.abs(Y_cal.data - Y_cal_hat.data)  # (n_cal,)
        y_test_vals = Y_test.data
        y_test_hat_vals = Y_test_hat.data
    r_cal = (1.0 - p_cal) / p_cal                # (n_cal,)
    
    r_ghost = (1.0 - p_test) / p_test            # (n_test,)

    n_test = int(y_test_vals.shape[0])
    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    in_interval = np.empty(n_test, dtype=bool)
    ifinf = 0

    # Pre-sort calibration scores once
    order_cal = np.argsort(S_cal)
    S_sorted_all = S_cal[order_cal]
    r_cal_sorted = r_cal[order_cal]

    for j in range(n_test):
        all_r = np.concatenate([r_cal_sorted, np.array([r_ghost[j]])])
        w = all_r / np.sum(all_r)
        # weights for finite scores are first n_cal entries
        w_sorted = w[:-1]
        cumw = np.cumsum(w_sorted)
        idx = np.searchsorted(cumw, 1 - alpha, side="left")
        if idx < S_sorted_all.shape[0]:
            q_alpha = S_sorted_all[idx]
        else:
            q_alpha = np.inf
            ifinf += 1
        lower_bound[j] = y_test_hat_vals[j] - q_alpha
        upper_bound[j] = y_test_hat_vals[j] + q_alpha
        in_interval[j] = (y_test_vals[j] >= lower_bound[j]) and (y_test_vals[j] <= upper_bound[j])

    overall_coverage = float(np.mean(in_interval)) if n_test > 0 else float("nan")

    patient_coverages = {}
    patient_ids = test_mask_indices[:, 0]
    for pid in np.unique(patient_ids):
        inds = np.where(patient_ids == pid)[0]
        if len(inds) > 0:
            patient_coverages[int(pid)] = float(np.mean(in_interval[inds]))

    return lower_bound, upper_bound, overall_coverage, patient_coverages, ifinf / max(1, len(test_mask_indices))


def kernel(X_center, X, type: str = "gaussian", bandwidth: float = 10):
    """
    Compute kernel density values for each point in X around X_center.
    """
    if type == "gaussian":
        return norm.pdf(X, loc=X_center, scale=bandwidth)
    elif type == "uniform":
        return uniform.pdf(X, loc=X_center - bandwidth, scale=2 * bandwidth)
    else:
        raise ValueError("Unsupported kernel type")


def kernel_sampler(X_center, type: str = "gaussian", bandwidth: float = 10):
    """
    Sample from a kernel centered at X_center and return a callable H(x, x') density.
    """
    if type == "gaussian":
        return np.random.normal(loc=X_center, scale=bandwidth), functools.partial(kernel, type=type, bandwidth=bandwidth)
    elif type == "uniform":
        return np.random.uniform(low=X_center - bandwidth, high=X_center + bandwidth), functools.partial(kernel, type=type, bandwidth=bandwidth)
    else:
        raise ValueError("Unsupported kernel type")


def localized_weighted_split_conformal_prediction(
    Y_cal,
    Y_cal_hat,
    Y_test,
    Y_test_hat,
    test_mask_indices,
    p_cal,
    p_test,
    H_samp,
    alpha: float = 0.05,
    use_numpy: bool = True,
    t_cal: np.ndarray = None,
    t_test: np.ndarray = None,
):
    """
    Localized weighted split‐conformal with RLCP‐style localization.
    Combines missingness-odds reweighting with localization via kernel H.
    """
    if use_numpy:
        S_cal = np.abs(np.asarray(Y_cal) - np.asarray(Y_cal_hat))
        y_test_vals = np.asarray(Y_test)
        y_test_hat_vals = np.asarray(Y_test_hat)
    else:
        S_cal = np.abs(Y_cal.data - Y_cal_hat.data)  # (n_cal,)
        y_test_vals = Y_test.data
        y_test_hat_vals = Y_test_hat.data
    n_cal = int(S_cal.shape[0])
    n_test = int(y_test_vals.shape[0])

    r_cal = (1.0 - p_cal) / p_cal  # (n_cal,)

    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    in_interval = np.empty(n_test, dtype=bool)
    ifinf = 0

    # Pre-extract coords time axes
    if use_numpy:
        if t_cal is None or t_test is None:
            raise ValueError("When use_numpy=True, provide t_cal and t_test arrays for time coordinates.")
        t_cal_arr = np.asarray(t_cal)
        t_test_arr = np.asarray(t_test)
    else:
        t_cal_arr = Y_cal.coords[2, :]
        t_test_arr = Y_test.coords[2, :]

    for j in range(n_test):
        tilde_X, H = H_samp(X_center=t_test_arr[j])
        w_loc_cal = H(X=t_cal_arr, X_center=tilde_X)
        w_loc_ghost = H(X=t_test_arr[j], X_center=tilde_X)

        r_ghost = (1.0 - p_test[j]) / p_test[j]
        w_tilde_cal = w_loc_cal * r_cal
        w_tilde_ghost = w_loc_ghost * r_ghost

        all_w = np.concatenate([w_tilde_cal, np.array([w_tilde_ghost])])
        w_norm = all_w / np.sum(all_w)

        order = np.argsort(S_cal)
        S_sorted = S_cal[order]
        w_sorted = w_norm[order]
        cumw = np.cumsum(w_sorted)

        idx = np.searchsorted(cumw, 1 - alpha, side="left")
        if idx < n_cal:
            q_alpha = S_sorted[idx]
        else:
            q_alpha = S_sorted[-1]
            ifinf += 1

        y_hat_j = y_test_hat_vals[j]
        lower_bound[j] = y_hat_j - q_alpha
        upper_bound[j] = y_hat_j + q_alpha

        y_true_j = y_test_vals[j]
        in_interval[j] = (y_true_j >= lower_bound[j]) and (y_true_j <= upper_bound[j])

    overall_coverage = float(np.mean(in_interval)) if n_test > 0 else float("nan")
    patient_coverages = {}
    patient_ids = test_mask_indices[:, 0]
    for pid in np.unique(patient_ids):
        inds = np.where(patient_ids == pid)[0]
        if len(inds) > 0:
            patient_coverages[int(pid)] = float(np.mean(in_interval[inds]))

    return lower_bound, upper_bound, overall_coverage, patient_coverages, ifinf / max(1, n_test)

