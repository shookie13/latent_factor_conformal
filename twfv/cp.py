"""
Conformal prediction utilities: unweighted, weighted, localized, and localized-weighted split CP.
Implemented in NumPy without external dependencies beyond the standard library.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional, Protocol, Tuple

import numpy as np


def unflatten(k: int, I: int, J: int, T: int) -> Tuple[int, int, int]:
    """
    Map a flattened index k into (i, j, t) using row-major order: k = i*(J*T) + j*T + t.
    """
    i = k // (J * T)
    rem = k % (J * T)
    j = rem // T
    t = rem % T
    return int(i), int(j), int(t)


class CPMethod(Protocol):
    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        ...

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        ...

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        ...

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        ...

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        ...


def _quantile_higher(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the split-conformal quantile level ceil((m+1)*(1-alpha))/m with "higher" interpolation.
    """
    m = scores.size
    if m == 0:
        return np.nan
    q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, q_level, interpolation="higher"))


class NaiveAbsoluteResidualCP:
    """
    Global split conformal using absolute residuals; expects external fitted means.

    Calibrates a single scalar q (across all i,t,j in the calibration set).
    """

    def __init__(self, shape_ref: np.ndarray, J: int, alpha: float = 0.1):
        """
        Args:
            shape_ref: any array with shape (I, T, ...); used only to infer I, T.
            J: number of channels.
            alpha: miscoverage level.
        """
        self.J = J
        self.alpha = alpha
        self.I = shape_ref.shape[0]
        self.T = shape_ref.shape[1]
        self.q_abs = float("nan")

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("No internal model. Use calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("Use predict_interval_from_fit(fit).")

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        # Note: X_cal_idx is accepted for API consistency, but not used here.
        res = np.abs(np.asarray(Y_cal_true, dtype=float) - np.asarray(Y_cal_fit, dtype=float))
        self.q_abs = _quantile_higher(res, self.alpha) if res.size > 0 else float("nan")

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        out = np.zeros((X_test_idx.size, 2), dtype=float)
        q = self.q_abs
        if not np.isfinite(q):
            q = 0.0
        y_fit = np.asarray(Y_test_fit, dtype=float)
        out[:, 0] = y_fit - q
        out[:, 1] = y_fit + q
        return out


class WeightedSplitConformalCP:
    """
    Split CP with importance weights r = (1-p)/p derived from missingness probabilities p_fn(idx).
    """

    def __init__(self, shape_ref: np.ndarray, J: int, alpha: float = 0.1, p_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Args:
            shape_ref: any array with shape (I, T, ...); used only to infer dimensions.
            J: number of channels.
            alpha: miscoverage level.
            p_fn: callable mapping flattened indices -> missingness probability in (0,1].
        """
        self.J = J
        self.alpha = alpha
        self._p_fn = p_fn or (lambda idx: np.full(idx.size, 0.5, dtype=float))
        self._S_sorted: Optional[np.ndarray] = None
        self._w_cal_sorted: Optional[np.ndarray] = None

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("No internal model. Use calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("Use predict_interval_from_fit(fit).")

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        S_cal = np.abs(Y_cal_true - Y_cal_fit)
        p_cal = np.clip(self._p_fn(np.asarray(X_cal_idx)), 1e-6, np.inf)
        w_cal = (1.0 - p_cal) / p_cal
        order = np.argsort(S_cal)
        self._S_sorted = S_cal[order]
        self._w_cal_sorted = w_cal[order]

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        if self._S_sorted is None or self._w_cal_sorted is None:
            raise RuntimeError("Call calibrate_from_fit before prediction.")
        X_test_idx = np.asarray(X_test_idx)
        p_test = np.clip(self._p_fn(X_test_idx), 1e-6, np.inf)
        r_ghost = (1.0 - p_test) / p_test
        out = np.zeros((X_test_idx.size, 2), dtype=float)
        S_sorted = self._S_sorted
        w_cal_sorted = self._w_cal_sorted

        for j in range(X_test_idx.size):
            all_w = np.concatenate([w_cal_sorted, np.array([r_ghost[j]])])
            w_norm = all_w / np.sum(all_w)
            w_cal_norm = w_norm[:-1]
            cumw = np.cumsum(w_cal_norm)
            idx = np.searchsorted(cumw, 1 - alpha, side="left")
            q_alpha = S_sorted[idx] if idx < S_sorted.size else np.inf
            yhat = float(Y_test_fit[j])
            out[j, 0] = yhat - q_alpha
            out[j, 1] = yhat + q_alpha
        return out


class LocalizedSplitConformalCP:
    """
    Split CP localized in time via a kernel sampler H_samp; no missingness weighting.
    """

    def __init__(self, shape_ref: np.ndarray, J: int, alpha: float = 0.1, H_samp: Optional[Callable[..., tuple]] = None):
        """
        Args:
            shape_ref: any array with shape (I, T, ...); used only to infer dimensions.
            J: number of channels.
            alpha: miscoverage level.
            H_samp: sampler returning (tilde_X, H) where H gives localization weights over times.
        """
        self.J = J
        self.alpha = alpha
        self.I = shape_ref.shape[0]
        self.T = shape_ref.shape[1]
        self._H_samp = H_samp or (lambda X_center: kernel_sampler(X_center=X_center, type="gaussian", bandwidth=10))
        self._S_cal: Optional[np.ndarray] = None
        self._t_cal: Optional[np.ndarray] = None

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("No internal model. Use calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("Use predict_interval_from_fit(fit).")

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        S_cal = np.abs(Y_cal_true - Y_cal_fit)
        t_cal = np.empty(X_cal_idx.size, dtype=int)
        for n, k in enumerate(X_cal_idx.tolist()):
            _, _, t = unflatten(int(k), self.I, self.J, self.T)
            t_cal[n] = t
        self._S_cal = S_cal
        self._t_cal = t_cal

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        if self._S_cal is None or self._t_cal is None:
            raise RuntimeError("Call calibrate_from_fit before prediction.")
        n_test = X_test_idx.size
        out = np.zeros((n_test, 2), dtype=float)
        t_test = np.empty(n_test, dtype=int)
        for n, k in enumerate(X_test_idx.tolist()):
            _, _, t = unflatten(int(k), self.I, self.J, self.T)
            t_test[n] = t

        S_cal = self._S_cal
        t_cal = self._t_cal
        for j in range(n_test):
            tilde_X, H = self._H_samp(X_center=t_test[j])
            w_loc_cal = H(X=t_cal, X_center=tilde_X)
            w_loc_ghost = H(X=t_test[j], X_center=tilde_X)
            all_w = np.concatenate([w_loc_cal, np.array([w_loc_ghost])])
            w_norm = all_w / np.sum(all_w)
            order = np.argsort(S_cal)
            S_sorted = S_cal[order]
            w_sorted = w_norm[order]
            cumw = np.cumsum(w_sorted)
            idx = np.searchsorted(cumw, 1 - alpha, side="left")
            q_alpha = S_sorted[idx] if idx < S_sorted.size else S_sorted[-1]
            yhat = float(Y_test_fit[j])
            out[j, 0] = yhat - q_alpha
            out[j, 1] = yhat + q_alpha
        return out


class LocalizedWeightedSplitConformalCP:
    """
    Split CP with both missingness-odds weighting and time localization.
    """

    def __init__(
        self,
        shape_ref: np.ndarray,
        J: int,
        alpha: float = 0.1,
        p_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        H_samp: Optional[Callable[..., tuple]] = None,
    ):
        """
        Args:
            shape_ref: any array with shape (I, T, ...); used only to infer dimensions.
            J: number of channels.
            alpha: miscoverage level.
            p_fn: callable mapping flattened indices -> missingness prob in (0,1].
            H_samp: sampler returning (tilde_X, H) where H gives localization weights over times.
        """
        self.J = J
        self.alpha = alpha
        self.I = shape_ref.shape[0]
        self.T = shape_ref.shape[1]
        self._p_fn = p_fn or (lambda idx: np.full(idx.size, 0.5, dtype=float))
        self._H_samp = H_samp or (lambda X_center: kernel_sampler(X_center=X_center, type="gaussian", bandwidth=10))
        self._S_cal: Optional[np.ndarray] = None
        self._t_cal: Optional[np.ndarray] = None
        self._r_cal: Optional[np.ndarray] = None

    def fit(self, X_obs_idx: np.ndarray, Y_obs: np.ndarray) -> None:
        raise NotImplementedError("No internal model. Use calibrate_from_fit().")

    def calibrate(self, X_cal_idx: np.ndarray, Y_cal: np.ndarray) -> None:
        raise NotImplementedError("Use calibrate_from_fit(true, fit).")

    def predict_interval(self, X_test_idx: np.ndarray, alpha: float) -> np.ndarray:
        raise NotImplementedError("Use predict_interval_from_fit(fit).")

    def calibrate_from_fit(self, X_cal_idx: np.ndarray, Y_cal_true: np.ndarray, Y_cal_fit: np.ndarray) -> None:
        S_cal = np.abs(Y_cal_true - Y_cal_fit)
        t_cal = np.empty(X_cal_idx.size, dtype=int)
        for n, k in enumerate(X_cal_idx.tolist()):
            _, _, t = unflatten(int(k), self.I, self.J, self.T)
            t_cal[n] = t
        p_cal = np.clip(self._p_fn(np.asarray(X_cal_idx)), 1e-6, 1.0)
        r_cal = (1.0 - p_cal) / p_cal
        self._S_cal = S_cal
        self._t_cal = t_cal
        self._r_cal = r_cal

    def predict_interval_from_fit(self, X_test_idx: np.ndarray, Y_test_fit: np.ndarray, alpha: float) -> np.ndarray:
        if self._S_cal is None or self._t_cal is None or self._r_cal is None:
            raise RuntimeError("Call calibrate_from_fit before prediction.")
        n_test = X_test_idx.size
        out = np.zeros((n_test, 2), dtype=float)
        t_test = np.empty(n_test, dtype=int)
        for n, k in enumerate(X_test_idx.tolist()):
            _, _, t = unflatten(int(k), self.I, self.J, self.T)
            t_test[n] = t

        p_test = np.clip(self._p_fn(np.asarray(X_test_idx)), 1e-6, 1.0)
        S_cal = self._S_cal
        t_cal = self._t_cal
        r_cal = self._r_cal

        for j in range(n_test):
            tilde_X, H = self._H_samp(X_center=t_test[j])
            w_loc_cal = H(X=t_cal, X_center=tilde_X)
            w_loc_ghost = H(X=t_test[j], X_center=tilde_X)
            w_cal = w_loc_cal * r_cal
            w_ghost = w_loc_ghost * ((1.0 - p_test[j]) / p_test[j])
            all_w = np.concatenate([w_cal, np.array([w_ghost])])
            w_norm = all_w / np.sum(all_w)
            order = np.argsort(S_cal)
            S_sorted = S_cal[order]
            w_sorted = w_norm[order]
            cumw = np.cumsum(w_sorted)
            idx = np.searchsorted(cumw, 1 - alpha, side="left")
            q_alpha = S_sorted[idx] if idx < S_sorted.size else S_sorted[-1]
            yhat = float(Y_test_fit[j])
            out[j, 0] = yhat - q_alpha
            out[j, 1] = yhat + q_alpha
        return out


def gaussian_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    z = (x - loc) / scale
    return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * scale)


def uniform_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    half = scale / 2.0
    return np.where((x >= loc - half) & (x <= loc + half), 1.0 / scale, 0.0)


def kernel(X_center, X, type: str = "gaussian", bandwidth: float = 10):
    if type == "gaussian":
        return gaussian_pdf(np.asarray(X, dtype=float), loc=X_center, scale=bandwidth)
    if type == "uniform":
        return uniform_pdf(np.asarray(X, dtype=float), loc=X_center, scale=2 * bandwidth)
    raise ValueError("Unsupported kernel type")


def kernel_sampler(X_center, type: str = "gaussian", bandwidth: float = 10):
    if type == "gaussian":
        sample = np.random.normal(loc=X_center, scale=bandwidth)
    elif type == "uniform":
        sample = np.random.uniform(low=X_center - bandwidth, high=X_center + bandwidth)
    else:
        raise ValueError("Unsupported kernel type")
    return sample, functools.partial(kernel, type=type, bandwidth=bandwidth)


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
    Vectorized weighted split conformal prediction (same as WeightedSplitConformalCP).
    """
    if use_numpy:
        S_cal = np.abs(np.asarray(Y_cal) - np.asarray(Y_cal_hat))
        y_test_vals = np.asarray(Y_test)
        y_test_hat_vals = np.asarray(Y_test_hat)
    else:
        S_cal = np.abs(Y_cal.data - Y_cal_hat.data)
        y_test_vals = Y_test.data
        y_test_hat_vals = Y_test_hat.data

    r_cal = (1.0 - p_cal) / p_cal
    r_ghost = (1.0 - p_test) / p_test

    order_cal = np.argsort(S_cal)
    S_sorted_all = S_cal[order_cal]
    r_cal_sorted = r_cal[order_cal]

    n_test = int(y_test_vals.shape[0])
    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    in_interval = np.empty(n_test, dtype=bool)
    ifinf = 0

    for j in range(n_test):
        all_r = np.concatenate([r_cal_sorted, np.array([r_ghost[j]])])
        w = all_r / np.sum(all_r)
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
    Vectorized localized + weighted split conformal prediction (same as LocalizedWeightedSplitConformalCP).
    """
    if use_numpy:
        S_cal = np.abs(np.asarray(Y_cal) - np.asarray(Y_cal_hat))
        y_test_vals = np.asarray(Y_test)
        y_test_hat_vals = np.asarray(Y_test_hat)
    else:
        S_cal = np.abs(Y_cal.data - Y_cal_hat.data)
        y_test_vals = Y_test.data
        y_test_hat_vals = Y_test_hat.data

    n_cal = int(S_cal.shape[0])
    n_test = int(y_test_vals.shape[0])
    r_cal = (1.0 - p_cal) / p_cal

    if use_numpy:
        if t_cal is None or t_test is None:
            raise ValueError("When use_numpy=True, provide t_cal and t_test.")
        t_cal_arr = np.asarray(t_cal)
        t_test_arr = np.asarray(t_test)
    else:
        t_cal_arr = Y_cal.coords[2, :]
        t_test_arr = Y_test.coords[2, :]

    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    in_interval = np.empty(n_test, dtype=bool)
    ifinf = 0

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

