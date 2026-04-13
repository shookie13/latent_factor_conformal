import numpy as np
from typing import Iterable, Optional, List, Tuple, Dict, Any

def miss_pool(O: np.ndarray, A: np.ndarray) -> np.ndarray:
    return (O * A) != 1

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _eligible_indices(mask_bool: np.ndarray, n: int, seed: int) -> np.ndarray:
    I, J, T = mask_bool.shape
    eligible = np.flatnonzero(mask_bool.ravel())
    if eligible.size == 0:
        return np.array([], dtype=int)
    r = rng(seed)
    if n > eligible.size:
        idx = np.arange(eligible.size)
    else:
        idx = r.integers(0, eligible.size, size=n)
    return eligible[idx]


def test_indices_marginal(O: np.ndarray, A: np.ndarray, n: int, seed: int =1) -> np.ndarray:
    Mmask = miss_pool(O, A)
    return _eligible_indices(Mmask, n=n, seed=seed + 101)

def test_indices_conditional(
    O: np.ndarray,
    A: np.ndarray,
    p: np.ndarray,
    subset: str,
    n: int,
    seed: int = 1,
    subject_idx: Optional[np.ndarray] = None,
    base_pool: str = "miss_pool",
) -> np.ndarray:
    I, J, T = O.shape
    base_pool = str(base_pool).lower()
    if base_pool == "miss_pool":
        Mmask = miss_pool(O, A)
    elif base_pool in {"observed_missing", "o_and_not_a", "o&~a"}:
        # Restrict the test pool to points that are observed-by-design but missing under A.
        Mmask = np.asarray(O, dtype=bool) & (~np.asarray(A, dtype=bool))
    else:
        raise ValueError(
            "base_pool must be one of {'miss_pool', 'observed_missing', 'o_and_not_a', 'o&~a'}"
        )
    subset = subset.lower()

    # Select which subjects (i indices) to condition on
    if subject_idx is None:
        subj_idx_arr = np.arange(I, dtype=int)
    else:
        subj_idx_arr = np.asarray(subject_idx, dtype=int)

    # Mask over (I,J,T) for selected subjects
    I_mask = np.zeros((I, 1, 1), dtype=bool)
    I_mask[subj_idx_arr, 0, 0] = True
    I_mask = np.broadcast_to(I_mask, (I, J, T))

    # A1: test points whose abs residual >= q75(calibration abs residuals)
    if p is not None:
        # Calibration residuals

        # Use only selected subjects for the calibration quantile
        q25 = np.nanquantile(p[subj_idx_arr, :, :], 0.25)
        q75 = np.nanquantile(p[subj_idx_arr, :, :], 0.75)

        A1 = p >= q75
        # A2: low-absolute-residual region (bottom quartile of calibration abs residuals)
        A2 = p <= q25
    else:
        # Fallback: no residual information → treat all as A1
        A1 = np.ones((I, J, T), dtype=bool)
        # Without residuals we cannot form A2 via q25; default to all points.
        A2 = np.ones((I, J, T), dtype=bool)

    if subset == "a1":
        eligible = Mmask & A1 & I_mask
        return _eligible_indices(eligible, n=n, seed=seed + 107)
    if subset == "a2":
        eligible = Mmask & A2 & I_mask
        return _eligible_indices(eligible, n=n, seed=seed + 109)
    if subset == "mixture_a1_a2" or subset == "mixture_a2_a3":
        r = rng(seed + 113)
        eligible_A1 = np.flatnonzero((Mmask & A1 & I_mask).ravel())
        eligible_A2 = np.flatnonzero((Mmask & A2 & I_mask).ravel())
        if eligible_A1.size == 0 and eligible_A2.size == 0:
            return np.array([], dtype=int)
        picks = r.random(size=n)
        out = np.empty(n, dtype=int)
        for idx, u in enumerate(picks):
            if u < 0.7 and eligible_A2.size > 0:
                out[idx] = eligible_A2[r.integers(0, eligible_A2.size)]
            elif eligible_A1.size > 0:
                out[idx] = eligible_A1[r.integers(0, eligible_A1.size)]
            elif eligible_A2.size > 0:
                out[idx] = eligible_A2[r.integers(0, eligible_A2.size)]
            else:
                out[idx] = eligible_A1[r.integers(0, eligible_A1.size)]
        return out
    return test_indices_marginal(O, A, n=n)

def consecutive_time_windows(T: int, K: int) -> List[Tuple[int, int]]:
    """
    Partition {0,1,...,T-1} into K consecutive (nearly equal-sized) windows.

    Returns a list of (start, end) pairs with 0 <= start < end <= T, covering [0, T).
    If K > T, we cap K at T (so windows are singletons).
    """
    T = int(T)
    K = int(K)
    if T <= 0:
        return []
    if K <= 0:
        raise ValueError("K must be positive.")
    if K > T:
        K = T

    splits = np.array_split(np.arange(T, dtype=int), K)
    windows: List[Tuple[int, int]] = []
    for arr in splits:
        if arr.size == 0:
            continue
        windows.append((int(arr[0]), int(arr[-1]) + 1))
    return windows


def mean_abs_window_coverage_gap(
    idx_flat: np.ndarray,
    covered: np.ndarray,
    *,
    I: int,
    J: int,
    T: int,
    K: int,
    target: float,
    j_filter: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Window-local coverage evaluation.

    For each subject i, split its timepoints into K consecutive windows. Within each window,
    compute empirical coverage over the provided points (idx_flat / covered), then compute
    abs(target - empirical_coverage). Finally average this gap over all (i, window) pairs.

    Notes:
    - idx_flat uses the repository's row-major flattening: k = ((i*J)+j)*T + t.
    - If a window has no points for a given subject, it is skipped.

    Returns dict with:
      - mean_abs_gap: float
      - n_windows_used: int
      - windows: list of (start,end)
    """
    idx_flat = np.asarray(idx_flat, dtype=np.int64).ravel()
    covered = np.asarray(covered, dtype=bool).ravel()
    if idx_flat.size != covered.size:
        raise ValueError("idx_flat and covered must have the same length.")

    I = int(I)
    J = int(J)
    T = int(T)
    K = int(K)
    if I <= 0 or J <= 0 or T <= 0:
        return {"mean_abs_gap": float("nan"), "n_windows_used": 0, "windows": []}

    windows = consecutive_time_windows(T, K)
    if len(windows) == 0 or idx_flat.size == 0:
        return {"mean_abs_gap": float("nan"), "n_windows_used": 0, "windows": windows}

    JT = J * T
    i_idx = (idx_flat // JT).astype(np.int64, copy=False)
    rem = (idx_flat - i_idx * JT).astype(np.int64, copy=False)
    j_idx = (rem // T).astype(np.int64, copy=False)
    t_idx = (rem - j_idx * T).astype(np.int64, copy=False)

    if j_filter is not None:
        jf = int(j_filter)
        keep = (j_idx == jf)
        i_idx = i_idx[keep]
        t_idx = t_idx[keep]
        covered = covered[keep]

    if i_idx.size == 0:
        return {"mean_abs_gap": float("nan"), "n_windows_used": 0, "windows": windows}

    # Accumulate gap over all subject-window pairs that have at least one point.
    gaps: List[float] = []
    for i in range(I):
        mask_i = (i_idx == i)
        if not np.any(mask_i):
            continue
        ti = t_idx[mask_i]
        ci = covered[mask_i]
        for (a, b) in windows:
            mw = (ti >= a) & (ti < b)
            if not np.any(mw):
                continue
            emp = float(np.mean(ci[mw]))
#             print(len(mw))
            gaps.append(abs(float(target) - emp))

    if len(gaps) == 0:
        return {"mean_abs_gap": float("nan"), "n_windows_used": 0, "windows": windows}

    return {
        "mean_abs_gap": float(np.mean(np.asarray(gaps, dtype=float))),
        "n_windows_used": int(len(gaps)),
        "windows": windows,
    }