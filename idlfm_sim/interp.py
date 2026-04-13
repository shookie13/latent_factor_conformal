import ruptures as rpt
import numpy as np


# ----------------------------
# NaN handling: linear interpolation + edge fill
# ----------------------------
def _interp_1d(y: np.ndarray) -> np.ndarray:
    """Return y with NaNs filled by linear interpolation (then edge-filled)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    x = np.arange(n)
    ok = np.isfinite(y)
    if not np.any(ok):
        # all missing -> return zeros (or raise if you prefer)
        return np.zeros_like(y, dtype=float)
    # linear interpolation over finite points
    y_filled = np.interp(x, x[ok], y[ok])
    return y_filled


def _interp_2d(Y: np.ndarray) -> np.ndarray:
    """Column-wise interpolate NaNs in a (T, J) signal."""
    Y = np.asarray(Y, dtype=float)
    T, J = Y.shape
    out = np.empty_like(Y, dtype=float)
    for j in range(J):
        out[:, j] = _interp_1d(Y[:, j])
    return out


# ----------------------------
# Build a ruptures detector
# ----------------------------
def _make_algo(
    algo: str,
    model: str = "l2",
    min_size: int = 5,
    jump: int = 1,
):
    """
    algo:
      - "pelt": penalized segmentation, use predict(pen=...)
      - "binseg": binary segmentation, use predict(n_bkps=...)
      - "window": window-based, use predict(n_bkps=...) or pen (depends on version)
    """
    algo = algo.lower()
    if algo == "pelt":
        return rpt.Pelt(model=model, min_size=min_size, jump=jump)  # fit -> predict(pen) :contentReference[oaicite:2]{index=2}
    if algo == "binseg":
        return rpt.Binseg(model=model, min_size=min_size, jump=jump)  # :contentReference[oaicite:3]{index=3}
    if algo == "window":
        return rpt.Window(model=model, min_size=min_size, jump=jump)
    raise ValueError(f"Unknown algo={algo!r}")


# ----------------------------
# Core: detect change points
# ----------------------------
def detect_changepoints_residual_IJT(
    residual_IJT: np.ndarray,
    *,
    mode: str = "per_ij",         # "per_ij" or "per_i"
    algo: str = "pelt",           # "pelt" | "binseg" | "window"
    model: str = "l2",            # common: "l2", "l1", "rbf", etc.
    pen: float = 10.0,            # used by pelt
    n_bkps: int = 5,              # used by binseg/window
    min_size: int = 5,
    jump: int = 1,
    preprocess: str = "interp",   # "interp" or "none"
) -> dict:
    """
    residual_IJT: array (I, J, T) with possible NaNs.

    Returns:
      if mode="per_ij":
        dict[(i, j)] -> list of breakpoints bkps (includes T at the end)
      if mode="per_i":
        dict[i] -> list of breakpoints bkps (includes T at the end)

    Notes:
      - ruptures returns a sorted list of segment end indices, typically including T as the last breakpoint.
        The actual change points are bkps[:-1]. :contentReference[oaicite:4]{index=4}
    """
    r = np.asarray(residual_IJT)
    if r.ndim != 3:
        raise ValueError(f"residual_IJT must be 3D (I,J,T), got shape {r.shape}")
    I, J, T = r.shape

    mode = mode.lower()
    preprocess = preprocess.lower()

    out = {}

    if mode == "per_ij":
        for i in range(I):
            for j in range(J):
                y = r[i, j, :]
                if preprocess == "interp":
                    y_use = _interp_1d(y)
                elif preprocess == "none":
                    # WARNING: NaNs can cause unexpected results in ruptures
                    y_use = np.asarray(y, dtype=float)
                else:
                    raise ValueError(f"Unknown preprocess={preprocess!r}")

                # ruptures accepts (n_samples,) or (n_samples, n_features) :contentReference[oaicite:5]{index=5}
                algo_obj = _make_algo(algo=algo, model=model, min_size=min_size, jump=jump)
                algo_obj.fit(y_use)

                if algo.lower() == "pelt":
                    bkps = algo_obj.predict(pen=pen)  # :contentReference[oaicite:6]{index=6}
                else:
                    bkps = algo_obj.predict(n_bkps=n_bkps)

                out[(i, j)] = bkps

        return out

    if mode == "per_i":
        for i in range(I):
            Y = r[i, :, :].T  # (T, J)
            if preprocess == "interp":
                Y_use = _interp_2d(Y)
            elif preprocess == "none":
                Y_use = np.asarray(Y, dtype=float)
            else:
                raise ValueError(f"Unknown preprocess={preprocess!r}")

            algo_obj = _make_algo(algo=algo, model=model, min_size=min_size, jump=jump)
            algo_obj.fit(Y_use)

            if algo.lower() == "pelt":
                bkps = algo_obj.predict(pen=pen)
            else:
                bkps = algo_obj.predict(n_bkps=n_bkps)

            out[i] = bkps

        return out

    raise ValueError(f"Unknown mode={mode!r}. Use 'per_ij' or 'per_i'.")


def bkps_to_changepoints(bkps: list[int]) -> np.ndarray:
    """Convert ruptures bkps list to change-point indices (exclude last endpoint)."""
    if bkps is None:
        return np.array([], dtype=int)
    bkps = np.asarray(bkps, dtype=int)
    if bkps.size == 0:
        return bkps
    return bkps[:-1]

def make_piecewise_constant_P_hat(
    bkps_ij,
    shape,                 # (I, J, T)
    *,
    seed=42,
    p_low=0.2,
    p_high=0.8,
    clip_eps=1e-6,
    sampler=None,
    variance_source=None,   # optional array of shape (I, J, T)
    variance_mode="linear", # "linear" or "inverse"
    variance_eps=1e-8,
    ddof=0,
):
    """
    Build P_hat of shape (I, J, T) such that for each (i, j),
    P_hat[i, j, :] is constant within each changepoint window and
    can change across windows.

    If variance_source is provided, then for each fixed (i, j),
    each segment/window gets a probability based on the variance of
    variance_source[i, j, start:end] relative to the other windows
    of the same (i, j).

    Parameters
    ----------
    bkps_ij : dict
        Example:
        {
            (0, 0): [T],
            (0, 1): [T],
            (0, 2): [t1, T],
            (0, 3): [t1, t2, T],
            ...
        }
        The trailing T is ignored.
    shape : tuple
        (I, J, T)
    seed : int
        RNG seed.
    p_low, p_high : float
        Range for segment-wise probabilities.
    clip_eps : float
        Final probabilities are clipped to [clip_eps, 1 - clip_eps].
    sampler : callable or None
        Optional custom sampler with signature:
            sampler(rng, n_segments, i, j) -> array of shape (n_segments,)
        Used only if variance_source is None.
    variance_source : np.ndarray or None
        Optional array of shape (I, J, T). If provided, use within-window
        variance to assign probabilities.
    variance_mode : str
        "linear"  -> higher variance => higher p_hat
        "inverse" -> higher variance => lower p_hat
    variance_eps : float
        Small constant to avoid divide-by-zero if all window variances match.
    ddof : int
        Degrees of freedom for np.nanvar.
    """
    I, J, T = shape
    rng = np.random.default_rng(seed)
    P_hat = np.empty((I, J, T), dtype=float)

    if sampler is None:
        def sampler(rng, n_segments, i, j):
            return rng.uniform(p_low, p_high, size=n_segments)

    if variance_source is not None:
        variance_source = np.asarray(variance_source, dtype=float)
        if variance_source.shape != (I, J, T):
            raise ValueError(
                f"variance_source must have shape {(I, J, T)}, got {variance_source.shape}"
            )

    for i in range(I):
        for j in range(J):
            bkps = bkps_ij.get((i, j), [T])

            # Keep only actual changepoints in (0, T); omit trailing T
            cps = sorted({int(b) for b in bkps if 0 < int(b) < T})

            # Segment boundaries: [0, cp1), [cp1, cp2), ..., [last_cp, T)
            edges = [0] + cps + [T]
            n_segments = len(edges) - 1

            if variance_source is not None:
                seg_var = np.empty(n_segments, dtype=float)

                for k, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
                    x = variance_source[i, j, start:end]
                    if x.size <= 1:
                        seg_var[k] = 0.0
                    else:
                        # Use mean squared error (around 0) instead of variance around the segment mean
                        seg_var[k] = float(np.nanmean(x ** 2))

                vmin = float(np.nanmin(seg_var))
                vmax = float(np.nanmax(seg_var))

                if vmax - vmin <= variance_eps:
                    seg_p = np.full(n_segments, 0.5 * (p_low + p_high), dtype=float)
                else:
                    scaled = (seg_var - vmin) / (vmax - vmin)
                    if variance_mode == "linear":
                        seg_p = p_low + scaled * (p_high - p_low)
                    elif variance_mode == "inverse":
                        seg_p = p_high - scaled * (p_high - p_low)
                    else:
                        raise ValueError("variance_mode must be 'linear' or 'inverse'")
            else:
                seg_p = np.asarray(sampler(rng, n_segments, i, j), dtype=float)
                if seg_p.shape != (n_segments,):
                    raise ValueError(
                        f"sampler must return shape ({n_segments},), got {seg_p.shape}"
                    )

            seg_p = np.clip(seg_p, clip_eps, 1.0 - clip_eps)

            for k, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
                P_hat[i, j, start:end] = seg_p[k]

    return P_hat