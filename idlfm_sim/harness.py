from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Callable, Optional, Tuple
import numpy as np

from .config import SimConfig
from .latent import simulate_theta
from .loadings import simulate_loadings
from .generator import simulate_Y_full
from .measurement import build_multi_resolution_masks
from .mar import sample_mar
from .splits import train_cal_split
from .missingness import estimate_missingness_kernel
from .testsamples import (
    test_indices_marginal,
    test_indices_conditional,
    mean_abs_window_coverage_gap,
)
from typing import Optional


def _build_observed_arrays(Y_full: np.ndarray, mask: np.ndarray):
    idx = np.flatnonzero(mask.ravel())
    return idx, Y_full.ravel()[idx]


def _split_train_fit_loc_masks(
    shape: Tuple[int, int, int],
    train_idx: np.ndarray,
    *,
    seed: int,
    frac_fit: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split training observed points into fit/loc boolean masks."""
    I, J, T = (int(shape[0]), int(shape[1]), int(shape[2]))
    train_idx = np.asarray(train_idx, dtype=np.int64).ravel()
    fit_mask = np.zeros((I, J, T), dtype=bool)
    loc_mask = np.zeros((I, J, T), dtype=bool)
    if train_idx.size == 0:
        return fit_mask, loc_mask

    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx.size)
    n_fit = int(np.clip(np.floor(float(frac_fit) * train_idx.size), 1, train_idx.size))
    fit_idx = train_idx[perm[:n_fit]]
    loc_idx = train_idx[perm[n_fit:]]
    fit_mask.ravel()[fit_idx] = True
    loc_mask.ravel()[loc_idx] = True
    return fit_mask, loc_mask


def _localizer_features_from_flat_idx(
    idx_flat: np.ndarray,
    *,
    I: int,
    J: int,
    T: int,
    mean_pred_flat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feature map X -> R^d for residual-localizer training/prediction.

    Subject/channel indices are encoded as one-hot (dummy) variables, while time and
    optional fitted means remain continuous features.
    """
    idx_flat = np.asarray(idx_flat, dtype=np.int64).ravel()
    JT = int(J) * int(T)
    i = idx_flat // JT
    rem = idx_flat - i * JT
    j = rem // int(T)
    t = rem - j * int(T)

    n = int(idx_flat.size)
    feat_blocks: List[np.ndarray] = []

    # One-hot encode subject id i.
    feat_i = np.zeros((n, int(I)), dtype=float)
    if n > 0:
        feat_i[np.arange(n), i.astype(int, copy=False)] = 1.0
    feat_blocks.append(feat_i)

    # One-hot encode channel id j.
    feat_j = np.zeros((n, int(J)), dtype=float)
    if n > 0:
        feat_j[np.arange(n), j.astype(int, copy=False)] = 1.0
    feat_blocks.append(feat_j)

    # Continuous time feature.
    feat_blocks.append((t.astype(float) / max(int(T) - 1, 1)).reshape(-1, 1))

    # Optional continuous fitted-mean feature.
    if mean_pred_flat is not None:
        feat_blocks.append(np.asarray(mean_pred_flat, dtype=float)[idx_flat].reshape(-1, 1))

    return np.concatenate(feat_blocks, axis=1).astype(float, copy=False)


def _build_learned_localizer_repr_fn(
    *,
    theta: np.ndarray,
    Y_full: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    frac_fit: float = 0.5,
    model_kind: str = "rf",
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Learn a scalar representation g(X) that predicts |Y - f_fit(X)| on a held-out
    localization subset of the training observations.
    """
    I, J, T = Y_full.shape
    fit_mask, loc_mask = _split_train_fit_loc_masks((I, J, T), train_idx, seed=seed, frac_fit=frac_fit)
    if not np.any(fit_mask) or not np.any(loc_mask):
        return None

    # Auxiliary predictor \hat f fitted on D_fit only.
    beta0_loc, beta_loc = _fit_linear_mean_by_stream(theta, Y_full, fit_mask)
    mean_loc = _predict_linear_mean(theta, beta0_loc, beta_loc)
    loc_idx = np.flatnonzero(loc_mask.ravel()).astype(np.int64)
    if loc_idx.size < 8:
        return None

    X_loc = _localizer_features_from_flat_idx(
        loc_idx,
        I=I,
        J=J,
        T=T,
        mean_pred_flat=mean_loc.ravel(),
    )
    y_loc = np.abs(Y_full.ravel()[loc_idx] - mean_loc.ravel()[loc_idx]).astype(float, copy=False)

    model_kind = str(model_kind).lower()
    model = None
    try:
        if model_kind in {"rf", "random_forest", "randomforest"}:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore

            model = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=-1,
            )
        else:
            from sklearn.linear_model import LinearRegression  # type: ignore

            model = LinearRegression()
        model.fit(X_loc, y_loc)
    except Exception:
        return None

    mean_loc_flat = mean_loc.ravel().astype(float, copy=False)

    def _repr_fn(idx_query: np.ndarray, *, _model=model, _mean_flat=mean_loc_flat) -> np.ndarray:
        idx_query = np.asarray(idx_query, dtype=np.int64).ravel()
        Xq = _localizer_features_from_flat_idx(
            idx_query,
            I=I,
            J=J,
            T=T,
            mean_pred_flat=_mean_flat,
        )
        z = np.asarray(_model.predict(Xq), dtype=float).reshape(-1, 1)
        return z

    return _repr_fn

def window_local_abs_coverage_gap(
    *,
    intervals: np.ndarray,
    y_true: np.ndarray,
    idx_flat: np.ndarray,
    I: int,
    J: int,
    T: int,
    K: int,
    target: float,
    j_filter: Optional[int] = None,
) -> float:
    """
    Compute the mean absolute window-local coverage gap:
      - per subject i, split time into K consecutive windows
      - within each window compute empirical coverage
      - take abs(target - empirical), then average over all (i, window) with data
    """
    if intervals.size == 0 or y_true.size == 0 or idx_flat.size == 0:
        return float("nan")
    L = intervals[:, 0]
    U = intervals[:, 1]
    covered = (y_true >= L) & (y_true <= U)
    res = mean_abs_window_coverage_gap(
        idx_flat,
        covered,
        I=I,
        J=J,
        T=T,
        K=K,
        target=target,
        j_filter=j_filter,
    )
    return float(res["mean_abs_gap"])

def _test_weight_fn_from_flat(flat_w: np.ndarray):
    flat_w = np.asarray(flat_w, dtype=float)
    return lambda idx, arr=flat_w: arr[np.asarray(idx, dtype=int)]

def _build_block_test_weights(
    res: np.ndarray,
    p: np.array,
    mix_prob_A2: float = 0.7,
) -> Dict[str, np.ndarray]:
    """
    Build (possibly unnormalized) test-sampling weights p^{test}_k for each block.

    Important: these weights are intended to represent the *test sampling distribution* F
    (up to a constant), and are NOT required to sum to 1. They are later normalized together
    with calibration weights and the ghost mass inside the CP methods.
    """
    I, J, T = res.shape
    t = np.arange(T)[None, None, :]  # (1,1,T)

    # marginal: uniform over everything (constant weights)
    w_marg = np.ones((I, J, T), dtype=float)
    # Conditional sets based on calibration residual threshold, applied to TEST residuals
    # cal_residuals is expected to be NaN outside calibration points (so nanquantile ignores them).
    q25 = float(np.nanquantile(p, 0.25))
    q75 = float(np.nanquantile(p, 0.75))
    A1 = p >= q75
    # A2: low-absolute-residual region (bottom quartile of calibration abs residuals)
    A2 = p <= q25

    w_a1 = A1.astype(float)
    w_a2 = A2.astype(float)
    # Mixture: mix of A1 and A2 (keep parameter name: mix_prob_A2 = P(select A2)).
    #
    # IMPORTANT: the sampler for the mixture is "choose subset (A2 vs A1) then sample uniformly
    # within that subset". To represent that as a density/weight function p^{test}(x) (up to a
    # constant), we must normalize by subset sizes; otherwise the mixture gets biased toward the
    # larger set and won't behave like the individual blocks.
    s1 = float(np.sum(w_a1))
    s2 = float(np.sum(w_a2))
    if s1 <= 0.0 and s2 <= 0.0:
        w_mix = np.zeros_like(w_a1)
    elif s1 <= 0.0:
        w_mix = w_a2
    elif s2 <= 0.0:
        w_mix = w_a1
    else:
        w_mix = (mix_prob_A2 * (w_a2 / s2)) + ((1.0 - mix_prob_A2) * (w_a1 / s1))

    return {
        "marginal": w_marg.ravel(),
        "cond_A1": w_a1.ravel(),
        "cond_A2": w_a2.ravel(),
        "cond_mix": w_mix.ravel(),
    }

def _compute_stream_time_variance(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    I, J, T = Y.shape
    var = np.full((J, T), np.nan, dtype=float)
    for j in range(J):
        for t in range(T):
            obs_i = mask[:, j, t]
            if obs_i.sum() >= 2:
                var[j, t] = np.var(Y[obs_i, j, t], ddof=1)
    return var


def _fit_linear_mean_by_stream(theta: np.ndarray, Y: np.ndarray, mask: np.ndarray):
    I, T, R = theta.shape
    J = Y.shape[1]
    beta0 = np.zeros(J, dtype=float)
    beta = np.zeros((J, R), dtype=float)
    for j in range(J):
        obs = mask[:, j, :].ravel()
        if obs.sum() < R + 1:
            continue
        X = theta.reshape(I * T, R)[obs]
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        y = Y[:, j, :].ravel()[obs]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta0[j] = coef[0]
        beta[j] = coef[1:]
    return beta0, beta


def _predict_linear_mean(theta: np.ndarray, beta0: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.einsum("itr,jr->ijt", theta, beta) + beta0[None, :, None]


def coverage_and_length(intervals: np.ndarray, y_true: np.ndarray):
    if intervals.size == 0 or y_true.size == 0:
        return float("nan"), float("nan"), float("nan")
    L = intervals[:, 0]
    U = intervals[:, 1]
    cov = np.mean((y_true >= L) & (y_true <= U))
    lengths = U - L
    return float(cov), float(np.nanquantile(lengths, 0.5)), float(np.nanmean(lengths))

def dataclass_replace_seed(cfg: SimConfig, new_seed: int) -> SimConfig:
    return SimConfig(
        I=cfg.I,
        J=cfg.J,
        T=cfg.T,
        R=cfg.R,
        M=cfg.M,
        Rk=cfg.Rk,
        sigma_u=cfg.sigma_u,
        sigma_theta=cfg.sigma_theta,
        sigma_Y=cfg.sigma_Y,
        kappa=None if cfg.kappa is None else np.array(cfg.kappa, copy=True),
        hetero_mode=cfg.hetero_mode,
        Ps=cfg.Ps,
        spike_window=cfg.spike_window,
        c_spike=cfg.c_spike,
        Psea=cfg.Psea,
        c_seasonal=cfg.c_seasonal,
        Pj=tuple(cfg.Pj),
        rhoj=cfg.rhoj,
        mar_mode=cfg.mar_mode,
        gamma0=None if cfg.gamma0 is None else np.array(cfg.gamma0, copy=True),
        gamma=None if cfg.gamma is None else np.array(cfg.gamma, copy=True),
        gamma_s=cfg.gamma_s,
        delta0=cfg.delta0,
        delta1=cfg.delta1,
        delta2=cfg.delta2,
        seed=new_seed,
        noise_beta_loc=cfg.noise_beta_loc,
        noise_beta_scale=cfg.noise_beta_scale,
        p_h=cfg.p_h,
        v_h=cfg.v_h,
        T_c=cfg.T_c,
        bandwidth=cfg.bandwidth,
    )

def apply_periodic_missingness_and_noise(
    Y_full: np.ndarray,
    O: np.ndarray,
    p_h: float,
    T_c: int,
    low_sigma: float = 0.5,
    high_sigma: float = 5.0,
    seed: int | None = None,
):
    """
    Alternate low/high missingness and noise with period T_c along time.

    For time index t:
      if floor(t / T_c) is even:  P(A=1) = p_h,   noise std = low_sigma
      if floor(t / T_c) is odd:   P(A=1) = 1-p_h, noise std = high_sigma
    """
    I, J, T = Y_full.shape
    rng = np.random.default_rng(seed)

    # Periodic regime: 0 = low, 1 = high
    t = np.arange(T)
    print('T_c:', T_c)
    regime = (t // T_c) % 2  # shape (T,)

    # Time-varying missingness prob p_t[t]
    p_t = np.where(regime == 0, p_h, 1.0 - p_h).astype(float)  # (T,)
    p = p_t[None, None, :]   # broadcastable to (I,J,T)

    # MAR mask A_ijt ~ Bernoulli(p_ijt), using broadcasting
    U = rng.random((I, J, T))
    A = U < p                # boolean (I,J,T)

    # True missingness probabilities per (i,j,t)
    p_true = np.broadcast_to(p, (I, J, T)).copy()

    # Time-varying noise std
    sigma_t = np.where(regime == 0, low_sigma, high_sigma).astype(float)  # (T,)
    eps = rng.normal(0.0, 1.0, size=Y_full.shape) * sigma_t[None, None, :]
    Y_full_noisy = Y_full + eps

    # Observed mask: multi-resolution mask O AND MAR mask A
    observed = O.astype(bool) & A

    return Y_full_noisy, A, p_true, observed

def run_replication(
    cfg: SimConfig,
    cp_method: Any = None,
    rep_id: int = 0,
    alpha: float = 0.1,
    n_test: int = 10000,
    cp_methods: Union[List[Any], Dict[str, Any], None] = None,
    use_test_sampling_weight: bool = False,
    apply_test_weight_to_localized: bool = False,
    K_windows: int = 5,
    return_plot_data: bool = False,
    plot_subject_idx: Optional[np.ndarray] = None,
    plot_blocks: Optional[List[str]] = None,
    use_learned_localizer: bool = False,
    learned_localizer_frac: float = 0.5,
    learned_localizer_model: str = "rf",
) -> Dict[str, Any]:
    cfg = dataclass_replace_seed(cfg, cfg.seed + rep_id)
    theta = simulate_theta(cfg)
    Aload = simulate_loadings(cfg)
    Y_full, mu, sigmaYj, eps, noise_sd = simulate_Y_full(cfg, theta, Aload)
    sched_mask, O = build_multi_resolution_masks(cfg)
    A = sample_mar(cfg, theta, O)
    A = np.ones_like(O) 
    Y_full, A, p_true, observed = apply_periodic_missingness_and_noise(
        Y_full, O, p_h=cfg.p_h, T_c=cfg.T_c, low_sigma=0.5*cfg.v_h, high_sigma=5.0*cfg.v_h, seed=cfg.seed
    )
    is_train, is_cal = train_cal_split(cfg, O, A)
    is_y1 = np.zeros_like(Y_full,dtype=bool)
    is_y1[:,cfg.J-1,:] = True
    D_obs = (O * A) == 1
    D_miss = ~D_obs
    p_hat = estimate_missingness_kernel(D_obs, h=5.0, mode="time_local")
    # Build flat observation arrays
    obs_idx, obs_y = _build_observed_arrays(Y_full, D_obs)
    train_idx, train_y = _build_observed_arrays(Y_full, is_train)
    cal_idx, cal_y = _build_observed_arrays(Y_full, is_cal) 
    y1_idx, y1_true = _build_observed_arrays(Y_full, is_y1)
    cal_y1_idx, cal_y1_true = _build_observed_arrays(Y_full, is_y1&is_cal)
    # Fit mean using DLTS (Torch if available; fallback to linear mean otherwise)
    n_var = cfg.J - 1
    mean_diag: Optional[np.ndarray] = None
    try:
        from .idlfm import dlts_torch  # type: ignore
        try:
            import torch  # type: ignore
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = None
        # Build training arrays with NaN for unobserved
        X = np.full((cfg.I, n_var, cfg.T), np.nan, dtype=float)
        Y1 = np.full((cfg.I, 1, cfg.T), np.nan, dtype=float)
        # Fill only training observations USING train_y only
        for idx, y in zip(train_idx, train_y):
            i = idx // (cfg.J * cfg.T)
            j = (idx % (cfg.J * cfg.T)) // cfg.T
            t = idx % cfg.T
            if j < n_var:
                X[i, j, t] = y
            else:  # j == n_var
                Y1[i, 0, t] = y
        # Hyperparameters (modest defaults)
        rank = max(1, int(cfg.R))
        k = 3
        N = min(max(30, cfg.T // 2), max(40, cfg.T))  # ensure N-k-1 > 0 and not too large
        lambda1 = 1e-7
        lambda2 = 1e-7
        Niter = 501
        alpha_dlts = 8*1e-3
        ebs = 1e-7
        l = 1e-6
        _W, _F, X_hat, Y_hat = dlts_torch(
            X, Y1, cfg.I, n_var, cfg.T, rank, k, N, lambda1, lambda2, Niter, alpha_dlts, ebs, l, device=dev
        )
        mean_diag = np.concatenate([X_hat, Y_hat], axis=1)
    except Exception:
        # Linear mean fit using train_y only
        beta0_diag, beta_diag = _fit_linear_mean_by_stream(theta, Y_full, is_train)
        mean_diag = _predict_linear_mean(theta, beta0_diag, beta_diag)
    # RMSE diagnostics for the response stream (last stream), using mean_diag (works for both DLTS and linear fallback)
    true_y = Y_full[:, n_var, :].reshape(-1)
    pred_y = mean_diag[:, n_var, :].reshape(-1)
    obs_mask = observed[:, n_var, :].reshape(-1)
    unobs_mask = (~observed[:, n_var, :]).reshape(-1)

    observed_rmse = np.sqrt(np.mean((true_y[obs_mask] - pred_y[obs_mask]) ** 2))
    unobserved_rmse = np.sqrt(np.mean((true_y[unobs_mask] - pred_y[unobs_mask]) ** 2))

    print(f"Observed RMSE: {observed_rmse:.4f}")
    print(f"Unobserved RMSE: {unobserved_rmse:.4f}")
    # Compute residuals and variance for downstream (conditional test sampling)
    cal_residuals = np.full_like(Y_full, np.nan)
    cal_residuals[is_cal] = (Y_full - mean_diag)[is_cal]
    idx_marg = test_indices_marginal(O, A, n=n_test)
    idx_cond_a1 = test_indices_conditional(O, A, subset="A1", n=n_test, p=p_true)
    idx_cond_a2 = test_indices_conditional(O, A, subset="A2", n=n_test, p=p_true)
    idx_cond_mix = test_indices_conditional(O, A, subset="mixture_A1_A2", n=n_test, p=p_true)
    # Build probability lookup callable for CP methods that support weighting (missingness odds)
    p_flat = p_true.ravel()
    p_fn = (lambda flat_idx, arr=p_flat: arr[np.asarray(flat_idx, dtype=int)])

    # Default localization bandwidth for methods that support it.
    bandwidth = getattr(cfg, "bandwidth", 10.0)
    learned_repr_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    if use_learned_localizer:
        learned_repr_fn = _build_learned_localizer_repr_fn(
            theta=theta,
            Y_full=Y_full,
            train_idx=train_idx,
            seed=int(cfg.seed) + 7919,
            frac_fit=float(learned_localizer_frac),
            model_kind=str(learned_localizer_model),
        )

    # Optional: build per-block test-sampling weights p^{test} (up to a constant)
    test_w_by_block: Optional[Dict[str, np.ndarray]] = None
    if use_test_sampling_weight:
        test_w_by_block = _build_block_test_weights(
            cfg=cfg,
            cal_residuals=cal_residuals,
            Y_full=Y_full,
            mean_diag=mean_diag,
            local_Delta=10,
            local_centers=[50, 110, 180],
            mix_prob_A2=0.7,
        )

    def _instantiate_method(method_key: str, maker: Any, block_name: Optional[str] = None):
        shape = (int(cfg.I), int(cfg.J), int(cfg.T))
        mk = str(method_key).lower()
        # If the method name suggests localization, attempt to pass bandwidth
        if "localized" in mk:
            try:
                obj = maker(shape, alpha, bandwidth=bandwidth)
            except TypeError:
                try:
                    obj = maker(shape, alpha, bandwidth)
                except TypeError:
                    # Backward-compat: old signature maker(theta, J, alpha, bandwidth=...)
                    try:
                        obj = maker(theta, cfg.J, alpha, bandwidth=bandwidth)
                    except TypeError:
                        obj = maker(theta, cfg.J, alpha)
        else:
            try:
                obj = maker(shape, alpha)
            except TypeError:
                # Backward-compat: old signature maker(theta, J, alpha)
                obj = maker(theta, cfg.J, alpha)

        # Inject missingness probability function (for weighted methods)
        try:
            obj._p_fn = p_fn
        except Exception:
            pass
        if learned_repr_fn is not None and ("localized" in mk):
            try:
                obj._repr_fn = learned_repr_fn
            except Exception:
                pass

        # Inject test sampling weights if requested
        if use_test_sampling_weight and test_w_by_block is not None and block_name is not None:
            should_apply = ("weighted" in mk) or (apply_test_weight_to_localized and ("localized" in mk))
            if should_apply:
                try:
                    obj._test_w_fn = _test_weight_fn_from_flat(test_w_by_block[block_name])
                except Exception:
                    pass
        return obj

    # Calibrate each method and evaluate blocks
    cp_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    cal_y1_fit = mean_diag.ravel()[cal_y1_idx] if cal_y1_idx.size > 0 else np.array([])

    def eval_block_for_cp(cp_obj: Any, idx: np.ndarray):
        if idx.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        if not hasattr(cp_obj, "predict_interval_from_fit"):
            raise AttributeError("CP method must implement predict_interval_from_fit(X_test_idx, Y_test_fit, alpha).")
        y_fit_test = mean_diag.ravel()[idx]
        intervals = cp_obj.predict_interval_from_fit(idx, y_fit_test, alpha=alpha)  # type: ignore[attr-defined]
        y_true = Y_full.ravel()[idx]
        cov, medlen, meanlen = coverage_and_length(intervals, y_true)
        # Window-local metric (defaults to response stream j=J-1 because idx is intersected with y1_idx)
        wgap = window_local_abs_coverage_gap(
            intervals=intervals,
            y_true=y_true,
            idx_flat=idx,
            I=cfg.I,
            J=cfg.J,
            T=cfg.T,
            K=K_windows,
            target=float(1.0 - alpha),
            j_filter=cfg.J - 1,
        )
        return cov, medlen, meanlen, wgap

    # Build method factories list
    method_factories: List[tuple[str, Any]] = []
    if cp_methods is not None:
        if isinstance(cp_methods, dict):
            method_factories = [(str(name), maker) for name, maker in cp_methods.items()]
        else:
            # best-effort naming for list of makers
            for maker in cp_methods:
                method_factories.append((getattr(maker, "__name__", str(maker)), maker))
    elif cp_method is not None:
        method_factories = [(getattr(cp_method, "__name__", "cp_method"), cp_method)]
    else:
        raise ValueError("Provide cp_method or cp_methods.")

    # Evaluate each block. If using test-sampling weights, re-instantiate/re-calibrate per block
    # so each block uses its own p^{test}.
    block_defs = [
        ("marginal", np.intersect1d(idx_marg, y1_idx)),
        ("cond_A1", np.intersect1d(idx_cond_a1, y1_idx)),
        ("cond_A2", np.intersect1d(idx_cond_a2, y1_idx)),
        ("cond_mix", np.intersect1d(idx_cond_mix, y1_idx)),
    ]

    for method_name, maker in method_factories:
        blocks: Dict[str, Dict[str, float]] = {}
        if not use_test_sampling_weight:
            # Fast path: instantiate once and reuse across blocks
            cp_obj = _instantiate_method(method_name, maker, block_name=None)
            if cal_y1_idx.size > 0:
                if not hasattr(cp_obj, "calibrate_from_fit"):
                    raise AttributeError("CP method must implement calibrate_from_fit(X_cal_idx, Y_cal_true, Y_cal_fit).")
                cp_obj.calibrate_from_fit(cal_y1_idx, cal_y1_true, cal_y1_fit)  # type: ignore[attr-defined]
            for block_name, idx in block_defs:
                cov, medlen, meanlen, wgap = eval_block_for_cp(cp_obj, idx)
                blocks[block_name] = {
                    "coverage": cov,
                    "median_len": medlen,
                    "mean_len": meanlen,
                    "window_abs_gap": wgap,
                }
        else:
            # Per-block instantiation so each block has its own p^{test}
            for block_name, idx in block_defs:
                cp_obj = _instantiate_method(method_name, maker, block_name=block_name)
                if cal_y1_idx.size > 0:
                    if not hasattr(cp_obj, "calibrate_from_fit"):
                        raise AttributeError("CP method must implement calibrate_from_fit(X_cal_idx, Y_cal_true, Y_cal_fit).")
                    cp_obj.calibrate_from_fit(cal_y1_idx, cal_y1_true, cal_y1_fit)  # type: ignore[attr-defined]
                cov, medlen, meanlen, wgap = eval_block_for_cp(cp_obj, idx)
                blocks[block_name] = {
                    "coverage": cov,
                    "median_len": medlen,
                    "mean_len": meanlen,
                    "window_abs_gap": wgap,
                }

        cp_results[method_name] = blocks

    plot_data: Optional[Dict[str, Any]] = None
    if return_plot_data:
        # Plot-ready payload for the outcome stream (j = J-1), across all requested subjects and all timepoints.
        j_out = int(cfg.J - 1)
        I0, J0, T0 = Y_full.shape
        subj = np.arange(I0, dtype=int) if plot_subject_idx is None else np.asarray(plot_subject_idx, dtype=int).ravel()
        subj = subj[(subj >= 0) & (subj < I0)]
        t_all = np.arange(T0, dtype=int)

        # Flat indices for all (i, j_out, t) for selected subjects, in row-major flattening.
        idx_grid = ((subj[:, None] * J0 + j_out) * T0 + t_all[None, :]).astype(np.int64, copy=False)  # (n_subj,T)
        idx_flat_all = idx_grid.ravel()

        y_true_mat = Y_full[subj, j_out, :].astype(float, copy=False)         # (n_subj,T)
        y_fit_mat = mean_diag[subj, j_out, :].astype(float, copy=False)       # (n_subj,T)
        is_missing_mat = D_miss[subj, j_out, :].astype(bool, copy=False)      # (n_subj,T)
        is_observed_mat = D_obs[subj, j_out, :].astype(bool, copy=False)      # (n_subj,T)
        is_cal_mat = is_cal[subj, j_out, :].astype(bool, copy=False)          # (n_subj,T)
        is_train_mat = is_train[subj, j_out, :].astype(bool, copy=False)      # (n_subj,T)

        # Decide which "blocks" to include in plot payload.
        if plot_blocks is None:
            blocks_for_plot = ["marginal"] if not use_test_sampling_weight else ["marginal", "local", "cond_A1", "cond_A2", "cond_mix"]
        else:
            blocks_for_plot = [str(b) for b in plot_blocks]

        methods_payload: Dict[str, Dict[str, Any]] = {}
        for method_name, maker in method_factories:
            methods_payload[method_name] = {}
            for block_name in blocks_for_plot:
                # Match the evaluation-time configuration: if test-sampling weights are enabled, use the per-block weights.
                cp_obj = _instantiate_method(method_name, maker, block_name=(block_name if use_test_sampling_weight else None))
                if cal_y1_idx.size > 0:
                    if not hasattr(cp_obj, "calibrate_from_fit"):
                        raise AttributeError("CP method must implement calibrate_from_fit(X_cal_idx, Y_cal_true, Y_cal_fit).")
                    cp_obj.calibrate_from_fit(cal_y1_idx, cal_y1_true, cal_y1_fit)  # type: ignore[attr-defined]

                y_fit_flat = mean_diag.ravel()[idx_flat_all]
                intervals = cp_obj.predict_interval_from_fit(idx_flat_all, y_fit_flat, alpha=alpha)  # type: ignore[attr-defined]
                L = intervals[:, 0].reshape(subj.size, T0)
                U = intervals[:, 1].reshape(subj.size, T0)
                length = (U - L)
                covered = ((y_true_mat >= L) & (y_true_mat <= U))

                methods_payload[method_name][block_name] = {
                    "L": L,
                    "U": U,
                    "length": length,
                    "covered": covered,
                }

        plot_data = {
            "stream_j": j_out,
            "subject_idx": subj,
            "t": t_all,
            "y_true": y_true_mat,
            "y_fit": y_fit_mat,
            "is_missing": is_missing_mat,
            "is_observed": is_observed_mat,
            "is_cal": is_cal_mat,
            "is_train": is_train_mat,
            "methods": methods_payload,
        }

    # Organized JSON-style result
    return {
        "meta": {
            "rep_id": rep_id,
            "alpha": alpha,
            "n_test": n_test,
            "seed": cfg.seed,
            "use_test_sampling_weight": use_test_sampling_weight,
            "K_windows": K_windows,
            "plot_data_included": bool(return_plot_data),
            "use_learned_localizer": bool(use_learned_localizer),
            "learned_localizer_model": str(learned_localizer_model),
        },
        "config": {
            "I": cfg.I,
            "J": cfg.J,
            "T": cfg.T,
            "R": cfg.R,
        },
        "splits": {
            "is_train": is_train,
            "is_cal": is_cal,
        },
        "indices": {
            "marginal": idx_marg,
            "cond_A1": idx_cond_a1,
            "cond_A2": idx_cond_a2,
            "cond_mix": idx_cond_mix,
        },
        "test_sampling_weights": {} if test_w_by_block is None else {k: v for k, v in test_w_by_block.items()},
        "cp_results": cp_results,
        "plot_data": plot_data,
    }


