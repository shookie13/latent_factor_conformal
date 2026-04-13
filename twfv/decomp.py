import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, Any, List

import torch

@dataclass
class FactorChannelScoresResult:
    target_j: int
    factor_mode: str
    split: str
    indices: torch.Tensor         # (n, 2) storing [i, t]
    pred_mean: torch.Tensor       # (n,)
    pred_var: torch.Tensor        # (n,)
    residual: torch.Tensor        # (n,)
    raw_score: torch.Tensor       # (n,) = |residual|
    std_score: torch.Tensor       # (n,) = |residual| / sqrt(pred_var)
    pit_score: torch.Tensor       # (n,) = F_{chi^2_1}(std_score^2)
    factor_mean: torch.Tensor     # (n, r)
    obs_counts: torch.Tensor      # (n,)


def _normalize_factor_mode(factor_mode: str) -> str:
    mode = str(factor_mode).strip().lower()
    aliases = {
        'loo': 'leave_target_out',
        'leave_one_out': 'leave_target_out',
        'leave_target_out': 'leave_target_out',
        'exclude_target': 'leave_target_out',
        'new': 'leave_target_out',
        'observed': 'observed',
        'as_observed': 'observed',
        'current': 'observed',
        'old': 'observed',
        'full': 'full',
        'full_j': 'full',
        'all_channels': 'full',
        'oracle_full': 'full',
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown factor_mode={factor_mode!r}. Use one of "
            f"{sorted(set(aliases.values()))} or a listed alias."
        )
    return aliases[mode]


def _normalize_score_kind(score_kind: str) -> str:
    kind = str(score_kind).strip().lower()
    aliases = {
        'raw': 'raw',
        'residual': 'raw',
        'abs_residual': 'raw',
        'standardized': 'standardized',
        'std': 'standardized',
        'scaled': 'standardized',
        'pit': 'pit',
        'cdf': 'pit',
    }
    if kind not in aliases:
        raise ValueError(
            f"Unknown score_kind={score_kind!r}. Use one of {sorted(set(aliases.values()))}."
        )
    return aliases[kind]


def _factor_channels_for_target(
    M_mask: torch.Tensor,
    i: int,
    t: int,
    j: int,
    *,
    factor_mode: str,
) -> torch.Tensor:
    mode = _normalize_factor_mode(factor_mode)
    device = M_mask.device
    J = M_mask.shape[2]

    if mode == 'full':
        return torch.arange(J, device=device, dtype=torch.long)

    obs_idx = torch.nonzero(M_mask[i, t], as_tuple=False).squeeze(1)
    if mode == 'leave_target_out':
        obs_idx = obs_idx[obs_idx != j]
    return obs_idx


def factor_posterior_given_indices(
    y_row: torch.Tensor,
    obs_idx: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s_i: torch.Tensor,
    a2_row: torch.Tensor,
    *,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Posterior mean/covariance of F for a single (i,t) row under
        e = L F + u,
        F ~ N(0, diag(a2_row)),
        u ~ N(0, s_i^2 diag(psi)).

    Parameters
    ----------
    y_row : (J,)
        Residual row e_{it, :}.
    obs_idx : (J_obs,)
        Channel indices used to estimate the factor posterior.
    L : (J, r)
    psi : (J,)
    s_i : scalar tensor
    a2_row : (r,)

    Returns
    -------
    f_mean : (r,)
    f_cov  : (r, r)
    """
    dtype = y_row.dtype
    device = y_row.device
    r = L.shape[1]

    prior_prec = torch.diag(1.0 / (a2_row + eps))
    eye_r = torch.eye(r, dtype=dtype, device=device)

    if obs_idx.numel() == 0:
        # No observed channels: posterior equals prior.
        f_cov = torch.diag(a2_row.clamp_min(eps))
        f_mean = torch.zeros(r, dtype=dtype, device=device)
        return f_mean, f_cov

    y_obs = y_row.index_select(0, obs_idx)
    L_obs = L.index_select(0, obs_idx)
    psi_obs = psi.index_select(0, obs_idx)
    noise_prec_diag = 1.0 / ((s_i * s_i) * psi_obs + eps)

    precision = prior_prec + L_obs.T @ (noise_prec_diag.unsqueeze(1) * L_obs)
    if jitter > 0.0:
        precision = precision + jitter * eye_r

    chol = torch.linalg.cholesky(precision)
    rhs = L_obs.T @ (noise_prec_diag * y_obs)
    f_mean = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    f_cov = torch.cholesky_inverse(chol)
    return f_mean, f_cov


def factor_channel_predictive_params(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    i: int,
    t: int,
    target_j: int,
    factor_mode: str = 'leave_target_out',
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Compute the target-channel predictive mean/variance for one (i,t,j).

    factor_mode:
      - 'leave_target_out': estimate F from observed channels excluding j (new procedure)
      - 'observed':         estimate F from all currently observed channels (old behavior)
      - 'full':             estimate F from all J channels, ignoring the mask (oracle comparison)
    """
    obs_idx = _factor_channels_for_target(M_mask, i, t, target_j, factor_mode=factor_mode)
    f_mean, f_cov = factor_posterior_given_indices(
        Y[i, t],
        obs_idx,
        L,
        psi,
        s[i],
        a2[i, t],
        eps=eps,
        jitter=jitter,
    )

    Lj = L[target_j]
    pred_mean = torch.dot(Lj, f_mean)
    pred_var = (s[i] * s[i]) * psi[target_j] + Lj @ f_cov @ Lj
    pred_var = pred_var.clamp_min(eps)

    return {
        'pred_mean': pred_mean,
        'pred_var': pred_var,
        'factor_mean': f_mean,
        'factor_cov': f_cov,
        'obs_idx': obs_idx,
    }


def _select_it_mask_for_channel(
    M_mask: torch.Tensor,
    target_j: int,
    *,
    split: str,
    point_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    split_l = str(split).strip().lower()
    if point_mask is not None:
        point_mask = point_mask.to(dtype=torch.bool, device=M_mask.device)
        if point_mask.shape != M_mask.shape[:2]:
            raise ValueError(
                f"point_mask must have shape (I,T)={tuple(M_mask.shape[:2])}, got {tuple(point_mask.shape)}"
            )
        return point_mask

    if split_l == 'cal':
        return M_mask[:, :, target_j]
    if split_l == 'test':
        return ~M_mask[:, :, target_j]
    if split_l == 'all':
        return torch.ones(M_mask.shape[:2], dtype=torch.bool, device=M_mask.device)
    raise ValueError("split must be one of {'cal', 'test', 'all'}")


def _pit_from_std_score(std_score: torch.Tensor) -> torch.Tensor:
    # If Z ~ N(0,1), then Z^2 ~ chi^2_1 and F_{chi^2_1}(z^2) = erf(|z| / sqrt(2)).
    return torch.erf(std_score / math.sqrt(2.0))


def _std_cutoff_from_pit_cutoff(pit_cutoff: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    u = pit_cutoff.clamp(min=eps, max=1.0 - eps)
    return math.sqrt(2.0) * torch.special.erfinv(u)


def compute_factor_channel_scores(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    target_j: int,
    split: str = 'cal',
    factor_mode: str = 'leave_target_out',
    point_mask: Optional[torch.Tensor] = None,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> FactorChannelScoresResult:
    """
    Compute raw / standardized / PIT scores for one target channel.

    Notes
    -----
    - split='cal' uses points with M[..., j] == True by default.
    - split='test' uses points with M[..., j] == False by default.
    - factor_mode='full' ignores the mask and conditions on the entire row Y[i,t,:].
      This is an oracle/diagnostic mode. By default it is blocked on hidden points,
      because it peeks at the missing target. Set allow_oracle_missing=True to force it.
    """
    factor_mode = _normalize_factor_mode(factor_mode)
    it_mask = _select_it_mask_for_channel(M_mask, target_j, split=split, point_mask=point_mask)
    it_idx = torch.nonzero(it_mask, as_tuple=False)

    if factor_mode == 'full' and split == 'test' and not allow_oracle_missing:
        raise ValueError(
            "factor_mode='full' on split='test' uses the hidden target and is therefore oracle-only. "
            "Pass allow_oracle_missing=True if you want this for diagnostics/comparison."
        )

    n = int(it_idx.shape[0])
    r = int(L.shape[1])
    device = Y.device
    dtype = Y.dtype

    pred_mean = torch.empty(n, dtype=dtype, device=device)
    pred_var = torch.empty(n, dtype=dtype, device=device)
    residual = torch.empty(n, dtype=dtype, device=device)
    raw_score = torch.empty(n, dtype=dtype, device=device)
    std_score = torch.empty(n, dtype=dtype, device=device)
    pit_score = torch.empty(n, dtype=dtype, device=device)
    factor_mean = torch.empty((n, r), dtype=dtype, device=device)
    obs_counts = torch.empty(n, dtype=torch.long, device=device)

    for k in range(n):
        i = int(it_idx[k, 0].item())
        t = int(it_idx[k, 1].item())
        out = factor_channel_predictive_params(
            Y,
            M_mask,
            L,
            psi,
            s,
            a2,
            i=i,
            t=t,
            target_j=target_j,
            factor_mode=factor_mode,
            eps=eps,
            jitter=jitter,
        )
        mu = out['pred_mean']
        var = out['pred_var']
        res = Y[i, t, target_j] - mu
        sd = torch.sqrt(var)

        pred_mean[k] = mu
        pred_var[k] = var
        residual[k] = res
        raw_score[k] = torch.abs(res)
        std_score[k] = torch.abs(res) / sd
        pit_score[k] = _pit_from_std_score(std_score[k])
        factor_mean[k] = out['factor_mean']
        obs_counts[k] = int(out['obs_idx'].numel())

    return FactorChannelScoresResult(
        target_j=target_j,
        factor_mode=factor_mode,
        split=str(split).lower(),
        indices=it_idx,
        pred_mean=pred_mean,
        pred_var=pred_var,
        residual=residual,
        raw_score=raw_score,
        std_score=std_score,
        pit_score=pit_score,
        factor_mean=factor_mean,
        obs_counts=obs_counts,
    )


def _weighted_conformal_quantile(
    scores: torch.Tensor,
    *,
    alpha: float,
    cal_weights: Optional[torch.Tensor] = None,
    test_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted split-conformal quantile with a single +infty ghost mass for the test point.

    When cal_weights=None, this reduces to the usual split-conformal quantile:
        k = ceil((n + 1) * (1 - alpha)).
    """
    vals = scores.reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        raise ValueError('No finite scores available for calibration.')

    if cal_weights is None:
        w = torch.ones_like(vals)
    else:
        w = cal_weights.reshape(-1).to(device=vals.device, dtype=vals.dtype)
        w = w[torch.isfinite(vals)]
        if w.numel() != vals.numel():
            raise ValueError('cal_weights must match the number of scores.')
        if torch.any(w < 0):
            raise ValueError('cal_weights must be nonnegative.')

    order = torch.argsort(vals)
    vals = vals[order]
    w = w[order]

    denom = w.sum() + torch.as_tensor(test_weight, device=vals.device, dtype=vals.dtype)
    if denom <= 0:
        raise ValueError('Sum of calibration weights plus test_weight must be positive.')

    target = 1.0 - float(alpha)
    cdf = torch.cumsum(w, dim=0) / denom
    hit = torch.nonzero(cdf >= target, as_tuple=False)
    if hit.numel() == 0:
        return torch.as_tensor(float('inf'), device=vals.device, dtype=vals.dtype)
    return vals[int(hit[0, 0].item())]


def calibrate_factor_channel_quantiles(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    alpha: float = 0.1,
    channels: Optional[Sequence[int]] = None,
    factor_mode: str = 'leave_target_out',
    score_kind: str = 'standardized',
    cal_point_mask: Optional[torch.Tensor] = None,
    cal_weights: Optional[torch.Tensor] = None,
    test_weight: float = 1.0,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[int, Dict[str, Any]]:
    """
    Calibrate per-channel conformal thresholds.

    Returns
    -------
    out[j] is a dict with keys:
        'threshold', 'scores', 'score_kind', 'factor_mode'
    """
    score_kind = _normalize_score_kind(score_kind)
    J = int(Y.shape[2])
    if channels is None:
        channels = list(range(J))

    out: Dict[int, Dict[str, Any]] = {}
    for j in channels:
        res = compute_factor_channel_scores(
            Y,
            M_mask,
            L,
            psi,
            s,
            a2,
            target_j=int(j),
            split='cal',
            factor_mode=factor_mode,
            point_mask=cal_point_mask,
            allow_oracle_missing=allow_oracle_missing,
            eps=eps,
            jitter=jitter,
        )
        if score_kind == 'raw':
            scores = res.raw_score
        elif score_kind == 'standardized':
            scores = res.std_score
        else:
            scores = res.pit_score

        w = None
        if cal_weights is not None:
            if cal_weights.shape == M_mask.shape:
                ij = res.indices
                w = cal_weights[ij[:, 0], ij[:, 1], int(j)]
            elif cal_weights.shape == M_mask.shape[:2]:
                ij = res.indices
                w = cal_weights[ij[:, 0], ij[:, 1]]
            else:
                raise ValueError(
                    'cal_weights must have shape (I,T,J) or (I,T) when provided.'
                )

        thr = _weighted_conformal_quantile(
            scores,
            alpha=alpha,
            cal_weights=w,
            test_weight=test_weight,
        )
        out[int(j)] = {
            'threshold': thr,
            'scores': res,
            'score_kind': score_kind,
            'factor_mode': _normalize_factor_mode(factor_mode),
            'alpha': float(alpha),
        }
    return out


def predict_factor_channel_intervals(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    quantiles_by_channel: Dict[int, Dict[str, Any]],
    *,
    pred_point_mask: Optional[torch.Tensor] = None,
    factor_mode: Optional[str] = None,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Build per-channel prediction intervals for points with M[...,j]==False by default.

    Parameters
    ----------
    quantiles_by_channel:
        Output of calibrate_factor_channel_quantiles(...).
    factor_mode:
        If None, each channel uses the factor_mode stored in quantiles_by_channel[j].
        Otherwise this overrides it.
    """
    I, T, J = Y.shape
    device = Y.device
    dtype = Y.dtype

    lower = torch.full((I, T, J), float('nan'), dtype=dtype, device=device)
    upper = torch.full((I, T, J), float('nan'), dtype=dtype, device=device)
    center = torch.full((I, T, J), float('nan'), dtype=dtype, device=device)
    pred_var = torch.full((I, T, J), float('nan'), dtype=dtype, device=device)

    for j, info in quantiles_by_channel.items():
        mode_j = _normalize_factor_mode(info['factor_mode'] if factor_mode is None else factor_mode)
        score_kind = _normalize_score_kind(info['score_kind'])
        q = info['threshold'].to(device=device, dtype=dtype)
        it_mask = _select_it_mask_for_channel(M_mask, int(j), split='test', point_mask=pred_point_mask)
        it_idx = torch.nonzero(it_mask, as_tuple=False)

        if mode_j == 'full' and not allow_oracle_missing:
            raise ValueError(
                "factor_mode='full' on hidden/test points is oracle-only. "
                "Pass allow_oracle_missing=True if you want diagnostic/oracle intervals."
            )

        if score_kind == 'pit':
            q_eff = _std_cutoff_from_pit_cutoff(q, eps=eps)
        else:
            q_eff = q

        for k in range(int(it_idx.shape[0])):
            i = int(it_idx[k, 0].item())
            t = int(it_idx[k, 1].item())
            out = factor_channel_predictive_params(
                Y,
                M_mask,
                L,
                psi,
                s,
                a2,
                i=i,
                t=t,
                target_j=int(j),
                factor_mode=mode_j,
                eps=eps,
                jitter=jitter,
            )
            mu = out['pred_mean']
            var = out['pred_var']
            if score_kind == 'raw':
                width = q_eff
            else:
                width = q_eff * torch.sqrt(var)

            center[i, t, int(j)] = mu
            pred_var[i, t, int(j)] = var
            lower[i, t, int(j)] = mu - width
            upper[i, t, int(j)] = mu + width

    return {
        'lower': lower,
        'upper': upper,
        'center': center,
        'pred_var': pred_var,
    }


def run_factor_channel_conformal(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    alpha: float = 0.1,
    channels: Optional[Sequence[int]] = None,
    factor_mode: str = 'leave_target_out',
    score_kind: str = 'standardized',
    cal_point_mask: Optional[torch.Tensor] = None,
    pred_point_mask: Optional[torch.Tensor] = None,
    cal_weights: Optional[torch.Tensor] = None,
    test_weight: float = 1.0,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[str, Any]:
    """
    One-stop wrapper:
      1) calibrate channelwise thresholds on observed/calibration entries,
      2) build intervals on hidden/test entries.
    """
    q = calibrate_factor_channel_quantiles(
        Y,
        M_mask,
        L,
        psi,
        s,
        a2,
        alpha=alpha,
        channels=channels,
        factor_mode=factor_mode,
        score_kind=score_kind,
        cal_point_mask=cal_point_mask,
        cal_weights=cal_weights,
        test_weight=test_weight,
        allow_oracle_missing=allow_oracle_missing,
        eps=eps,
        jitter=jitter,
    )
    intervals = predict_factor_channel_intervals(
        Y,
        M_mask,
        L,
        psi,
        s,
        a2,
        q,
        pred_point_mask=pred_point_mask,
        factor_mode=factor_mode,
        allow_oracle_missing=allow_oracle_missing,
        eps=eps,
        jitter=jitter,
    )
    return {
        'quantiles_by_channel': q,
        'intervals': intervals,
    }

# =============================
# Joint hidden-block conformal
# =============================

@dataclass
class FactorJointScoresResult:
    factor_mode: str
    split: str
    indices: torch.Tensor                 # (n, 2) storing [i, t]
    hidden_idx_list: List[torch.Tensor]   # length n; hidden indices for each row
    obs_idx_list: List[torch.Tensor]      # length n; conditioning indices used for factor posterior
    hidden_sizes: torch.Tensor            # (n,)
    pred_mean_list: List[torch.Tensor]    # length n; each (d,)
    pred_cov_list: List[torch.Tensor]     # length n; each (d, d)
    residual_list: List[torch.Tensor]     # length n; each (d,)
    mahal_sq: torch.Tensor                # (n,) squared Mahalanobis scores
    mahal_score: torch.Tensor             # (n,) sqrt(mahal_sq)
    pit_score: torch.Tensor               # (n,) = F_{chi^2_d}(mahal_sq)
    factor_mean: torch.Tensor             # (n, r)
    obs_counts: torch.Tensor              # (n,)


def _normalize_joint_factor_mode(factor_mode: str) -> str:
    mode = str(factor_mode).strip().lower()
    aliases = {
        'observed': 'observed',
        'as_observed': 'observed',
        'current': 'observed',
        'mask': 'observed',
        'leave_block_out': 'observed',
        'block': 'observed',
        'full': 'full',
        'full_j': 'full',
        'all_channels': 'full',
        'oracle_full': 'full',
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown joint factor_mode={factor_mode!r}. Use one of {sorted(set(aliases.values()))} or a listed alias."
        )
    return aliases[mode]


def _normalize_joint_score_kind(score_kind: str) -> str:
    kind = str(score_kind).strip().lower()
    aliases = {
        'mahal': 'mahalanobis',
        'mahalanobis': 'mahalanobis',
        'joint_mahalanobis': 'mahalanobis',
        'pit': 'pit',
        'cdf': 'pit',
        'pooled_pit': 'pit',
    }
    if kind not in aliases:
        raise ValueError(
            f"Unknown joint score_kind={score_kind!r}. Use one of {sorted(set(aliases.values()))}."
        )
    return aliases[kind]


def _normalize_joint_group_by(group_by: str) -> str:
    gb = str(group_by).strip().lower()
    aliases = {
        'pattern': 'pattern',
        'mask_pattern': 'pattern',
        'exact_pattern': 'pattern',
        'size': 'size',
        'dim': 'size',
        'dimension': 'size',
        'pooled': 'all',
        'all': 'all',
        'global': 'all',
    }
    if gb not in aliases:
        raise ValueError(
            f"Unknown group_by={group_by!r}. Use one of {sorted(set(aliases.values()))}."
        )
    return aliases[gb]


def _canonical_long_index(idx: Optional[torch.Tensor], *, device, max_size: int) -> torch.Tensor:
    if idx is None:
        return torch.empty(0, dtype=torch.long, device=device)
    idx = torch.as_tensor(idx, dtype=torch.long, device=device).reshape(-1)
    if idx.numel() == 0:
        return idx
    if torch.any(idx < 0) or torch.any(idx >= max_size):
        raise ValueError(f'Index values must lie in [0, {max_size - 1}].')
    idx = torch.unique(idx, sorted=True)
    return idx


def _select_it_mask_for_joint(
    M_mask: torch.Tensor,
    *,
    split: str,
    point_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if point_mask is not None:
        point_mask = point_mask.to(dtype=torch.bool, device=M_mask.device)
        if point_mask.shape != M_mask.shape[:2]:
            raise ValueError(
                f"point_mask must have shape (I,T)={tuple(M_mask.shape[:2])}, got {tuple(point_mask.shape)}"
            )
        return point_mask

    split_l = str(split).strip().lower()
    any_missing = (~M_mask).any(dim=2)
    if split_l in {'cal', 'test', 'missing', 'joint'}:
        return any_missing
    if split_l == 'all':
        return torch.ones(M_mask.shape[:2], dtype=torch.bool, device=M_mask.device)
    if split_l in {'observed_only', 'full_obs'}:
        return M_mask.all(dim=2)
    raise ValueError("split must be one of {'cal', 'test', 'missing', 'joint', 'all', 'observed_only'}")


def _joint_group_key_from_hidden_idx(hidden_idx: torch.Tensor, *, group_by: str):
    gb = _normalize_joint_group_by(group_by)
    if gb == 'pattern':
        return tuple(int(x) for x in hidden_idx.detach().cpu().tolist())
    if gb == 'size':
        return int(hidden_idx.numel())
    return 'all'


def _chi2_cdf_from_mahal_sq(mahal_sq: torch.Tensor, df: int) -> torch.Tensor:
    x = torch.as_tensor(mahal_sq)
    if df <= 0:
        raise ValueError('df must be positive for chi-square CDF.')
    if x.numel() == 0:
        return x.clone()
    a = torch.as_tensor(0.5 * float(df), dtype=x.dtype, device=x.device)
    z = 0.5 * x.clamp_min(0.0)
    return torch.special.gammainc(a, z)


def _chi2_ppf_bisection(
    u: torch.Tensor,
    *,
    df: int,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> torch.Tensor:
    """
    Numerically invert the chi-square CDF using bisection.
    """
    u = torch.as_tensor(u)
    if df <= 0:
        raise ValueError('df must be positive for chi-square quantiles.')
    out = torch.empty_like(u)
    finite_mask = torch.isfinite(u)
    if not torch.all(finite_mask):
        out[~finite_mask] = torch.as_tensor(float('nan'), dtype=u.dtype, device=u.device)
    if finite_mask.any():
        x = u[finite_mask].clamp(min=0.0, max=1.0)
        out_local = torch.empty_like(x)
        zero_mask = x <= 0.0
        one_mask = x >= 1.0
        mid_mask = (~zero_mask) & (~one_mask)
        if zero_mask.any():
            out_local[zero_mask] = 0.0
        if one_mask.any():
            out_local[one_mask] = torch.as_tensor(float('inf'), dtype=x.dtype, device=x.device)
        if mid_mask.any():
            x_mid = x[mid_mask]
            lo = torch.zeros_like(x_mid)
            hi = torch.full_like(x_mid, max(1.0, float(df)))
            cdf_hi = _chi2_cdf_from_mahal_sq(hi, df)
            expand_steps = 0
            while torch.any(cdf_hi < x_mid):
                hi = torch.where(cdf_hi < x_mid, hi * 2.0, hi)
                cdf_hi = _chi2_cdf_from_mahal_sq(hi, df)
                expand_steps += 1
                if expand_steps > 200:
                    break
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                cdf_mid = _chi2_cdf_from_mahal_sq(mid, df)
                go_right = cdf_mid < x_mid
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
                if torch.max(hi - lo) <= tol:
                    break
            out_local[mid_mask] = 0.5 * (lo + hi)
        out[finite_mask] = out_local
    return out


def factor_joint_predictive_params(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    i: int,
    t: int,
    hidden_idx: Optional[torch.Tensor] = None,
    factor_mode: str = 'observed',
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Compute the predictive mean/covariance of the hidden block e_{it,H} given the observed block.

    Parameters
    ----------
    hidden_idx:
        Indices of the block to predict. If None, uses the missing block under M_mask[i, t].
        If a custom hidden_idx contains currently observed channels, those channels are pseudo-hidden
        when factor_mode='observed'.
    factor_mode:
      - 'observed': estimate F from the currently observed channels excluding hidden_idx
      - 'full':     estimate F from all J channels (oracle comparison)
    """
    mode = _normalize_joint_factor_mode(factor_mode)
    J = int(Y.shape[2])
    device = Y.device
    dtype = Y.dtype

    if hidden_idx is None:
        hidden_idx = torch.nonzero(~M_mask[i, t], as_tuple=False).squeeze(1)
    hidden_idx = _canonical_long_index(hidden_idx, device=device, max_size=J)

    if hidden_idx.numel() == 0:
        raise ValueError('hidden_idx is empty for this (i,t); joint prediction requires a non-empty hidden block.')

    if mode == 'full':
        if torch.any(~M_mask[i, t].index_select(0, hidden_idx)) and not allow_oracle_missing:
            raise ValueError(
                "factor_mode='full' uses the hidden block itself. Set allow_oracle_missing=True for oracle diagnostics."
            )
        obs_idx = torch.arange(J, device=device, dtype=torch.long)
    else:
        obs_idx = torch.nonzero(M_mask[i, t], as_tuple=False).squeeze(1)
        if hidden_idx.numel() > 0 and obs_idx.numel() > 0:
            keep = torch.ones(obs_idx.numel(), dtype=torch.bool, device=device)
            for h in hidden_idx:
                keep &= (obs_idx != h)
            obs_idx = obs_idx[keep]

    f_mean, f_cov = factor_posterior_given_indices(
        Y[i, t],
        obs_idx,
        L,
        psi,
        s[i],
        a2[i, t],
        eps=eps,
        jitter=jitter,
    )

    L_H = L.index_select(0, hidden_idx)
    psi_H = psi.index_select(0, hidden_idx)
    pred_mean = L_H @ f_mean
    pred_cov = L_H @ f_cov @ L_H.T + (s[i] * s[i]) * torch.diag(psi_H)
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    if jitter > 0.0:
        eye = torch.eye(hidden_idx.numel(), dtype=dtype, device=device)
        pred_cov = pred_cov + jitter * eye

    return {
        'hidden_idx': hidden_idx,
        'obs_idx': obs_idx,
        'factor_mean': f_mean,
        'factor_cov': f_cov,
        'pred_mean': pred_mean,
        'pred_cov': pred_cov,
        'pred_var_diag': torch.diagonal(pred_cov),
    }


def compute_factor_joint_scores(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    split: str = 'cal',
    factor_mode: str = 'observed',
    point_mask: Optional[torch.Tensor] = None,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> FactorJointScoresResult:
    """
    Compute joint hidden-block Mahalanobis and PIT scores row-by-row.

    By default, each row uses its actual missing block H_{it} = {j : M[i,t,j] = 0}.
    This is the natural joint analogue of predicting all missing/test channels together.
    """
    factor_mode = _normalize_joint_factor_mode(factor_mode)
    it_mask = _select_it_mask_for_joint(M_mask, split=split, point_mask=point_mask)
    candidates = torch.nonzero(it_mask, as_tuple=False)

    rows: List[torch.Tensor] = []
    hidden_idx_list: List[torch.Tensor] = []
    obs_idx_list: List[torch.Tensor] = []
    pred_mean_list: List[torch.Tensor] = []
    pred_cov_list: List[torch.Tensor] = []
    residual_list: List[torch.Tensor] = []
    mahal_sq_list: List[torch.Tensor] = []
    mahal_list: List[torch.Tensor] = []
    pit_list: List[torch.Tensor] = []
    factor_mean_rows: List[torch.Tensor] = []
    obs_counts_list: List[int] = []
    hidden_sizes_list: List[int] = []

    for k in range(int(candidates.shape[0])):
        i = int(candidates[k, 0].item())
        t = int(candidates[k, 1].item())
        hidden_idx = torch.nonzero(~M_mask[i, t], as_tuple=False).squeeze(1)
        if hidden_idx.numel() == 0:
            continue

        out = factor_joint_predictive_params(
            Y,
            M_mask,
            L,
            psi,
            s,
            a2,
            i=i,
            t=t,
            hidden_idx=hidden_idx,
            factor_mode=factor_mode,
            allow_oracle_missing=allow_oracle_missing,
            eps=eps,
            jitter=jitter,
        )

        mean_H = out['pred_mean']
        cov_H = out['pred_cov']
        y_H = Y[i, t].index_select(0, hidden_idx)
        resid_H = y_H - mean_H

        chol = torch.linalg.cholesky(cov_H)
        sol = torch.cholesky_solve(resid_H.unsqueeze(-1), chol).squeeze(-1)
        mahal_sq = torch.clamp((resid_H * sol).sum(), min=0.0)
        mahal = torch.sqrt(mahal_sq)
        pit = _chi2_cdf_from_mahal_sq(mahal_sq.view(1), int(hidden_idx.numel())).squeeze(0)

        rows.append(candidates[k])
        hidden_idx_list.append(hidden_idx)
        obs_idx_list.append(out['obs_idx'])
        pred_mean_list.append(mean_H)
        pred_cov_list.append(cov_H)
        residual_list.append(resid_H)
        mahal_sq_list.append(mahal_sq)
        mahal_list.append(mahal)
        pit_list.append(pit)
        factor_mean_rows.append(out['factor_mean'])
        obs_counts_list.append(int(out['obs_idx'].numel()))
        hidden_sizes_list.append(int(hidden_idx.numel()))

    device = Y.device
    dtype = Y.dtype
    r = int(L.shape[1])
    if len(rows) == 0:
        empty_idx = torch.empty((0, 2), dtype=torch.long, device=device)
        empty_score = torch.empty(0, dtype=dtype, device=device)
        return FactorJointScoresResult(
            factor_mode=factor_mode,
            split=str(split).lower(),
            indices=empty_idx,
            hidden_idx_list=[],
            obs_idx_list=[],
            hidden_sizes=torch.empty(0, dtype=torch.long, device=device),
            pred_mean_list=[],
            pred_cov_list=[],
            residual_list=[],
            mahal_sq=empty_score,
            mahal_score=empty_score,
            pit_score=empty_score,
            factor_mean=torch.empty((0, r), dtype=dtype, device=device),
            obs_counts=torch.empty(0, dtype=torch.long, device=device),
        )

    return FactorJointScoresResult(
        factor_mode=factor_mode,
        split=str(split).lower(),
        indices=torch.stack(rows, dim=0),
        hidden_idx_list=hidden_idx_list,
        obs_idx_list=obs_idx_list,
        hidden_sizes=torch.tensor(hidden_sizes_list, dtype=torch.long, device=device),
        pred_mean_list=pred_mean_list,
        pred_cov_list=pred_cov_list,
        residual_list=residual_list,
        mahal_sq=torch.stack(mahal_sq_list),
        mahal_score=torch.stack(mahal_list),
        pit_score=torch.stack(pit_list),
        factor_mean=torch.stack(factor_mean_rows, dim=0),
        obs_counts=torch.tensor(obs_counts_list, dtype=torch.long, device=device),
    )


def calibrate_factor_joint_quantiles(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    alpha: float = 0.1,
    factor_mode: str = 'observed',
    score_kind: str = 'mahalanobis',
    group_by: str = 'pattern',
    cal_point_mask: Optional[torch.Tensor] = None,
    cal_weights: Optional[torch.Tensor] = None,
    test_weight: float = 1.0,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[Any, Dict[str, Any]]:
    """
    Calibrate joint hidden-block thresholds.

    Parameters
    ----------
    score_kind:
      - 'mahalanobis': calibrate the raw Mahalanobis norm directly.
      - 'pit':         calibrate the chi-square PIT score U = F_{chi^2_d}(S^2).
    group_by:
      - 'pattern': separate threshold for each exact hidden pattern H.
      - 'size':    separate threshold for each hidden-block size |H|.
      - 'all':     one pooled threshold for all rows. Usually most sensible with score_kind='pit'.
    """
    score_kind = _normalize_joint_score_kind(score_kind)
    group_by = _normalize_joint_group_by(group_by)

    res = compute_factor_joint_scores(
        Y,
        M_mask,
        L,
        psi,
        s,
        a2,
        split='cal',
        factor_mode=factor_mode,
        point_mask=cal_point_mask,
        allow_oracle_missing=allow_oracle_missing,
        eps=eps,
        jitter=jitter,
    )

    if cal_weights is not None:
        cal_weights = cal_weights.to(device=Y.device, dtype=Y.dtype)
        if cal_weights.shape != M_mask.shape[:2]:
            raise ValueError(
                f"cal_weights must have shape (I,T)={tuple(M_mask.shape[:2])}, got {tuple(cal_weights.shape)}"
            )

    thresholds: Dict[Any, Dict[str, Any]] = {}
    score_tensor = res.mahal_score if score_kind == 'mahalanobis' else res.pit_score

    group_to_pos: Dict[Any, List[int]] = {}
    for pos in range(int(res.indices.shape[0])):
        key = _joint_group_key_from_hidden_idx(res.hidden_idx_list[pos], group_by=group_by)
        group_to_pos.setdefault(key, []).append(pos)

    for key, positions in group_to_pos.items():
        pos_t = torch.tensor(positions, dtype=torch.long, device=Y.device)
        vals = score_tensor.index_select(0, pos_t)
        w = None
        if cal_weights is not None:
            it_idx = res.indices.index_select(0, pos_t)
            w = cal_weights[it_idx[:, 0], it_idx[:, 1]]
        thr = _weighted_conformal_quantile(
            vals,
            alpha=alpha,
            cal_weights=w,
            test_weight=test_weight,
        )

        hidden_size = int(res.hidden_sizes[positions[0]].item())
        example_pattern = tuple(int(x) for x in res.hidden_idx_list[positions[0]].detach().cpu().tolist())
        info: Dict[str, Any] = {
            'group_key': key,
            'group_by': group_by,
            'score_kind': score_kind,
            'factor_mode': factor_mode,
            'alpha': float(alpha),
            'threshold': thr,
            'n_cal': int(len(positions)),
            'hidden_size': hidden_size,
            'example_pattern': example_pattern,
        }
        if score_kind == 'pit':
            info['pit_threshold'] = thr
            info['mahal_sq_threshold'] = _chi2_ppf_bisection(
                thr.view(1),
                df=hidden_size,
            ).squeeze(0)
            info['mahal_threshold'] = torch.sqrt(info['mahal_sq_threshold'])
        else:
            info['mahal_threshold'] = thr
            info['mahal_sq_threshold'] = thr * thr
        thresholds[key] = info

    return thresholds


def joint_region_contains_candidate(
    y_candidate: torch.Tensor,
    *,
    pred_mean: torch.Tensor,
    pred_cov: torch.Tensor,
    threshold: torch.Tensor,
    score_kind: str = 'mahalanobis',
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Test whether a candidate hidden block lies in the calibrated joint region.
    """
    score_kind = _normalize_joint_score_kind(score_kind)
    y_candidate = torch.as_tensor(y_candidate, dtype=pred_mean.dtype, device=pred_mean.device).reshape(-1)
    resid = y_candidate - pred_mean
    chol = torch.linalg.cholesky(pred_cov + eps * torch.eye(pred_cov.shape[0], dtype=pred_cov.dtype, device=pred_cov.device))
    sol = torch.cholesky_solve(resid.unsqueeze(-1), chol).squeeze(-1)
    mahal_sq = torch.clamp((resid * sol).sum(), min=0.0)
    if score_kind == 'pit':
        score = _chi2_cdf_from_mahal_sq(mahal_sq.view(1), int(pred_mean.numel())).squeeze(0)
    else:
        score = torch.sqrt(mahal_sq)
    return score <= threshold


def predict_factor_joint_regions(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    quantiles_by_group: Dict[Any, Dict[str, Any]],
    *,
    pred_point_mask: Optional[torch.Tensor] = None,
    factor_mode: Optional[str] = None,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[str, Any]:
    """
    Build implicit ellipsoidal regions for each row's hidden block.

    Returns one dict per row containing the hidden pattern, predictive mean/covariance,
    calibrated threshold, and the realized hidden-block score.
    """
    if not quantiles_by_group:
        raise ValueError('quantiles_by_group is empty.')

    sample_info = next(iter(quantiles_by_group.values()))
    group_by = _normalize_joint_group_by(sample_info['group_by'])
    score_kind = _normalize_joint_score_kind(sample_info['score_kind'])
    if factor_mode is None:
        factor_mode = sample_info['factor_mode']
    factor_mode = _normalize_joint_factor_mode(factor_mode)

    res = compute_factor_joint_scores(
        Y,
        M_mask,
        L,
        psi,
        s,
        a2,
        split='test',
        factor_mode=factor_mode,
        point_mask=pred_point_mask,
        allow_oracle_missing=allow_oracle_missing,
        eps=eps,
        jitter=jitter,
    )

    rows: List[Dict[str, Any]] = []
    for pos in range(int(res.indices.shape[0])):
        hidden_idx = res.hidden_idx_list[pos]
        key = _joint_group_key_from_hidden_idx(hidden_idx, group_by=group_by)
        if key not in quantiles_by_group:
            raise KeyError(
                f'No calibrated threshold found for group {key!r}. '
                f'Available groups: {list(quantiles_by_group.keys())[:10]}'
            )
        info = quantiles_by_group[key]
        row_dict: Dict[str, Any] = {
            'i': int(res.indices[pos, 0].item()),
            't': int(res.indices[pos, 1].item()),
            'hidden_idx': hidden_idx,
            'obs_idx': res.obs_idx_list[pos],
            'hidden_size': int(hidden_idx.numel()),
            'pred_mean': res.pred_mean_list[pos],
            'pred_cov': res.pred_cov_list[pos],
            'residual': res.residual_list[pos],
            'factor_mean': res.factor_mean[pos],
            'obs_count': int(res.obs_counts[pos].item()),
            'group_key': key,
            'score_kind': score_kind,
            'factor_mode': factor_mode,
            'threshold': info['threshold'],
            'mahal_score': res.mahal_score[pos],
            'mahal_sq': res.mahal_sq[pos],
            'pit_score': res.pit_score[pos],
            'covered_true_hidden': bool((res.pit_score[pos] <= info['threshold']).item()) if score_kind == 'pit'
                                 else bool((res.mahal_score[pos] <= info['threshold']).item()),
        }
        if 'mahal_threshold' in info:
            row_dict['mahal_threshold'] = info['mahal_threshold']
        if 'mahal_sq_threshold' in info:
            row_dict['mahal_sq_threshold'] = info['mahal_sq_threshold']
        rows.append(row_dict)

    return {
        'rows': rows,
        'group_by': group_by,
        'score_kind': score_kind,
        'factor_mode': factor_mode,
    }


def run_factor_joint_conformal(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
    *,
    alpha: float = 0.1,
    factor_mode: str = 'observed',
    score_kind: str = 'mahalanobis',
    group_by: str = 'pattern',
    cal_point_mask: Optional[torch.Tensor] = None,
    pred_point_mask: Optional[torch.Tensor] = None,
    cal_weights: Optional[torch.Tensor] = None,
    test_weight: float = 1.0,
    allow_oracle_missing: bool = False,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> Dict[str, Any]:
    """
    One-stop wrapper for joint hidden-block conformal:
      1) calibrate row-level Mahalanobis / PIT thresholds,
      2) build implicit ellipsoidal regions for the hidden block on prediction rows.
    """
    q = calibrate_factor_joint_quantiles(
        Y,
        M_mask,
        L,
        psi,
        s,
        a2,
        alpha=alpha,
        factor_mode=factor_mode,
        score_kind=score_kind,
        group_by=group_by,
        cal_point_mask=cal_point_mask,
        cal_weights=cal_weights,
        test_weight=test_weight,
        allow_oracle_missing=allow_oracle_missing,
        eps=eps,
        jitter=jitter,
    )
    regions = predict_factor_joint_regions(
        Y,
        M_mask,
        L,
        psi,
        s,
        a2,
        q,
        pred_point_mask=pred_point_mask,
        factor_mode=factor_mode,
        allow_oracle_missing=allow_oracle_missing,
        eps=eps,
        jitter=jitter,
    )
    joint_coverage = empirical_joint_coverage(regions)
    projected_intervals = project_factor_joint_regions_to_channels(regions, shape=tuple(Y.shape))
    projected_channel_coverage = summarize_projected_channel_coverage(
        Y,
        projected_intervals,
        point_mask=pred_point_mask,
    )
    return {
        'quantiles_by_group': q,
        'regions': regions,
        'joint_coverage': joint_coverage,
        'projected_intervals': projected_intervals,
        'projected_channel_coverage': projected_channel_coverage,
    }



def _safe_mean_bool(vals: Sequence[bool]) -> float:
    if len(vals) == 0:
        return float('nan')
    return float(sum(bool(v) for v in vals) / len(vals))


def empirical_joint_coverage(regions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute empirical joint coverage summaries from the output of
    ``predict_factor_joint_regions`` or the ``regions`` field of
    ``run_factor_joint_conformal``.

    Coverage is evaluated row-wise using the stored indicator
    ``covered_true_hidden`` for the realized hidden block.
    """
    rows = regions.get('rows', [])
    covered = [bool(row['covered_true_hidden']) for row in rows]

    by_hidden_size: Dict[int, Dict[str, Any]] = {}
    by_pattern: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    by_group: Dict[Any, Dict[str, Any]] = {}

    size_to_flags: Dict[int, List[bool]] = {}
    pattern_to_flags: Dict[Tuple[int, ...], List[bool]] = {}
    group_to_flags: Dict[Any, List[bool]] = {}

    for row in rows:
        flag = bool(row['covered_true_hidden'])
        hidden_idx = tuple(int(x) for x in row['hidden_idx'].detach().cpu().tolist())
        hidden_size = int(row['hidden_size'])
        group_key = row['group_key']

        size_to_flags.setdefault(hidden_size, []).append(flag)
        pattern_to_flags.setdefault(hidden_idx, []).append(flag)
        group_to_flags.setdefault(group_key, []).append(flag)

    for key, vals in size_to_flags.items():
        by_hidden_size[key] = {
            'hidden_size': int(key),
            'n': int(len(vals)),
            'coverage': _safe_mean_bool(vals),
        }

    for key, vals in pattern_to_flags.items():
        by_pattern[key] = {
            'pattern': key,
            'hidden_size': int(len(key)),
            'n': int(len(vals)),
            'coverage': _safe_mean_bool(vals),
        }

    for key, vals in group_to_flags.items():
        by_group[key] = {
            'group_key': key,
            'n': int(len(vals)),
            'coverage': _safe_mean_bool(vals),
        }

    return {
        'n_rows': int(len(rows)),
        'coverage': _safe_mean_bool(covered),
        'covered_count': int(sum(covered)),
        'miss_count': int(len(covered) - sum(covered)),
        'by_hidden_size': by_hidden_size,
        'by_pattern': by_pattern,
        'by_group': by_group,
    }



def project_factor_joint_regions_to_channels(
    regions: Dict[str, Any],
    *,
    shape: Optional[Tuple[int, int, int]] = None,
    fill_value: float = float('nan'),
) -> Dict[str, Any]:
    """
    Project each row's joint ellipsoidal hidden-block region onto its individual
    hidden coordinates.

    If a row-level region is
        {y_H : (y_H - mu_H)^T Sigma_H^{-1} (y_H - mu_H) <= c^2},
    then the projection onto coordinate j in H is
        [mu_j - c * sqrt(Sigma_H[j,j]), mu_j + c * sqrt(Sigma_H[j,j])].

    Parameters
    ----------
    regions:
        Output from ``predict_factor_joint_regions`` or the ``regions`` field of
        ``run_factor_joint_conformal``.
    shape:
        Optional full tensor shape (I, T, J). When provided, the function also
        returns dense lower/upper tensors of this shape filled with ``fill_value``
        outside the projected hidden coordinates.
    fill_value:
        Fill value for dense tensors outside projected hidden coordinates.
    """
    rows = regions.get('rows', [])

    dense_lower = None
    dense_upper = None
    if shape is not None:
        I, T, J = shape
        dense_lower = torch.full(shape, fill_value, dtype=rows[0]['pred_mean'].dtype if rows else torch.float32,
                                 device=rows[0]['pred_mean'].device if rows else None)
        dense_upper = torch.full(shape, fill_value, dtype=rows[0]['pred_mean'].dtype if rows else torch.float32,
                                 device=rows[0]['pred_mean'].device if rows else None)

    projected_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []

    for row in rows:
        hidden_idx = row['hidden_idx']
        pred_mean = row['pred_mean']
        pred_cov = row['pred_cov']
        if 'mahal_threshold' in row:
            mahal_thr = row['mahal_threshold']
        elif 'mahal_sq_threshold' in row:
            mahal_thr = torch.sqrt(row['mahal_sq_threshold'])
        else:
            raise KeyError('Each row must include mahal_threshold or mahal_sq_threshold to project intervals.')

        diag_sd = torch.sqrt(torch.diagonal(pred_cov).clamp_min(0.0))
        lower = pred_mean - mahal_thr * diag_sd
        upper = pred_mean + mahal_thr * diag_sd

        projected_rows.append({
            'i': row['i'],
            't': row['t'],
            'hidden_idx': hidden_idx,
            'lower': lower,
            'upper': upper,
            'pred_mean': pred_mean,
            'pred_var_diag': torch.diagonal(pred_cov),
            'mahal_threshold': mahal_thr,
            'group_key': row['group_key'],
            'score_kind': row['score_kind'],
            'factor_mode': row['factor_mode'],
        })

        for local_pos in range(int(hidden_idx.numel())):
            j = int(hidden_idx[local_pos].item())
            lo = lower[local_pos]
            hi = upper[local_pos]
            channel_rows.append({
                'i': row['i'],
                't': row['t'],
                'j': j,
                'lower': lo,
                'upper': hi,
                'pred_mean': pred_mean[local_pos],
                'pred_var': pred_cov[local_pos, local_pos],
                'hidden_size': int(hidden_idx.numel()),
                'group_key': row['group_key'],
            })
            if dense_lower is not None and dense_upper is not None:
                dense_lower[row['i'], row['t'], j] = lo
                dense_upper[row['i'], row['t'], j] = hi

    out: Dict[str, Any] = {
        'rows': projected_rows,
        'channel_rows': channel_rows,
    }
    if dense_lower is not None and dense_upper is not None:
        out['lower'] = dense_lower
        out['upper'] = dense_upper
    return out



def summarize_projected_channel_coverage(
    Y: torch.Tensor,
    projected_intervals: Dict[str, Any],
    *,
    point_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Compute empirical per-channel coverage summaries for the projected intervals.

    Parameters
    ----------
    Y:
        Tensor of realized hidden-block targets on the same scale used by the joint
        conformal module (in the current setting, the residual tensor).
    projected_intervals:
        Output of ``project_factor_joint_regions_to_channels``.
    point_mask:
        Optional (I,T) boolean mask restricting which rows contribute. This is mainly
        useful when you projected a larger set but want coverage on a subset.
    """
    if point_mask is not None:
        point_mask = point_mask.to(dtype=torch.bool, device=Y.device)
        if point_mask.shape != Y.shape[:2]:
            raise ValueError(f'point_mask must have shape (I,T)={tuple(Y.shape[:2])}, got {tuple(point_mask.shape)}')

    channel_rows = projected_intervals.get('channel_rows', [])
    J = int(Y.shape[2])

    total = 0
    covered_total = 0
    channel_counts = torch.zeros(J, dtype=torch.long, device=Y.device)
    channel_covered = torch.zeros(J, dtype=torch.long, device=Y.device)

    by_hidden_size_flags: Dict[int, List[bool]] = {}
    by_channel_rows: Dict[int, List[Dict[str, Any]]] = {j: [] for j in range(J)}

    realized_rows: List[Dict[str, Any]] = []
    for row in channel_rows:
        i = int(row['i'])
        t = int(row['t'])
        j = int(row['j'])
        if point_mask is not None and not bool(point_mask[i, t].item()):
            continue
        y_true = Y[i, t, j]
        covered = bool(((y_true >= row['lower']) & (y_true <= row['upper'])).item())
        total += 1
        covered_total += int(covered)
        channel_counts[j] += 1
        channel_covered[j] += int(covered)
        by_hidden_size_flags.setdefault(int(row['hidden_size']), []).append(covered)

        realized = {
            **row,
            'y_true': y_true,
            'covered': covered,
        }
        by_channel_rows[j].append(realized)
        realized_rows.append(realized)

    by_channel: Dict[int, Dict[str, Any]] = {}
    for j in range(J):
        n_j = int(channel_counts[j].item())
        cov_j = float(channel_covered[j].item() / n_j) if n_j > 0 else float('nan')
        by_channel[j] = {
            'channel': int(j),
            'n': n_j,
            'coverage': cov_j,
            'covered_count': int(channel_covered[j].item()),
            'miss_count': int(n_j - channel_covered[j].item()),
            'rows': by_channel_rows[j],
        }

    by_hidden_size: Dict[int, Dict[str, Any]] = {}
    for size, vals in by_hidden_size_flags.items():
        by_hidden_size[size] = {
            'hidden_size': int(size),
            'n': int(len(vals)),
            'coverage': _safe_mean_bool(vals),
        }

    overall_cov = float(covered_total / total) if total > 0 else float('nan')
    return {
        'n_projected_coordinates': int(total),
        'coverage': overall_cov,
        'covered_count': int(covered_total),
        'miss_count': int(total - covered_total),
        'by_channel': by_channel,
        'by_hidden_size': by_hidden_size,
        'rows': realized_rows,
    }