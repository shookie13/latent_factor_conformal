import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, Any

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
