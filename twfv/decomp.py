import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def higher_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Split-conformal 'higher' quantile:
        q = sorted_scores[ ceil((n+1)*(1-alpha)) - 1 ]
    """
    scores = scores.reshape(-1)
    if scores.numel() == 0:
        raise ValueError("No scores provided.")
    scores_sorted = torch.sort(scores).values
    n = scores_sorted.numel()
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return scores_sorted[k - 1]


def weighted_higher_quantile(
    scores: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Weighted empirical quantile with 'higher'-style behavior:
    smallest score whose cumulative normalized weight >= 1 - alpha.

    This is useful if you later want latent-space localization weights.
    """
    scores = scores.reshape(-1)
    weights = weights.reshape(-1)
    if scores.numel() == 0:
        raise ValueError("No scores provided.")
    if scores.numel() != weights.numel():
        raise ValueError("scores and weights must have the same length.")

    weights = torch.clamp(weights, min=0.0)
    wsum = weights.sum()
    if wsum <= eps:
        return higher_quantile(scores, alpha)

    order = torch.argsort(scores)
    s = scores[order]
    w = weights[order] / wsum
    cdf = torch.cumsum(w, dim=0)
    idx = torch.searchsorted(cdf, torch.tensor(1.0 - alpha, device=cdf.device, dtype=cdf.dtype))
    idx = min(int(idx.item()), s.numel() - 1)
    return s[idx]


def unique_row_indices_from_flat_ijt(flat_idx: np.ndarray, J: int, T: int) -> List[Tuple[int, int]]:
    """
    Convert flattened channel-level indices k = i*(J*T) + j*T + t
    into unique row-level indices (i,t).

    Important:
    This is only a compatibility helper for your current pipeline.
    For this hierarchical score, the clean split should really be done
    at the row level from the beginning.
    """
    flat_idx = np.asarray(flat_idx, dtype=int).ravel()
    if flat_idx.size == 0:
        return []

    i = flat_idx // (J * T)
    t = flat_idx % T
    rows = np.stack([i, t], axis=1)
    rows = np.unique(rows, axis=0)
    return [(int(ii), int(tt)) for ii, tt in rows]


def all_nonempty_row_indices(M_mask: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Return all (i,t) rows with at least one observed channel.
    """
    I, T, _ = M_mask.shape
    out: List[Tuple[int, int]] = []
    for i in range(I):
        for t in range(T):
            if bool(M_mask[i, t].any()):
                out.append((i, t))
    return out


@dataclass
class HierarchicalRowScore:
    total: float
    latent: float
    idio: float
    num_obs: int
    f_post: torch.Tensor
    eps_hat_obs: torch.Tensor


class HierarchicalFactorSplitCP:
    """
    Hierarchical factor-score split conformal wrapper.

    Assumes you already have fitted factor-model outputs for the rows you want to score:
        L   : (J, r)
        a2  : (I, T, r)
        psi : (J,)
        s   : (I,)

    The wrapper calibrates a scalar nonconformity score per row (i,t),
    and returns a conservative channelwise prediction box by inverting:
        score = max(latent_score, idio_score)

    If you provide partially observed outputs at prediction time, the method
    conditions on them via Gaussian posterior updating of the latent factor.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        jitter: float = 1e-6,
        eps: float = 1e-12,
    ):
        self.alpha = float(alpha)
        self.jitter = float(jitter)
        self.eps = float(eps)

        self.L: Optional[torch.Tensor] = None
        self.a2: Optional[torch.Tensor] = None
        self.psi: Optional[torch.Tensor] = None
        self.s: Optional[torch.Tensor] = None

        self.qhat: Optional[torch.Tensor] = None
        self.cal_scores_: Optional[torch.Tensor] = None
        self.cal_rows_: Optional[List[Tuple[int, int]]] = None

    @classmethod
    def from_run_em_like(
        cls,
        em_out: Sequence[Any],
        alpha: float = 0.1,
        jitter: float = 1e-6,
        eps: float = 1e-12,
    ) -> "HierarchicalFactorSplitCP":
        """
        Build from your existing run_em_like output.

        Expected leading entries:
            L, a2, psi, s, ...
        """
        obj = cls(alpha=alpha, jitter=jitter, eps=eps)
        if len(em_out) < 4:
            raise ValueError("em_out must start with (L, a2, psi, s, ...).")
        obj.set_factor_params(em_out[0], em_out[1], em_out[2], em_out[3])
        return obj

    def set_factor_params(
        self,
        L: torch.Tensor,
        a2: torch.Tensor,
        psi: torch.Tensor,
        s: torch.Tensor,
    ) -> None:
        self.L = L
        self.a2 = a2
        self.psi = psi
        self.s = s

    def _check_ready(self) -> None:
        if self.L is None or self.a2 is None or self.psi is None or self.s is None:
            raise RuntimeError("Factor parameters are not set. Call set_factor_params(...) first.")

    def _device(self) -> torch.device:
        self._check_ready()
        return self.L.device

    def _dtype(self) -> torch.dtype:
        self._check_ready()
        return self.L.dtype

    def _row_prior_cov(self, i: int, t: int) -> torch.Tensor:
        """
        A_it = diag(a2[i,t]) in factor space.
        """
        self._check_ready()
        return torch.diag(torch.clamp(self.a2[i, t], min=self.eps))

    def _row_idio_diag(self, i: int) -> torch.Tensor:
        """
        D_i = s_i^2 * diag(psi)
        """
        self._check_ready()
        return (self.s[i] ** 2) * torch.clamp(self.psi, min=self.eps)

    def _posterior_factor_given_obs(
        self,
        i: int,
        t: int,
        residual_obs: torch.Tensor,
        obs_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian posterior of latent factor f | observed residuals.

        Prior:
            f ~ N(0, A_it),  A_it = diag(a2[i,t])

        Observation model on observed channels:
            r_obs = L_obs f + e_obs,
            e_obs ~ N(0, D_obs), D_obs = s_i^2 diag(psi_obs)

        Returns
        -------
        f_post : (r,)
            posterior mean of latent factor
        A_post : (r, r)
            posterior covariance of latent factor
        """
        self._check_ready()

        A = self._row_prior_cov(i, t)  # (r,r)
        invA = torch.diag(1.0 / torch.diagonal(A))

        if obs_idx.numel() == 0:
            r = A.shape[0]
            return torch.zeros(r, device=A.device, dtype=A.dtype), A

        L_obs = self.L.index_select(0, obs_idx)  # (Jo,r)
        D_obs_diag = self._row_idio_diag(i).index_select(0, obs_idx)  # (Jo,)
        invD_obs = torch.diag(1.0 / torch.clamp(D_obs_diag, min=self.eps))

        precision = invA + L_obs.T @ invD_obs @ L_obs
        precision = precision + self.jitter * torch.eye(
            precision.shape[0], device=precision.device, dtype=precision.dtype
        )
        A_post = torch.linalg.inv(precision)
        f_post = A_post @ L_obs.T @ invD_obs @ residual_obs
        return f_post, A_post

    def score_row(
        self,
        Y: torch.Tensor,
        M_mask: torch.Tensor,
        i: int,
        t: int,
        mu: Optional[torch.Tensor] = None,
    ) -> HierarchicalRowScore:
        """
        Compute hierarchical score for a single row (i,t).

        Y      : (I,T,J)
        M_mask : (I,T,J) bool, True = observed
        mu     : optional mean tensor (I,T,J); if None, zero mean is used
        """
        self._check_ready()

        device = self._device()
        dtype = self._dtype()

        y_row = Y[i, t].to(device=device, dtype=dtype)
        m_row = M_mask[i, t].to(device=device)
        obs_idx = torch.nonzero(m_row, as_tuple=False).squeeze(1)

        if mu is None:
            mu_row = torch.zeros_like(y_row)
        else:
            mu_row = mu[i, t].to(device=device, dtype=dtype)

        if obs_idx.numel() == 0:
            return HierarchicalRowScore(
                total=float("nan"),
                latent=float("nan"),
                idio=float("nan"),
                num_obs=0,
                f_post=torch.zeros(self.L.shape[1], device=device, dtype=dtype),
                eps_hat_obs=torch.empty(0, device=device, dtype=dtype),
            )

        residual_obs = (y_row - mu_row).index_select(0, obs_idx)
        f_post, _ = self._posterior_factor_given_obs(i=i, t=t, residual_obs=residual_obs, obs_idx=obs_idx)

        # latent score: ||f_post||_{A^{-1}}
        a2_it = torch.clamp(self.a2[i, t], min=self.eps)
        latent_score = torch.sqrt(torch.sum((f_post ** 2) / a2_it).clamp_min(0.0))

        # idiosyncratic residual on observed channels
        L_obs = self.L.index_select(0, obs_idx)
        eps_hat_obs = residual_obs - L_obs @ f_post
        idio_sd_obs = self.s[i] * torch.sqrt(torch.clamp(self.psi.index_select(0, obs_idx), min=self.eps))
        idio_z = torch.abs(eps_hat_obs) / torch.clamp(idio_sd_obs, min=self.eps)
        idio_score = torch.max(idio_z)

        total = torch.maximum(latent_score, idio_score)

        return HierarchicalRowScore(
            total=float(total.detach().cpu().item()),
            latent=float(latent_score.detach().cpu().item()),
            idio=float(idio_score.detach().cpu().item()),
            num_obs=int(obs_idx.numel()),
            f_post=f_post,
            eps_hat_obs=eps_hat_obs,
        )

    def score_rows(
        self,
        Y: torch.Tensor,
        M_mask: torch.Tensor,
        row_indices: Optional[Sequence[Tuple[int, int]]] = None,
        mu: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Score multiple rows and return tensors/dicts for diagnostics.
        """
        if row_indices is None:
            row_indices = all_nonempty_row_indices(M_mask)

        totals = []
        latents = []
        idios = []
        kept_rows = []

        for (i, t) in row_indices:
            out = self.score_row(Y=Y, M_mask=M_mask, i=i, t=t, mu=mu)
            if np.isnan(out.total):
                continue
            totals.append(out.total)
            latents.append(out.latent)
            idios.append(out.idio)
            kept_rows.append((i, t))

        device = self._device()
        dtype = self._dtype()

        return {
            "rows": kept_rows,
            "total": torch.tensor(totals, device=device, dtype=dtype),
            "latent": torch.tensor(latents, device=device, dtype=dtype),
            "idio": torch.tensor(idios, device=device, dtype=dtype),
        }

    def calibrate(
        self,
        Y_cal: torch.Tensor,
        M_cal: torch.Tensor,
        row_indices: Optional[Sequence[Tuple[int, int]]] = None,
        mu_cal: Optional[torch.Tensor] = None,
        row_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calibrate qhat on row-level calibration data.

        row_weights:
            optional weights for weighted/localized quantile.
            If provided, must align with the rows that are actually kept.
        """
        scored = self.score_rows(Y=Y_cal, M_mask=M_cal, row_indices=row_indices, mu=mu_cal)
        scores = scored["total"]
        rows = scored["rows"]

        if row_weights is None:
            qhat = higher_quantile(scores, self.alpha)
        else:
            row_weights = row_weights.reshape(-1).to(device=scores.device, dtype=scores.dtype)
            if row_indices is None:
                if row_weights.numel() != len(rows):
                    raise ValueError("row_weights must match the number of kept calibration rows.")
                weights_kept = row_weights
            else:
                if row_weights.numel() != len(row_indices):
                    raise ValueError("row_weights must match row_indices length.")
                # keep only weights corresponding to rows that survived scoring
                row_to_weight = {tuple(row_indices[k]): row_weights[k] for k in range(len(row_indices))}
                weights_kept = torch.stack([row_to_weight[(i, t)] for (i, t) in rows], dim=0)
            qhat = weighted_higher_quantile(scores, weights_kept, self.alpha)

        self.qhat = qhat
        self.cal_scores_ = scores
        self.cal_rows_ = rows
        return qhat

    def predict_box_row(
        self,
        i: int,
        t: int,
        mu_row: torch.Tensor,
        qhat: Optional[torch.Tensor] = None,
        observed_y: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        degenerate_observed: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Conservative channelwise inversion of the hierarchical score.

        If observed_y / observed_mask are provided, we refine the latent factor
        by conditioning on those observed channels.

        Returned box:
            center_j +/- q * ( latent_sd_j + idio_sd_j )

        where latent_sd_j = sqrt( L_j A_post L_j^T )
        and idio_sd_j   = s_i * sqrt(psi_j).

        This is conservative but easy to compute.
        """
        self._check_ready()
        device = self._device()
        dtype = self._dtype()

        if qhat is None:
            if self.qhat is None:
                raise RuntimeError("No qhat stored. Call calibrate(...) or pass qhat explicitly.")
            q = self.qhat.to(device=device, dtype=dtype)
        else:
            q = qhat.to(device=device, dtype=dtype)

        mu_row = mu_row.to(device=device, dtype=dtype)

        if observed_y is None or observed_mask is None:
            f_post = torch.zeros(self.L.shape[1], device=device, dtype=dtype)
            A_post = self._row_prior_cov(i, t)
            obs_idx = torch.empty(0, dtype=torch.long, device=device)
            observed_y = None
            observed_mask = None
        else:
            observed_y = observed_y.to(device=device, dtype=dtype)
            observed_mask = observed_mask.to(device=device)
            obs_idx = torch.nonzero(observed_mask, as_tuple=False).squeeze(1)
            residual_obs = (observed_y - mu_row).index_select(0, obs_idx)
            f_post, A_post = self._posterior_factor_given_obs(
                i=i, t=t, residual_obs=residual_obs, obs_idx=obs_idx
            )

        # predictive center = mean + latent posterior mean contribution
        latent_center = self.L @ f_post
        center = mu_row + latent_center

        # channelwise latent uncertainty after conditioning
        LA = self.L @ A_post
        latent_var = torch.sum(LA * self.L, dim=1).clamp_min(0.0)
        latent_sd = torch.sqrt(latent_var)

        # channelwise idiosyncratic uncertainty
        idio_sd = self.s[i] * torch.sqrt(torch.clamp(self.psi, min=self.eps))

        half_width = q * (latent_sd + idio_sd)
        lower = center - half_width
        upper = center + half_width

        if degenerate_observed and observed_y is not None and observed_mask is not None and obs_idx.numel() > 0:
            lower = lower.clone()
            upper = upper.clone()
            lower[obs_idx] = observed_y[obs_idx]
            upper[obs_idx] = observed_y[obs_idx]

        aux = {
            "center": center,
            "half_width": half_width,
            "latent_center": latent_center,
            "latent_sd": latent_sd,
            "idio_sd": idio_sd,
            "f_post": f_post,
            "A_post": A_post,
        }
        return lower, upper, aux

    def predict_box_dataset(
        self,
        row_indices: Sequence[Tuple[int, int]],
        mu: torch.Tensor,
        qhat: Optional[torch.Tensor] = None,
        observed_Y: Optional[torch.Tensor] = None,
        observed_M: Optional[torch.Tensor] = None,
        degenerate_observed: bool = True,
    ) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """
        Batch wrapper around predict_box_row.
        """
        out: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        for (i, t) in row_indices:
            mu_row = mu[i, t]
            if observed_Y is None or observed_M is None:
                lower, upper, aux = self.predict_box_row(
                    i=i,
                    t=t,
                    mu_row=mu_row,
                    qhat=qhat,
                    observed_y=None,
                    observed_mask=None,
                    degenerate_observed=degenerate_observed,
                )
            else:
                lower, upper, aux = self.predict_box_row(
                    i=i,
                    t=t,
                    mu_row=mu_row,
                    qhat=qhat,
                    observed_y=observed_Y[i, t],
                    observed_mask=observed_M[i, t],
                    degenerate_observed=degenerate_observed,
                )
            out[(i, t)] = {
                "lower": lower,
                "upper": upper,
                **aux,
            }
        return out


# ---------------------------------------------------------------------
# Example usage in your current pipeline
# ---------------------------------------------------------------------
#
# 1) Fit your factor model on residual tensor Y_resid_train-like data.
#    If you are using zero-mean innovations in the simulation, mu_full can be zeros.
#
#    em_out = run_em_like(
#        Y_resid_full,      # (I,T,J) residual tensor for the rows you want a2 on
#        M_mask_full,       # (I,T,J) bool mask
#        r=r,
#        M_ctrl=M_ctrl,
#        degree=degree,
#        device=device,
#        use_woodbury=True,
#        return_history=True,
#    )
#
# 2) Wrap the fitted params.
#
#    hfcp = HierarchicalFactorSplitCP.from_run_em_like(em_out, alpha=0.1)
#
# 3) Build row-level calibration indices.
#    Strongly preferred: split at row level from the start.
#    If you only have old flat indices k=i*(J*T)+j*T+t, convert them:
#
#    cal_rows = unique_row_indices_from_flat_ijt(cal_idx_flat, J=J, T=T)
#
# 4) Calibrate:
#
#    mu_full = torch.zeros_like(Y_resid_full)   # or your fitted mean tensor
#    qhat = hfcp.calibrate(
#        Y_cal=Y_resid_full,
#        M_cal=M_mask_full,
#        row_indices=cal_rows,
#        mu_cal=mu_full,
#    )
#
# 5) Predict a conservative box for a test row (i,t).
#    If nothing is observed at test time yet:
#
#    lower, upper, aux = hfcp.predict_box_row(
#        i=i_test,
#        t=t_test,
#        mu_row=mu_full[i_test, t_test],
#        qhat=qhat,
#    )
#
# 6) If some channels are already observed and you want refinement:
#
#    lower_ref, upper_ref, aux_ref = hfcp.predict_box_row(
#        i=i_test,
#        t=t_test,
#        mu_row=mu_full[i_test, t_test],
#        qhat=qhat,
#        observed_y=Y_resid_full[i_test, t_test],   # or true partially revealed row
#        observed_mask=M_mask_full[i_test, t_test], # mask of revealed channels
#        degenerate_observed=True,
#    )
#