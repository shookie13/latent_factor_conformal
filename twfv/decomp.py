import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_numpy_array(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def higher_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split-conformal 'higher' quantile:
        q = sorted_scores[ ceil((n+1)*(1-alpha)) - 1 ]
    """
    scores = _as_numpy_array(scores).reshape(-1)
    if scores.size == 0:
        raise ValueError("No scores provided.")
    scores_sorted = np.sort(scores)
    n = scores_sorted.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(scores_sorted[k - 1])


def weighted_higher_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    eps: float = 1e-12,
) -> float:
    """
    Weighted empirical quantile with 'higher'-style behavior:
    smallest score whose cumulative normalized weight >= 1 - alpha.

    This is useful if you later want latent-space localization weights.
    """
    scores = _as_numpy_array(scores).reshape(-1)
    weights = _as_numpy_array(weights).reshape(-1)
    if scores.size == 0:
        raise ValueError("No scores provided.")
    if scores.size != weights.size:
        raise ValueError("scores and weights must have the same length.")

    weights = np.clip(weights, 0.0, None)
    wsum = weights.sum()
    if wsum <= eps:
        return higher_quantile(scores, alpha)

    order = np.argsort(scores)
    s = scores[order]
    w = weights[order] / wsum
    cdf = np.cumsum(w)
    idx = np.searchsorted(cdf, 1.0 - alpha, side="left")
    idx = min(int(idx), s.size - 1)
    return float(s[idx])


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


def all_nonempty_row_indices(M_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return all (i,t) rows with at least one observed channel.
    """
    I, T, _ = M_mask.shape
    out: List[Tuple[int, int]] = []
    for i in range(I):
        for t in range(T):
            if bool(np.any(M_mask[i, t])):
                out.append((i, t))
    return out


@dataclass
class HierarchicalRowScore:
    total: float
    latent: float
    idio: float
    num_obs: int
    f_post: np.ndarray
    eps_hat_obs: np.ndarray


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

        self.L: Optional[np.ndarray] = None
        self.a2: Optional[np.ndarray] = None
        self.psi: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None

        self.qhat: Optional[float] = None
        self.cal_scores_: Optional[np.ndarray] = None
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
        L: np.ndarray,
        a2: np.ndarray,
        psi: np.ndarray,
        s: np.ndarray,
    ) -> None:
        self.L = _as_numpy_array(L)
        self.a2 = _as_numpy_array(a2, dtype=self.L.dtype)
        self.psi = _as_numpy_array(psi, dtype=self.L.dtype)
        self.s = _as_numpy_array(s, dtype=self.L.dtype)

    def _check_ready(self) -> None:
        if self.L is None or self.a2 is None or self.psi is None or self.s is None:
            raise RuntimeError("Factor parameters are not set. Call set_factor_params(...) first.")

    def _dtype(self) -> np.dtype:
        self._check_ready()
        return self.L.dtype

    def _row_prior_cov(self, i: int, t: int) -> np.ndarray:
        """
        A_it = diag(a2[i,t]) in factor space.
        """
        self._check_ready()
        return np.diag(np.clip(self.a2[i, t], self.eps, None))

    def _row_idio_diag(self, i: int) -> np.ndarray:
        """
        D_i = s_i^2 * diag(psi)
        """
        self._check_ready()
        return (self.s[i] ** 2) * np.clip(self.psi, self.eps, None)

    def _posterior_factor_given_obs(
        self,
        i: int,
        t: int,
        residual_obs: np.ndarray,
        obs_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        invA = np.diag(1.0 / np.diag(A))

        if obs_idx.size == 0:
            r = A.shape[0]
            return np.zeros(r, dtype=A.dtype), A

        L_obs = self.L[obs_idx]  # (Jo,r)
        D_obs_diag = self._row_idio_diag(i)[obs_idx]  # (Jo,)
        invD_obs = np.diag(1.0 / np.clip(D_obs_diag, self.eps, None))

        precision = invA + L_obs.T @ invD_obs @ L_obs
        precision = precision + self.jitter * np.eye(precision.shape[0], dtype=precision.dtype)
        A_post = np.linalg.inv(precision)
        f_post = A_post @ L_obs.T @ invD_obs @ residual_obs
        return f_post, A_post

    def score_row(
        self,
        Y: np.ndarray,
        M_mask: np.ndarray,
        i: int,
        t: int,
        mu: Optional[np.ndarray] = None,
    ) -> HierarchicalRowScore:
        """
        Compute hierarchical score for a single row (i,t).

        Y      : (I,T,J)
        M_mask : (I,T,J) bool, True = observed
        mu     : optional mean tensor (I,T,J); if None, zero mean is used
        """
        self._check_ready()

        dtype = self._dtype()

        y_row = _as_numpy_array(Y[i, t], dtype=dtype)
        m_row = _as_numpy_array(M_mask[i, t], dtype=bool)
        obs_idx = np.flatnonzero(m_row)

        if mu is None:
            mu_row = np.zeros_like(y_row)
        else:
            mu_row = _as_numpy_array(mu[i, t], dtype=dtype)

        if obs_idx.size == 0:
            return HierarchicalRowScore(
                total=float("nan"),
                latent=float("nan"),
                idio=float("nan"),
                num_obs=0,
                f_post=np.zeros(self.L.shape[1], dtype=dtype),
                eps_hat_obs=np.empty(0, dtype=dtype),
            )

        try:
            residual_obs = (y_row - mu_row)[obs_idx]
        except Exception as e:
            print("Error while computing residual_obs in score_row:")
            print("y_row:", y_row)
            print("M_mask[i,t]:", M_mask[i,t])
            print("mu_row:", mu_row)
            print("obs_idx:", obs_idx)
            raise
        f_post, _ = self._posterior_factor_given_obs(i=i, t=t, residual_obs=residual_obs, obs_idx=obs_idx)

        # latent score: ||f_post||_{A^{-1}}
        a2_it = np.clip(self.a2[i, t], self.eps, None)
        latent_score = float(np.sqrt(np.clip(np.sum((f_post ** 2) / a2_it), 0.0, None)))

        # idiosyncratic residual on observed channels
        L_obs = self.L[obs_idx]
        eps_hat_obs = residual_obs - L_obs @ f_post
        idio_sd_obs = self.s[i] * np.sqrt(np.clip(self.psi[obs_idx], self.eps, None))
        idio_z = np.abs(eps_hat_obs) / np.clip(idio_sd_obs, self.eps, None)
        idio_score = float(np.max(idio_z))

        total = max(latent_score, idio_score)

        return HierarchicalRowScore(
            total=float(total),
            latent=float(latent_score),
            idio=float(idio_score),
            num_obs=int(obs_idx.size),
            f_post=f_post,
            eps_hat_obs=eps_hat_obs,
        )

    def score_rows(
        self,
        Y: np.ndarray,
        M_mask: np.ndarray,
        row_indices: Optional[Sequence[Tuple[int, int]]] = None,
        mu: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Score multiple rows and return arrays/dicts for diagnostics.
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

        dtype = self._dtype()

        return {
            "rows": kept_rows,
            "total": np.asarray(totals, dtype=dtype),
            "latent": np.asarray(latents, dtype=dtype),
            "idio": np.asarray(idios, dtype=dtype),
        }

    def calibrate(
        self,
        Y_cal: np.ndarray,
        M_cal: np.ndarray,
        row_indices: Optional[Sequence[Tuple[int, int]]] = None,
        mu_cal: Optional[np.ndarray] = None,
        row_weights: Optional[np.ndarray] = None,
    ) -> float:
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
            row_weights = _as_numpy_array(row_weights, dtype=scores.dtype).reshape(-1)
            if row_indices is None:
                if row_weights.size != len(rows):
                    raise ValueError("row_weights must match the number of kept calibration rows.")
                weights_kept = row_weights
            else:
                if row_weights.size != len(row_indices):
                    raise ValueError("row_weights must match row_indices length.")
                # keep only weights corresponding to rows that survived scoring
                row_to_weight = {tuple(row_indices[k]): row_weights[k] for k in range(len(row_indices))}
                weights_kept = np.asarray([row_to_weight[(i, t)] for (i, t) in rows], dtype=scores.dtype)
            qhat = weighted_higher_quantile(scores, weights_kept, self.alpha)

        self.qhat = float(qhat)
        self.cal_scores_ = scores
        self.cal_rows_ = rows
        return self.qhat

    def predict_box_row(
        self,
        i: int,
        t: int,
        mu_row: np.ndarray,
        qhat: Optional[float] = None,
        observed_y: Optional[np.ndarray] = None,
        observed_mask: Optional[np.ndarray] = None,
        degenerate_observed: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
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
        dtype = self._dtype()

        if qhat is None:
            if self.qhat is None:
                raise RuntimeError("No qhat stored. Call calibrate(...) or pass qhat explicitly.")
            q = dtype.type(self.qhat)
        else:
            q = dtype.type(qhat)

        mu_row = _as_numpy_array(mu_row, dtype=dtype)

        if observed_y is None or observed_mask is None:
            f_post = np.zeros(self.L.shape[1], dtype=dtype)
            A_post = self._row_prior_cov(i, t)
            obs_idx = np.empty(0, dtype=int)
            observed_y = None
            observed_mask = None
        else:
            observed_y = _as_numpy_array(observed_y, dtype=dtype)
            observed_mask = _as_numpy_array(observed_mask, dtype=bool)
            obs_idx = np.flatnonzero(observed_mask)
            residual_obs = (observed_y - mu_row)[obs_idx]
            f_post, A_post = self._posterior_factor_given_obs(
                i=i, t=t, residual_obs=residual_obs, obs_idx=obs_idx
            )

        # predictive center = mean + latent posterior mean contribution
        latent_center = self.L @ f_post
        center = mu_row + latent_center

        # channelwise latent uncertainty after conditioning
        LA = self.L @ A_post
        latent_var = np.clip(np.sum(LA * self.L, axis=1), 0.0, None)
        latent_sd = np.sqrt(latent_var)

        # channelwise idiosyncratic uncertainty
        idio_sd = self.s[i] * np.sqrt(np.clip(self.psi, self.eps, None))

        half_width = q * (latent_sd + idio_sd)
        lower = center - half_width
        upper = center + half_width

        if degenerate_observed and observed_y is not None and observed_mask is not None and obs_idx.size > 0:
            lower = lower.copy()
            upper = upper.copy()
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
        mu: np.ndarray,
        qhat: Optional[float] = None,
        observed_Y: Optional[np.ndarray] = None,
        observed_M: Optional[np.ndarray] = None,
        degenerate_observed: bool = True,
    ) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """
        Batch wrapper around predict_box_row.
        """
        out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
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
#    mu_full = np.zeros_like(Y_resid_full)      # or your fitted mean tensor
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