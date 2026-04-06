import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2


def _flatten_rows_lastdim(arr: np.ndarray) -> np.ndarray:
    """
    Flatten all leading dimensions into one row dimension, keep the last dim.
    Example:
        (I, T, J) -> (I*T, J)
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError("Expected at least 2 dimensions.")
    return arr.reshape(-1, arr.shape[-1])


def _flatten_rows_last2dims(arr: np.ndarray) -> np.ndarray:
    """
    Flatten all leading dimensions into one row dimension, keep the last 2 dims.
    Example:
        (I, T, J, J) -> (I*T, J, J)
    """
    arr = np.asarray(arr)
    if arr.ndim < 3:
        raise ValueError("Expected at least 3 dimensions.")
    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])


def _higher_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split-conformal 'higher' quantile:
        q = sorted_scores[ ceil((n+1)*(1-alpha)) - 1 ].
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("No calibration scores were produced.")
    s = np.sort(scores)
    n = s.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _solve_spd(A: np.ndarray, b: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """
    Numerically stable solve for symmetric positive definite-ish systems.
    """
    A = _symmetrize(np.asarray(A, dtype=float))
    n = A.shape[0]
    A = A + jitter * np.eye(n)
    return np.linalg.solve(A, b)


def _safe_inv_spd(A: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    A = _symmetrize(np.asarray(A, dtype=float))
    n = A.shape[0]
    A = A + jitter * np.eye(n)
    return np.linalg.inv(A)


def _psd_sqrt_factor_numpy(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Return F such that approximately M ≈ F.T @ F, clipping tiny negative eigvals.
    """
    M = _symmetrize(np.asarray(M, dtype=float))
    evals, evecs = np.linalg.eigh(M)
    evals = np.clip(evals, 0.0, None)
    keep = evals > eps
    if not np.any(keep):
        return np.zeros((0, M.shape[0]), dtype=float)
    return (np.sqrt(evals[keep])[:, None] * evecs[:, keep].T)


@dataclass
class ScoreRecord:
    row: int
    score: float
    raw_score: float
    observed_idx: np.ndarray
    target_idx: np.ndarray


class PartialOutputConformalCP:
    """
    Partially observed-output conformal predictor.

    Supports two score families:
      - score_kind="mahalanobis"
      - score_kind="hierarchical"

    Supports two calibration modes:
      - calibration_mode="observed"
          Calibrate on the actually observed subvector only.
          This is the cleanest choice for Mahalanobis + PIT.
      - calibration_mode="pseudo_mask"
          Randomly split the observed coordinates into
          revealed part R and pseudo-hidden part H, then calibrate on H | R.
          This is the better-aligned choice when the deployment task is
          "predict missing coordinates from observed coordinates."

    Parameters
    ----------
    mu : array, shape (..., J)
        Row-wise predictive means.
    Sigma : array, shape (..., J, J), optional
        Row-wise predictive covariances. Needed for Mahalanobis mode if factor
        parameters are not supplied.
    L : array, shape (J, r), optional
        Loading matrix. Needed for hierarchical mode.
    A_diag : array, shape (..., r), optional
        Row-wise latent variances (diagonal of A_m).
    D_diag : array, shape (..., J), optional
        Row-wise idiosyncratic diagonal variances (diagonal of D_m).
    alpha : float
        Target miscoverage level.
    score_kind : {"mahalanobis", "hierarchical"}
    calibration_mode : {"observed", "pseudo_mask"}
    use_pit : bool
        If True:
          - Mahalanobis mode uses chi-square PIT.
          - Hierarchical mode currently supports use_pit=False only.
    pseudo_hidden_frac : float
        Fraction of observed coordinates to pseudo-hide in pseudo_mask mode.
    min_hidden : int
        Minimum number of pseudo-hidden coordinates.
    n_mc : int
        Reserved for future Monte Carlo extensions; not required in the current
        Mahalanobis implementation.
    random_state : int or None
    jitter : float
        Numerical stabilization for covariance solves.
    solver : str
        CVXPY solver name for hierarchical exact projections / conditional scores.
    solver_opts : dict or None
        Passed into CVXPY solve().
    """

    def __init__(
        self,
        mu: np.ndarray,
        *,
        Sigma: Optional[np.ndarray] = None,
        L: Optional[np.ndarray] = None,
        A_diag: Optional[np.ndarray] = None,
        D_diag: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        score_kind: str = "mahalanobis",
        calibration_mode: str = "pseudo_mask",
        use_pit: bool = True,
        pseudo_hidden_frac: float = 0.3,
        min_hidden: int = 1,
        n_mc: int = 1000,
        random_state: Optional[int] = None,
        jitter: float = 1e-8,
        solver: str = "SCS",
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        self.alpha = float(alpha)
        self.score_kind = str(score_kind).lower()
        self.calibration_mode = str(calibration_mode).lower()
        self.use_pit = bool(use_pit)
        self.pseudo_hidden_frac = float(pseudo_hidden_frac)
        self.min_hidden = int(min_hidden)
        self.n_mc = int(n_mc)
        self.jitter = float(jitter)
        self.solver = str(solver)
        self.solver_opts = {"verbose": False} if solver_opts is None else dict(solver_opts)
        self.rng = np.random.default_rng(random_state)

        self.mu = _flatten_rows_lastdim(np.asarray(mu, dtype=float))
        self.N, self.J = self.mu.shape

        self.Sigma = None
        if Sigma is not None:
            self.Sigma = _flatten_rows_last2dims(np.asarray(Sigma, dtype=float))
            if self.Sigma.shape[0] != self.N or self.Sigma.shape[1:] != (self.J, self.J):
                raise ValueError("Sigma has incompatible shape.")

        self.L = None
        self.A_diag = None
        self.D_diag = None
        self.r = None

        if L is not None:
            self.L = np.asarray(L, dtype=float)
            if self.L.ndim != 2 or self.L.shape[0] != self.J:
                raise ValueError("L must have shape (J, r).")
            self.r = self.L.shape[1]

        if A_diag is not None:
            self.A_diag = _flatten_rows_lastdim(np.asarray(A_diag, dtype=float))
            if self.A_diag.shape[0] != self.N:
                raise ValueError("A_diag has incompatible row count.")
            if self.r is not None and self.A_diag.shape[1] != self.r:
                raise ValueError("A_diag second dimension must match L.shape[1].")
            self.r = self.A_diag.shape[1]

        if D_diag is not None:
            self.D_diag = _flatten_rows_lastdim(np.asarray(D_diag, dtype=float))
            if self.D_diag.shape != (self.N, self.J):
                raise ValueError("D_diag must have shape (..., J).")

        if self.score_kind not in {"mahalanobis", "hierarchical"}:
            raise ValueError("score_kind must be 'mahalanobis' or 'hierarchical'.")

        if self.calibration_mode not in {"observed", "pseudo_mask"}:
            raise ValueError("calibration_mode must be 'observed' or 'pseudo_mask'.")

        if self.score_kind == "hierarchical":
            if self.L is None or self.A_diag is None or self.D_diag is None:
                raise ValueError(
                    "hierarchical mode requires L, A_diag, and D_diag."
                )

        if self.score_kind == "mahalanobis":
            if self.Sigma is None and (self.L is None or self.A_diag is None or self.D_diag is None):
                raise ValueError(
                    "mahalanobis mode requires either Sigma or (L, A_diag, D_diag)."
                )

        # if self.score_kind == "hierarchical" and self.use_pit:
        #     raise NotImplementedError(
        #         "Hierarchical mode currently uses raw scores only. "
        #         "Set use_pit=False."
        #     )

        self.qhat_: Optional[float] = None
        self.calibration_scores_: Optional[np.ndarray] = None
        self.calibration_records_: Optional[List[ScoreRecord]] = None

    # ------------------------------------------------------------------
    # Row-wise model access
    # ------------------------------------------------------------------

    def _row_A(self, n: int) -> np.ndarray:
        a = np.asarray(self.A_diag[n], dtype=float)
        return np.diag(np.clip(a, self.jitter, None))

    def _row_D_diag(self, n: int) -> np.ndarray:
        d = np.asarray(self.D_diag[n], dtype=float)
        return np.clip(d, self.jitter, None)

    def _row_D(self, n: int) -> np.ndarray:
        return np.diag(self._row_D_diag(n))

    def _row_Sigma(self, n: int) -> np.ndarray:
        if self.Sigma is not None:
            return _symmetrize(self.Sigma[n])
        A = self._row_A(n)
        D = self._row_D(n)
        return _symmetrize(self.L @ A @ self.L.T + D)

    def _subset_cov(self, n: int, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        if self.Sigma is not None:
            S = self._row_Sigma(n)
            return _symmetrize(S[np.ix_(idx, idx)])
        Ls = self.L[idx]
        A = self._row_A(n)
        ds = self._row_D_diag(n)[idx]
        return _symmetrize(Ls @ A @ Ls.T + np.diag(ds))

    def _subset_mean(self, n: int, idx: np.ndarray) -> np.ndarray:
        return self.mu[n, idx]

    # ------------------------------------------------------------------
    # Gaussian conditional law for Mahalanobis mode and pseudo-mask bridge
    # ------------------------------------------------------------------

    def _conditional_gaussian(
        self,
        n: int,
        obs_idx: np.ndarray,
        target_idx: np.ndarray,
        y_obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (conditional_mean, conditional_cov) of target coords given observed coords.
        """
        obs_idx = np.asarray(obs_idx, dtype=int)
        target_idx = np.asarray(target_idx, dtype=int)

        if target_idx.size == 0:
            return np.empty(0, dtype=float), np.empty((0, 0), dtype=float)

        mu = self.mu[n]
        Sigma = self._row_Sigma(n)

        mu_O = mu[obs_idx]
        mu_U = mu[target_idx]
        S_OO = _symmetrize(Sigma[np.ix_(obs_idx, obs_idx)])
        S_UO = Sigma[np.ix_(target_idx, obs_idx)]
        S_OU = Sigma[np.ix_(obs_idx, target_idx)]
        S_UU = _symmetrize(Sigma[np.ix_(target_idx, target_idx)])

        if obs_idx.size == 0:
            return mu_U.copy(), S_UU.copy()

        diff = np.asarray(y_obs, dtype=float) - mu_O
        beta = _solve_spd(S_OO, diff, jitter=self.jitter)
        cond_mu = mu_U + S_UO @ beta

        S_OO_inv_S_OU = _solve_spd(S_OO, S_OU, jitter=self.jitter)
        cond_cov = _symmetrize(S_UU - S_UO @ S_OO_inv_S_OU)

        # Small diagonal stabilization
        cond_cov = cond_cov + self.jitter * np.eye(cond_cov.shape[0])
        return cond_mu, cond_cov

    # ------------------------------------------------------------------
    # Factor posterior
    # ------------------------------------------------------------------

    def _posterior_latent_from_obs(
        self,
        n: int,
        obs_idx: np.ndarray,
        y_obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior of latent factor f given observed coordinates only.

        Returns
        -------
        m_post : shape (r,)
        A_post : shape (r, r)
        """
        obs_idx = np.asarray(obs_idx, dtype=int)
        A = self._row_A(n)
        if obs_idx.size == 0:
            return np.zeros(A.shape[0], dtype=float), A.copy()

        mu_O = self.mu[n, obs_idx]
        r_obs = np.asarray(y_obs, dtype=float) - mu_O

        L_O = self.L[obs_idx]
        d_O = self._row_D_diag(n)[obs_idx]

        A_inv = np.diag(1.0 / np.diag(A))
        D_O_inv = np.diag(1.0 / d_O)

        prec_post = _symmetrize(A_inv + L_O.T @ D_O_inv @ L_O)
        A_post = _safe_inv_spd(prec_post, jitter=self.jitter)
        m_post = A_post @ L_O.T @ D_O_inv @ r_obs
        return m_post, A_post

    # ------------------------------------------------------------------
    # Raw scores
    # ------------------------------------------------------------------

    def _mahal_raw(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        Sigma: np.ndarray,
    ) -> float:
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if y.size == 0:
            return 0.0
        r = y - mu
        z = _solve_spd(Sigma, r, jitter=self.jitter)
        return float(r @ z)

    def _hier_raw_observed_subset(
        self,
        n: int,
        obs_idx: np.ndarray,
        y_obs: np.ndarray,
    ) -> float:
        """
        Raw hierarchical score on the marginal observed subset.

        T = max( ||A^{-1/2} f_hat||_2, ||D_O^{-1/2}(r_O - L_O f_hat)||_inf )
        """
        obs_idx = np.asarray(obs_idx, dtype=int)
        if obs_idx.size == 0:
            return np.nan

        A = self._row_A(n)
        A_inv = np.diag(1.0 / np.diag(A))
        L_O = self.L[obs_idx]
        d_O = self._row_D_diag(n)[obs_idx]
        Sigma_O = self._subset_cov(n, obs_idx)

        r_O = np.asarray(y_obs, dtype=float) - self._subset_mean(n, obs_idx)

        z = _solve_spd(Sigma_O, r_O, jitter=self.jitter)
        f_hat = A @ L_O.T @ z
        eps_hat = r_O - L_O @ f_hat

        latent = math.sqrt(max(float(f_hat @ A_inv @ f_hat), 0.0))
        idio = float(np.max(np.abs(eps_hat) / np.sqrt(d_O)))
        return max(latent, idio)

    def _hier_raw_conditional(
        self,
        n: int,
        obs_idx: np.ndarray,
        y_obs: np.ndarray,
        target_idx: np.ndarray,
        y_target: np.ndarray,
    ) -> float:
        """
        Raw hierarchical score for target coordinates conditional on observed coordinates.

        T = inf_u max(
                ||A_post^{-1/2} u||_2,
                || D_U^{-1/2}( y_U - center_U - L_U u ) ||_inf
            )

        This is solved as an SOCP in CVXPY.
        """
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(
                "Hierarchical conditional scoring requires cvxpy. "
                "Install it with `pip install cvxpy`."
            ) from e

        obs_idx = np.asarray(obs_idx, dtype=int)
        target_idx = np.asarray(target_idx, dtype=int)
        y_target = np.asarray(y_target, dtype=float)

        if target_idx.size == 0:
            return 0.0

        m_post, A_post = self._posterior_latent_from_obs(n, obs_idx, y_obs)
        L_U = self.L[target_idx]
        mu_U = self.mu[n, target_idx]
        dU = np.sqrt(self._row_D_diag(n)[target_idx])

        center_U = mu_U + L_U @ m_post

        # u-score cone: ||F u||_2 <= t, where F.T F = A_post^{-1}
        A_post_inv = _safe_inv_spd(A_post, jitter=self.jitter)
        F = _psd_sqrt_factor_numpy(A_post_inv)

        u = cp.Variable(self.r)
        t = cp.Variable(nonneg=True)

        resid = y_target - center_U - L_U @ u
        constraints = [
            cp.norm(F @ u, 2) <= t,
            resid <= t * dU,
            -resid <= t * dU,
        ]
        prob = cp.Problem(cp.Minimize(t), constraints)
        prob.solve(solver=self.solver, **self.solver_opts)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Hierarchical score solve failed at row {n} with status={prob.status}"
            )
        return float(t.value)
    # ------------------------------------------------------------------
    # PIT / PPF
    # ------------------------------------------------------------------
    def _empirical_pit(self, raw_value: float, ref_scores: np.ndarray) -> float:
        """
        Smoothed empirical CDF:
            \hat G(t) = (1 + #{ref <= t}) / (B + 1)

        Returns a value in (0, 1].
        """
        ref_scores = np.asarray(ref_scores, dtype=float).ravel()
        if ref_scores.size == 0:
            raise ValueError("ref_scores is empty.")
        return float((1.0 + np.sum(ref_scores <= raw_value)) / (ref_scores.size + 1.0))


    def _empirical_ppf(self, pit_level: float, ref_scores: np.ndarray) -> float:
        """
        Invert the empirical CDF to get a raw-score threshold corresponding to
        a PIT-level threshold.

        Uses the 'higher' quantile convention.
        """
        ref_scores = np.sort(np.asarray(ref_scores, dtype=float).ravel())
        if ref_scores.size == 0:
            raise ValueError("ref_scores is empty.")
        u = float(np.clip(pit_level, 0.0, 1.0))
        B = ref_scores.size
        k = int(math.ceil(u * (B + 1))) - 1
        k = min(max(k, 0), B - 1)
        return float(ref_scores[k])


    def _rng_multivariate_normal(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """
        Draw samples from N(mean, cov) with small stabilization.
        Returns shape (size, d).
        """
        mean = np.asarray(mean, dtype=float).reshape(-1)
        cov = _symmetrize(np.asarray(cov, dtype=float))
        d = mean.size
        cov = cov + self.jitter * np.eye(d)
        return self.rng.multivariate_normal(mean=mean, cov=cov, size=size)


    def _hier_ref_scores_observed(
        self,
        n: int,
        obs_idx: np.ndarray,
        B: Optional[int] = None,
    ) -> np.ndarray:
        """
        Monte Carlo reference distribution for the raw hierarchical score on the
        observed subset only:
            T_obs = max( ||A^{-1/2} f_hat||_2,
                        ||D_O^{-1/2}(r_O - L_O f_hat)||_inf )

        Simulate Y_O ~ N(mu_O, Sigma_O).
        """
        obs_idx = np.asarray(obs_idx, dtype=int)
        if obs_idx.size == 0:
            return np.array([0.0], dtype=float)

        if B is None:
            B = self.n_mc

        mu_O = self._subset_mean(n, obs_idx)
        Sigma_O = self._subset_cov(n, obs_idx)

        sims = self._rng_multivariate_normal(mu_O, Sigma_O, size=B)
        out = np.empty(B, dtype=float)
        for b in range(B):
            out[b] = self._hier_raw_observed_subset(n, obs_idx, sims[b])
        return out


    def _hier_ref_scores_conditional(
        self,
        n: int,
        obs_idx: np.ndarray,
        y_obs: np.ndarray,
        target_idx: np.ndarray,
        B: Optional[int] = None,
    ) -> np.ndarray:
        """
        Monte Carlo reference distribution for the raw hierarchical *conditional*
        score on target coordinates given observed coordinates:
            T_cond(y_U) = inf_u max(
                ||A_post^{-1/2} u||_2,
                ||D_U^{-1/2}(y_U - center_U - L_U u)||_inf
            )

        Simulate Y_U | Y_O=y_obs from the fitted conditional Gaussian law.
        """
        obs_idx = np.asarray(obs_idx, dtype=int)
        target_idx = np.asarray(target_idx, dtype=int)

        if target_idx.size == 0:
            return np.array([0.0], dtype=float)

        if B is None:
            B = self.n_mc

        cond_mu, cond_cov = self._conditional_gaussian(
            n=n,
            obs_idx=obs_idx,
            target_idx=target_idx,
            y_obs=y_obs,
        )

        sims = self._rng_multivariate_normal(cond_mu, cond_cov, size=B)
        out = np.empty(B, dtype=float)
        for b in range(B):
            out[b] = self._hier_raw_conditional(
                n=n,
                obs_idx=obs_idx,
                y_obs=y_obs,
                target_idx=target_idx,
                y_target=sims[b],
            )
        return out
    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def _draw_pseudo_split(self, obs_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split observed indices into revealed R and pseudo-hidden H.
        """
        obs_idx = np.asarray(obs_idx, dtype=int)
        m = obs_idx.size
        if m < 2:
            raise ValueError("Need at least 2 observed coordinates for pseudo-mask calibration.")

        h = max(self.min_hidden, int(round(self.pseudo_hidden_frac * m)))
        h = min(h, m - 1)  # leave at least 1 revealed coordinate

        perm = self.rng.permutation(obs_idx)
        hidden = np.sort(perm[:h])
        reveal = np.sort(perm[h:])
        return reveal, hidden

    def _score_for_row(
        self,
        n: int,
        y_row: np.ndarray,
        obs_mask: np.ndarray,
    ) -> ScoreRecord:
        obs_idx = np.flatnonzero(obs_mask)
        if obs_idx.size == 0:
            return ScoreRecord(
                row=n,
                score=np.nan,
                raw_score=np.nan,
                observed_idx=np.array([], dtype=int),
                target_idx=np.array([], dtype=int),
            )

        if self.calibration_mode == "observed":
            target_idx = np.array([], dtype=int)

            if self.score_kind == "mahalanobis":
                mu_O = self._subset_mean(n, obs_idx)
                Sigma_O = self._subset_cov(n, obs_idx)
                raw = self._mahal_raw(y_row[obs_idx], mu_O, Sigma_O)
                score = chi2.cdf(raw, df=obs_idx.size) if self.use_pit else raw

            elif self.score_kind == "hierarchical":
                raw = self._hier_raw_observed_subset(n, obs_idx, y_row[obs_idx])
                if self.use_pit:
                    ref = self._hier_ref_scores_observed(n, obs_idx, B=self.n_mc)
                    score = self._empirical_pit(raw, ref)
                else:
                    score = raw

            else:
                raise ValueError("Unknown score_kind.")

            return ScoreRecord(
                row=n,
                score=float(score),
                raw_score=float(raw),
                observed_idx=obs_idx,
                target_idx=target_idx,
            )

        elif self.calibration_mode == "pseudo_mask":
            reveal_idx, hidden_idx = self._draw_pseudo_split(obs_idx)
            y_reveal = y_row[reveal_idx]
            y_hidden = y_row[hidden_idx]

            if self.score_kind == "mahalanobis":
                cond_mu, cond_cov = self._conditional_gaussian(
                    n=n,
                    obs_idx=reveal_idx,
                    target_idx=hidden_idx,
                    y_obs=y_reveal,
                )
                raw = self._mahal_raw(y_hidden, cond_mu, cond_cov)
                score = chi2.cdf(raw, df=hidden_idx.size) if self.use_pit else raw

            elif self.score_kind == "hierarchical":
                raw = self._hier_raw_conditional(
                    n=n,
                    obs_idx=reveal_idx,
                    y_obs=y_reveal,
                    target_idx=hidden_idx,
                    y_target=y_hidden,
                )
                if self.use_pit:
                    ref = self._hier_ref_scores_conditional(
                        n=n,
                        obs_idx=reveal_idx,
                        y_obs=y_reveal,
                        target_idx=hidden_idx,
                        B=self.n_mc,
                    )
                    score = self._empirical_pit(raw, ref)
                else:
                    score = raw

            else:
                raise ValueError("Unknown score_kind.")

            return ScoreRecord(
                row=n,
                score=float(score),
                raw_score=float(raw),
                observed_idx=reveal_idx,
                target_idx=hidden_idx,
            )

        else:
            raise ValueError("Unknown calibration_mode.")

    # ------------------------------------------------------------------
    # Public calibration API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        Y: np.ndarray,
        M: np.ndarray,
        calibration_rows: Optional[Sequence[int]] = None,
    ) -> float:
        """
        Calibrate qhat from partially observed rows.

        Parameters
        ----------
        Y : array, shape (..., J)
            Outcomes / residual rows.
        M : array, shape (..., J), bool
            Observation mask, True = observed.
        calibration_rows : sequence of row indices or None
            If None, use all rows.
        """
        Y = _flatten_rows_lastdim(np.asarray(Y, dtype=float))
        M = _flatten_rows_lastdim(np.asarray(M, dtype=bool))

        if Y.shape != (self.N, self.J):
            raise ValueError("Y shape incompatible with mu.")
        if M.shape != (self.N, self.J):
            raise ValueError("M shape incompatible with mu.")

        rows = np.arange(self.N) if calibration_rows is None else np.asarray(calibration_rows, dtype=int)

        records: List[ScoreRecord] = []
        scores: List[float] = []

        for n in rows:
            try:
                rec = self._score_for_row(n, Y[n], M[n])
            except ValueError:
                # e.g. pseudo-mask row with <2 observed coords; skip
                continue

            if np.isfinite(rec.score):
                records.append(rec)
                scores.append(rec.score)

        if len(scores) == 0:
            raise ValueError("No finite calibration scores were produced.")

        self.qhat_ = _higher_quantile(np.asarray(scores, dtype=float), self.alpha)
        self.calibration_scores_ = np.asarray(scores, dtype=float)
        self.calibration_records_ = records
        return self.qhat_

    # ------------------------------------------------------------------
    # Candidate scoring at test time
    # ------------------------------------------------------------------

    def score_candidate(
        self,
        n: int,
        y_row: np.ndarray,
        obs_mask: np.ndarray,
    ) -> float:
        """
        Score a full row in 'observed' mode, or a row with target coordinates in
        'pseudo_mask' mode if you provide a mask where False entries are treated
        as targets and True entries as revealed.

        In deployment for missing-coordinate prediction, use predict_row() instead.
        """
        y_row = np.asarray(y_row, dtype=float).reshape(-1)
        obs_mask = np.asarray(obs_mask, dtype=bool).reshape(-1)
        rec = self._score_for_row(n, y_row, obs_mask)
        return rec.score

    # ------------------------------------------------------------------
    # Prediction for missing coordinates
    # ------------------------------------------------------------------

    def _require_calibrated(self) -> None:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate(...) first.")

    def predict_row(
        self,
        n: int,
        y_obs_row: np.ndarray,
        obs_mask: np.ndarray,
        *,
        target_idx: Optional[Sequence[int]] = None,
        exact_hierarchical: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a model-based/asymptotic prediction object for the missing coordinates.

        Parameters
        ----------
        n : int
            Row index.
        y_obs_row : array, shape (J,)
            Row with observed values in the observed entries. Unobserved entries
            can contain anything; they are ignored.
        obs_mask : array, shape (J,), bool
            True = observed.
        target_idx : sequence or None
            If None, targets are the missing coordinates (~obs_mask).
        exact_hierarchical : bool
            If True and score_kind='hierarchical', compute exact coordinate
            projections using CVXPY. If False, return conservative intervals.

        Returns
        -------
        dict with keys depending on score_kind
        """
        self._require_calibrated()

        y_obs_row = np.asarray(y_obs_row, dtype=float).reshape(-1)
        obs_mask = np.asarray(obs_mask, dtype=bool).reshape(-1)

        obs_idx = np.flatnonzero(obs_mask)
        if target_idx is None:
            tgt_idx = np.flatnonzero(~obs_mask)
        else:
            tgt_idx = np.asarray(target_idx, dtype=int)

        if tgt_idx.size == 0:
            return {
                "row": n,
                "target_idx": tgt_idx,
                "center": np.empty(0, dtype=float),
                "lower": np.empty(0, dtype=float),
                "upper": np.empty(0, dtype=float),
                "qhat": self.qhat_,
            }

        if self.score_kind == "mahalanobis":
            cond_mu, cond_cov = self._conditional_gaussian(
                n=n,
                obs_idx=obs_idx,
                target_idx=tgt_idx,
                y_obs=y_obs_row[obs_idx],
            )
            raw_thr = chi2.ppf(self.qhat_, df=tgt_idx.size) if self.use_pit else self.qhat_
            raw_thr = max(float(raw_thr), 0.0)

            # Exact coordinate projection of ellipsoid:
            # [mu_j ± sqrt(raw_thr * Sigma_jj)]
            sd = np.sqrt(np.clip(np.diag(cond_cov), self.jitter, None))
            half = np.sqrt(raw_thr) * sd
            lower = cond_mu - half
            upper = cond_mu + half

            return {
                "row": n,
                "target_idx": tgt_idx,
                "center": cond_mu,
                "cov": cond_cov,
                "raw_threshold": raw_thr,
                "lower": lower,
                "upper": upper,
                "qhat": self.qhat_,
            }

        if self.score_kind == "hierarchical":
            m_post, A_post = self._posterior_latent_from_obs(
                n=n,
                obs_idx=obs_idx,
                y_obs=y_obs_row[obs_idx],
            )

            L_U = self.L[tgt_idx]
            mu_U = self.mu[n, tgt_idx]
            dU = np.sqrt(self._row_D_diag(n)[tgt_idx])

            center = mu_U + L_U @ m_post
            if self.use_pit:
                ref = self._hier_ref_scores_conditional(
                    n=n,
                    obs_idx=obs_idx,
                    y_obs=y_obs_row[obs_idx],
                    target_idx=tgt_idx,
                    B=self.n_mc,
                )
                raw_thr = self._empirical_ppf(self.qhat_, ref)
            else:
                raw_thr = float(self.qhat_)

            if exact_hierarchical:
                lower, upper = self._hierarchical_exact_project(
                    n=n,
                    center=center,
                    A_post=A_post,
                    L_U=L_U,
                    dU=dU,
                    raw_thr=raw_thr,
                )
            else:
                # Conservative outer bound
                latent_sd = np.sqrt(np.clip(np.sum((L_U @ A_post) * L_U, axis=1), self.jitter, None))
                half = raw_thr * (latent_sd + dU)
                lower = center - half
                upper = center + half

            return {
                "row": n,
                "target_idx": tgt_idx,
                "center": center,
                "A_post": A_post,
                "raw_threshold": raw_thr,
                "lower": lower,
                "upper": upper,
                "qhat": self.qhat_,
            }

        raise ValueError("Unknown score_kind.")

    def _hierarchical_exact_project(
        self,
        n: int,
        center: np.ndarray,
        A_post: np.ndarray,
        L_U: np.ndarray,
        dU: np.ndarray,
        raw_thr: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact coordinate projections for the hierarchical conditional region:

            exists u s.t.
                || A_post^{-1/2} u ||_2 <= raw_thr
                | y_U - center - L_U u | <= raw_thr * dU

        One SOCP solve per target coordinate endpoint.
        """
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError(
                "Exact hierarchical projections require cvxpy. "
                "Install it with `pip install cvxpy`."
            ) from e

        center = np.asarray(center, dtype=float)
        L_U = np.asarray(L_U, dtype=float)
        dU = np.asarray(dU, dtype=float)

        p = center.size
        r = A_post.shape[0]

        A_post_inv = _safe_inv_spd(A_post, jitter=self.jitter)
        F = _psd_sqrt_factor_numpy(A_post_inv)

        lower = np.full(p, np.nan, dtype=float)
        upper = np.full(p, np.nan, dtype=float)

        for j in range(p):
            y = cp.Variable(p)
            u = cp.Variable(r)

            resid = y - center - L_U @ u
            constraints = [
                cp.norm(F @ u, 2) <= raw_thr,
                resid <= raw_thr * dU,
                -resid <= raw_thr * dU,
            ]

            prob_lo = cp.Problem(cp.Minimize(y[j]), constraints)
            prob_lo.solve(solver=self.solver, **self.solver_opts)
            if prob_lo.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(
                    f"Hierarchical exact lower projection failed at row {n}, "
                    f"target position {j}, status={prob_lo.status}"
                )
            lower[j] = float(y.value[j])

            prob_hi = cp.Problem(cp.Maximize(y[j]), constraints)
            prob_hi.solve(solver=self.solver, **self.solver_opts)
            if prob_hi.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(
                    f"Hierarchical exact upper projection failed at row {n}, "
                    f"target position {j}, status={prob_hi.status}"
                )
            upper[j] = float(y.value[j])

        return lower, upper