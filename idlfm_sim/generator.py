import numpy as np
from typing import Tuple
from .config import SimConfig
from .hetero import hetero_v_grid
from .utils import rng


def simulate_Y_full(cfg: SimConfig, theta: np.ndarray, Aload: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Y_full on full grid (I, J, T).
    Returns: Y_full, mu (J,), sigmaYj (J,)
    Y_{ij}(t) = mu_j + a_j^T theta_i(t) + eps, eps ~ N(0, [sigma_Yj * sqrt(v_{ij}(t)) + |b_j^T theta_i(t)|]^2 ).
    """
    assert theta.shape == (cfg.I, cfg.T, cfg.R)
    assert Aload.shape == (cfg.J, cfg.R)
    r = rng(cfg.seed + 13)
    mu = np.zeros(cfg.J, dtype=float)
    if cfg.kappa is None:
        kappa = np.ones(cfg.J, dtype=float)
    else:
        kappa = np.asarray(cfg.kappa, dtype=float)
        if kappa.shape != (cfg.J,):
            raise ValueError("kappa must have shape (J,)")
    sigmaYj = cfg.sigma_Y * np.sqrt(kappa)
    mean_term = np.einsum("itr,jr->ijt", theta, Aload) + mu[None, :, None]
    v = hetero_v_grid(cfg)
    # Base heteroskedastic term
    noise_sd = sigmaYj[None, :, None] * np.sqrt(v)
    # Add absolute value of a linear transform of theta over R-dim (per stream j)
    # b_j ~ N(cfg.noise_beta_loc, cfg.noise_beta_scale^2 I_R)
    B_noise = r.normal(loc=cfg.noise_beta_loc, scale=cfg.noise_beta_scale, size=(cfg.J, cfg.R))
    proj = np.abs(np.einsum("itr,jr->ijt", theta, B_noise))  # (I,J,T)
    print(proj.mean(),proj.std())
    print(noise_sd.mean(),noise_sd.std())
    noise_sd = noise_sd + proj
    eps = r.normal(loc=0.0, scale=0.1, size=(cfg.I, cfg.J, cfg.T)) * noise_sd
    Y_full = mean_term + eps
    return Y_full, mu, sigmaYj, eps, noise_sd


