import numpy as np
from .config import SimConfig
from .basis import sample_basis
from .utils import rng


def simulate_theta(cfg: SimConfig) -> np.ndarray:
    """
    Simulate latent factors theta with shape (I, T, R).
    theta_i(t) = sum_m b_m(t) u_{im} + eps_{it}
    """
    r = rng(cfg.seed)
    B = sample_basis(cfg.T, cfg.M)  # (T, M)
    u = r.normal(loc=0.0, scale=cfg.sigma_u, size=(cfg.I, cfg.M, cfg.R))
    eps = r.normal(loc=0.0, scale=cfg.sigma_theta, size=(cfg.I, cfg.T, cfg.R))
    # Computes: theta_{i t r} = sum_m B_{t m} u_{i m r} + eps_{i t r}
    theta = np.einsum("tm,imr->itr", B, u) + eps
    return theta


