import numpy as np
from .config import SimConfig


def hetero_v(cfg: SimConfig, i: np.ndarray, j: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Heteroskedasticity modulator v_{ij}(t) > 0 for provided indices (broadcastable arrays).
    Modes:
    - spikes: 1 + c_spike * 1{ (t mod Ps) in [ell, u] }
    - seasonal: 1 + c_seasonal * sin^2(2π t / Psea)
    """
    if cfg.hetero_mode == "spikes":
        ell, u = cfg.spike_window
        in_window = ((t % cfg.Ps) >= ell) & ((t % cfg.Ps) <= u)
        return 1.0 + cfg.c_spike * in_window.astype(float)
    elif cfg.hetero_mode == "seasonal":
        return 1.0 + cfg.c_seasonal * (np.sin(2.0 * np.pi * t / cfg.Psea) ** 2)
    else:
        return np.ones_like(t, dtype=float)


def hetero_v_grid(cfg: SimConfig) -> np.ndarray:
    i = np.arange(cfg.I)[:, None, None]
    j = np.arange(cfg.J)[None, :, None]
    t = np.arange(cfg.T)[None, None, :]
    return hetero_v(cfg, i, j, t)


