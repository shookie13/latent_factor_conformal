import numpy as np
from typing import Tuple
from .config import SimConfig
from .utils import rng, ensure_len


def build_multi_resolution_masks(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns sched_mask (I,J,T) and O (I,J,T) per §4.
    """
    r = rng(cfg.seed + 23)
    Pj = ensure_len(cfg.Pj, cfg.J)
    rhoj = np.full(cfg.J, cfg.rhoj, dtype=float)
    U = np.zeros((cfg.I, cfg.J), dtype=int)
    for j in range(cfg.J):
        if Pj[j] <= 0:
            raise ValueError("Pj must be positive")
        U[:, j] = r.integers(low=0, high=Pj[j], size=cfg.I, endpoint=False)
    t = np.arange(cfg.T)[None, None, :]
    sched_mask = np.zeros((cfg.I, cfg.J, cfg.T), dtype=np.int8)
    for j in range(cfg.J):
        sj = (((t + U[:, j][:, None, None]) % Pj[j]) == 0).astype(np.int8)
        sched_mask[:, j, :] = sj[:, 0, :]
    B = r.random(size=(cfg.I, cfg.J, cfg.T))
    O = ((B < (1.0 - rhoj[None, :, None])) & (sched_mask == 1)).astype(np.int8)
    return sched_mask, O


