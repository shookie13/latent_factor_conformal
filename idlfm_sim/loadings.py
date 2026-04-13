import numpy as np
from .config import SimConfig
from .utils import rng


def simulate_loadings(cfg: SimConfig) -> np.ndarray:
    """Draw stream loadings a_j ~ N(0, I_R), shape (J, R)."""
    r = rng(cfg.seed + 7)
    return r.normal(size=(cfg.J, cfg.R))


