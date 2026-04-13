from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import numpy as np


@dataclass
class SimConfig:
    I: int = 100
    J: int = 5
    T: int = 400
    R: int = 2  # factor dimensionality
    M: int = 3  # number of temporal basis functions
    Rk: int = 2  # factor rank used in MAR dependency
    sigma_u: float = 0.7
    sigma_theta: float = 0.5
    sigma_Y: float = 0.0
    kappa: Optional[np.ndarray] = None  # shape (J,)
    hetero_mode: str = "spikes"  # "spikes" | "seasonal"
    Ps: int = 80
    spike_window: Tuple[int, int] = (0, 20)
    c_spike: float = 0.7
    Psea: int = 50
    c_seasonal: float = 0.0
    Pj: Iterable[int] = (1,1,1,1,1)#(1, 1, 2, 5, 5, 10, 10, 10)
    rhoj: float = 0.1
    mar_mode: str = "linear"  # "linear" | "seasonal" | "duration"
    gamma0: Optional[np.ndarray] = None  # shape (J,)
    gamma: Optional[np.ndarray] = None  # shape (J, Rk)
    gamma_s: float = 0.0
    delta0: float = 0.0
    delta1: float = 0.5
    delta2: float = 0.2
    seed: int = 123
    # Parameters for the latent-driven noise projection b_j ~ N(loc, scale^2)
    noise_beta_loc: float = 0.0
    noise_beta_scale: float = 0.0
    p_h: float = 0.7
    v_h: float = 1.0
    T_c: int = 80
    bandwidth: float = 10.0


