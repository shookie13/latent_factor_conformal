import numpy as np
from typing import Tuple


def train_cal_split(cfg, O: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    I, J, T = O.shape
    observed = (O * A) == 1
    t = np.arange(T)[None, None, :]
    is_train = observed & ((t % 2) == 1)
    is_cal = observed & ((t % 2) == 0)
    return is_train, is_cal


def miss_pool(O: np.ndarray, A: np.ndarray) -> np.ndarray:
    return (O * A) != 1


