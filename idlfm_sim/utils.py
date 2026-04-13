from typing import Iterable
import numpy as np


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_len(arr_like: Iterable[int], J: int) -> np.ndarray:
    arr = np.array(list(arr_like), dtype=int)
    if arr.size == J:
        return arr
    if arr.size == 0:
        return np.ones(J, dtype=int)
    reps = int(np.ceil(J / arr.size))
    return np.tile(arr, reps)[:J]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p) - np.log1p(-p)


