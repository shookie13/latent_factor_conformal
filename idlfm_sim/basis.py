import numpy as np


def sample_basis(T: int, M: int) -> np.ndarray:
    """
    Return temporal basis B with shape (T, M).
    Defaults: sin(2π t / {40, 60, 90}) truncated/padded to M.
    """
    t = np.arange(T, dtype=float)
    periods = np.array([40, 60, 90], dtype=float)
    base = [np.sin(2.0 * np.pi * t / p) for p in periods]
    B = np.stack(base, axis=1)
    if M <= B.shape[1]:
        return B[:, :M]
    reps = [50.0, 75.0, 110.0, 140.0]
    extra = [np.sin(2.0 * np.pi * t / reps[idx % len(reps)]) for idx in range(M - B.shape[1])]
    if extra:
        B = np.concatenate([B, np.stack(extra, axis=1)], axis=1)
    return B


