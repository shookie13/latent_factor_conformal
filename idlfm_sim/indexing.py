from typing import Tuple


def flatten(i: int, j: int, t: int, I: int, J: int, T: int) -> int:
    """Row-major flatten: k = ((i * J) + j) * T + t (0-based)."""
    return ((i * J) + j) * T + t


def unflatten(k: int, I: int, J: int, T: int) -> Tuple[int, int, int]:
    """Inverse of flatten for 0-based indices."""
    ij, t = divmod(k, T)
    i, j = divmod(ij, J)
    return i, j, t


