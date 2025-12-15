import torch


def build_time_warp_from_proxy(vol_proxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Build monotone time warp from a volatility proxy.

    Args:
        vol_proxy: tensor (I, T, r) with nonnegative entries.
        eps: small positive constant to keep weights > 0.

    Returns:
        u_tilde: tensor (I, T, r) in (0, 1], strictly increasing over t.
    """
    w = vol_proxy + eps
    S = w.cumsum(dim=1)
    total = S[:, -1:, :]
    u_tilde = S / total.clamp_min(1e-12)
    return u_tilde


def update_time_warp(F_tilde: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Update warp using squared posterior factors as the proxy.

    Args:
        F_tilde: tensor (I, T, r).
        eps: positive floor for weights.

    Returns:
        u_tilde: tensor (I, T, r).
    """
    vol = F_tilde.pow(2)
    return build_time_warp_from_proxy(vol, eps=eps)

