"""
Numerical equivalence check: dense factor E-step vs Woodbury + pattern caching.

Run:
  python scripts/test_woodbury_equivalence.py
"""

from __future__ import annotations

import os
import sys
import math

import torch

# Ensure project root is importable when running as a script from ./scripts
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from twfv.factor import factor_E_step, build_mask_pattern_groups
from twfv.woodbury import factor_E_step_woodbury


def _make_mask(I: int, T: int, J: int, p_obs: float, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.rand((I, T, J), generator=g) < p_obs
    # Ensure at least one observed channel per (i,t) to avoid trivial all-missing rows.
    for i in range(I):
        for t in range(T):
            if not bool(M[i, t].any()):
                M[i, t, torch.randint(0, J, (1,), generator=g).item()] = True
    return M


def main():
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    # Small-ish random problem
    I, T, J, r = 4, 7, 9, 3

    Y = torch.randn(I, T, J, dtype=dtype, device=device)
    M_mask = _make_mask(I, T, J, p_obs=0.7, seed=1).to(device)

    L = torch.randn(J, r, dtype=dtype, device=device) * 0.3
    psi = torch.rand(J, dtype=dtype, device=device) * 0.5 + 0.1  # positive
    s = torch.rand(I, dtype=dtype, device=device) * 0.5 + 0.8  # positive
    kappa = torch.rand(I, dtype=dtype, device=device) * 0.7 + 0.2  # positive
    a2 = torch.rand(I, T, r, dtype=dtype, device=device) * 0.7 + 0.2  # positive

    # Dense baseline
    F_dense, ll_dense = factor_E_step(Y, M_mask, L, psi, s, kappa, a2)

    # Fast Woodbury + pattern caching (set jitter=0 for strict equivalence)
    pattern_groups = build_mask_pattern_groups(M_mask)
    F_fast, ll_fast = factor_E_step_woodbury(
        Y, M_mask, L, psi, s, kappa, a2, pattern_groups=pattern_groups, jitter=0.0
    )

    max_abs_F = (F_dense - F_fast).abs().max().item()
    rel_ll = abs((ll_dense - ll_fast).item()) / max(1.0, abs(ll_dense.item()))

    print("Woodbury equivalence check")
    print(f"max|F_dense - F_fast| = {max_abs_F:.3e}")
    print(f"ll_dense = {ll_dense.item():.12e}")
    print(f"ll_fast  = {ll_fast.item():.12e}")
    print(f"relative ll diff       = {rel_ll:.3e}")

    # Tolerances: should be tight in float64 when jitter=0.
    # F_mean may differ at ~1e-7 due to floating point ordering differences.
    if not (max_abs_F < 1e-7 and rel_ll < 1e-10 and math.isfinite(ll_fast.item())):
        raise SystemExit("FAILED: Woodbury and dense computations differ beyond tolerance.")

    print("PASSED")


if __name__ == "__main__":
    main()


