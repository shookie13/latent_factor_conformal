# TWFV Conformal

Subject-specific time-warped B-spline factor variance model implemented in PyTorch. The package follows the provided specification: warped time built from factor posterior volatility, B-spline log-variance per subject-factor, and alternating E/M steps.

## Layout

- `twfv/bspline.py`: B-spline knot construction and basis evaluation.
- `twfv/warp.py`: volatility-based time warping utilities.
- `twfv/variance.py`: log-variance and factor variance construction.
- `twfv/factor.py`: posterior factor means and log-likelihood.
- `twfv/runner.py`: EM-like outer loop glue.

## Quick start

```python
import torch
from twfv.runner import run_em_like

# Fake data: I subjects, T time points, J channels
I, T, J, r, M_ctrl = 4, 20, 6, 2, 6
Y = torch.randn(I, T, J)
M_mask = torch.ones_like(Y, dtype=torch.bool)

L, a2, psi, s, kappa, C, u_tilde, F_tilde, history = run_em_like(
    Y, M_mask, r=r, M_ctrl=M_ctrl, degree=3,
    max_outer=5, grad_steps=2, lr=1e-2, device="cpu"
)
```

All tensors are compatible with autograd, so you can swap the optimizer or integrate into a larger training loop.

