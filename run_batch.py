from __future__ import annotations

import argparse
import numpy as np
import torch
from twfv import run_em_like
from twfv.simulate import simulate_factor_data_bspline
from twfv.cp import unflatten
from twfv.bspline import make_open_uniform_knots
from twfv.variance import build_a2_from_C_and_warp


def flatten(i: int, j: int, t: int, I: int, J: int, T: int) -> int:
    return i * (J * T) + j * T + t


def make_train_mask_from_flat(train_idx: np.ndarray, I: int, T: int, J: int) -> np.ndarray:
    """
    train_idx uses flatten convention k = i*(J*T) + j*T + t.
    Return M_train bool mask in shape (I,T,J).
    """
    flat_mask = np.zeros(I * J * T, dtype=bool)
    flat_mask[train_idx] = True
    return flat_mask.reshape(I, J, T).transpose(0, 2, 1)


def local_var_from_fit(L: torch.Tensor, psi: torch.Tensor, s: torch.Tensor, C: torch.Tensor, u_tilde: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Compute fitted channel-wise innovation variance Var(y_{i,t,j}) under the fitted model:
      Var(y_{i,t,j}) = sum_k L_{j,k}^2 * a2_{i,t,k} + s_i^2 * psi_j
    """
    with torch.no_grad():
        _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)  # (I,T,r)
        base = torch.einsum("itr,jr->itj", a2, L.pow(2))
        var = base + (s**2)[:, None, None] * psi[None, None, :]
    return var
def conformal_scaled_abs(Y_cal_true, Y_cal_fit, sigma_cal, Y_test_fit, sigma_test, alpha=0.1, sigma_min=1e-6):
    sigma_cal = np.maximum(sigma_cal, sigma_min)
    sigma_test = np.maximum(sigma_test, sigma_min)

    S_cal = np.abs(Y_cal_true - Y_cal_fit) / sigma_cal

    m = S_cal.size
    q_level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
    try:
        q = np.quantile(S_cal, q_level, method="higher")
    except TypeError:
        q = np.quantile(S_cal, q_level, interpolation="higher")

    lower = Y_test_fit - q * sigma_test
    upper = Y_test_fit + q * sigma_test
    return lower, upper, q


def run_one(main_seed: int, *, I: int, T: int, J: int, r: int, M_ctrl: int, degree: int, max_outer: int, grad_steps: int, lr: float, alpha: float, use_variance_tied_missing: bool) -> dict:
    """
    One full run (simulate -> split -> fit on train mask -> calibrate on cal -> evaluate on test).
    main_seed deterministically controls all randomness.

    Returns a dict with coverage and metadata.
    """
    rng = np.random.default_rng(int(main_seed))
    seed_data = int(rng.integers(0, 2**31 - 1))
    seed_split = int(rng.integers(0, 2**31 - 1))
    seed_model = int(rng.integers(0, 2**31 - 1))

    # deterministic parameter draws for simulation
    kappa_true = np.exp(rng.normal(2.0, 0.5, size=I))
    C_true = rng.normal(scale=3.0, size=(I, r, M_ctrl))
    miss_prob_per_channel = np.linspace(0.1, 0.3, J)

    Y, M, _info = simulate_factor_data_bspline(
        I,
        T,
        J,
        r,
        M_ctrl=M_ctrl,
        degree=degree,
        seed=seed_data,
        use_variance_tied_missing=use_variance_tied_missing,
        miss_prob_per_channel=miss_prob_per_channel,
        kappa_true=kappa_true,
        C_true=C_true,
    )

    # Build flattened indices for observed/test from M
    obs_idx = np.flatnonzero(np.transpose(M, (0, 2, 1)).reshape(-1))  # M is (I,T,J) -> (I,J,T) -> flat
    test_idx = np.flatnonzero(~np.transpose(M, (0, 2, 1)).reshape(-1))

    rng_split = np.random.default_rng(seed_split)
    rng_split.shuffle(obs_idx)
    split = int(0.7 * obs_idx.size)
    train_idx = obs_idx[:split]
    cal_idx = obs_idx[split:]

    # Extract y on cal and test; mean predictor is 0 (innovations)
    Y_flat = np.transpose(Y, (0, 2, 1)).reshape(-1)
    Y_cal_true = Y_flat[cal_idx]
    Y_test_true = Y_flat[test_idx]
    Y_cal_fit = np.zeros_like(Y_cal_true)
    Y_test_fit = np.zeros_like(Y_test_true)

    # Fit model on train mask only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed_model)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_model)

    Y_torch = torch.as_tensor(Y, dtype=torch.float32, device=device)
    M_train = make_train_mask_from_flat(train_idx, I, T, J)
    M_train_torch = torch.as_tensor(M_train, dtype=torch.bool, device=device)

    em_result = run_em_like(
        Y=Y_torch,
        M_mask=M_train_torch,
        r=r,
        M_ctrl=int(M_ctrl),
        degree=degree,
        max_outer=max_outer,
        grad_steps=grad_steps,
        lr=lr,
        device=device,
    )

    # run_em_like has evolved over time; support both tuple layouts:
    #   (L, psi, s, C, u_tilde, F_tilde, history)
    # and the newer layout that includes a2 explicitly:
    #   (L, a2, psi, s, C, u_tilde, F_tilde, history)
    if isinstance(em_result, tuple) or isinstance(em_result, list):
        if len(em_result) == 8:
            L_hat, _a2_hat, psi_hat, s_hat, C_hat, u_tilde_hat, _F_hat, _hist = em_result
        elif len(em_result) == 7:
            L_hat, psi_hat, s_hat, C_hat, u_tilde_hat, _F_hat, _hist = em_result
        else:
            raise RuntimeError(f"Unexpected run_em_like return length: {len(em_result)}")
    else:
        raise RuntimeError(f"Unexpected run_em_like return type: {type(em_result)}")

    knots = make_open_uniform_knots(M_ctrl, degree, device=device)
    var_fitted = local_var_from_fit(L_hat, psi_hat, s_hat, C_hat, u_tilde_hat, knots, degree).cpu().numpy()  # (I,T,J)
    sigma_fitted = np.sqrt(np.maximum(var_fitted, 1e-12))
    sigma_flat = np.transpose(sigma_fitted, (0, 2, 1)).reshape(-1)  # (I*J*T,)

    sigma_cal = sigma_flat[cal_idx]
    sigma_test = sigma_flat[test_idx]

    lower, upper, q = conformal_scaled_abs(
        Y_cal_true, Y_cal_fit, sigma_cal, Y_test_fit, sigma_test, alpha=alpha, sigma_min=1e-6
    )
    covered = (Y_test_true >= lower) & (Y_test_true <= upper)
    cov = float(np.mean(covered)) if covered.size > 0 else float("nan")

    return {
        "main_seed": int(main_seed),
        "seed_data": seed_data,
        "seed_split": seed_split,
        "seed_model": seed_model,
        "coverage": cov,
        "q": float(q),
        "n_cal": int(cal_idx.size),
        "n_test": int(test_idx.size),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--I", type=int, default=10)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--J", type=int, default=5)
    p.add_argument("--r", type=int, default=2)
    p.add_argument("--M-ctrl", type=int, default=10)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--max-outer", type=int, default=100)
    p.add_argument("--grad-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--variance-tied-missing", action="store_true")
    p.add_argument("--out-csv", type=str, default="batch_results.csv")
    args = p.parse_args()

    results = []
    for ms in args.main_seeds:
        res = run_one(
            ms,
            I=args.I,
            T=args.T,
            J=args.J,
            r=args.r,
            M_ctrl=args.M_ctrl,
            degree=args.degree,
            max_outer=args.max_outer,
            grad_steps=args.grad_steps,
            lr=args.lr,
            alpha=args.alpha,
            use_variance_tied_missing=bool(args.variance_tied_missing),
        )
        results.append(res)
        print(f"main_seed={ms} coverage={res['coverage']:.3f} q={res['q']:.3f} n_test={res['n_test']}")

    # Save CSV
    import csv

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    covs = np.array([r["coverage"] for r in results], dtype=float)
    print(f"coverage mean={np.nanmean(covs):.3f} std={np.nanstd(covs):.3f}")

if __name__ == "__main__":
    main()