from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from twfv import run_em_like
from twfv.simulate import simulate_factor_data_bspline
from twfv.cp import NaiveAbsoluteResidualCP, unflatten
from twfv.metrics import conditional_interval_report
from twfv.bspline import make_open_uniform_knots
from twfv.variance import build_a2_from_C_and_warp
# Save CSV (append if exists)
import csv
import os

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


def local_var_from_fit(
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    kappa: torch.Tensor,
    C: torch.Tensor,
    u_tilde: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Compute fitted channel-wise innovation variance Var(y_{i,t,j}) under the fitted model:
      Var(y_{i,t,j}) = kappa_i * sum_k L_{j,k}^2 * a2_{i,t,k} + s_i^2 * psi_j
    """
    with torch.no_grad():
        _, a2 = build_a2_from_C_and_warp(C, knots, degree, u_tilde)  # (I,T,r)
        base = torch.einsum("itr,jr->itj", a2, L.pow(2))
        var = kappa[:, None, None] * base + (s**2)[:, None, None] * psi[None, None, :]
    return var


def local_var_Y_from_info(info: dict) -> np.ndarray:
    """
    Oracle Var(Y_{i,t,j}) implied by `simulate_factor_data_bspline`'s returned `info`.

    Model in simulate.py:
      F_it ~ N(0, diag(kappa_i * a2_true[i,t,:]))
      Y_it = X_true @ F_it + u_it,  u_it[j] ~ N(0, s_i^2 * Psi_true[j])

    So:
      Var(Y_{i,t,j}) = sum_k X_{j,k}^2 * (kappa_i * a2_{i,t,k}) + s_i^2 * Psi_j
    """
    X_true = np.asarray(info["X_true"], dtype=float)          # (J,r)
    Psi_true = np.asarray(info["Psi_true"], dtype=float)      # (J,)
    s_true = np.asarray(info["s_true"], dtype=float)          # (I,)
    kappa_true = np.asarray(info["kappa_true"], dtype=float)  # (I,)
    a2_true = np.asarray(info["a2_true"], dtype=float)        # (I,T,r)

    Xsq = X_true ** 2  # (J,r)
    var_f = kappa_true[:, None, None] * a2_true  # (I,T,r)
    factor_part = np.einsum("jr,itr->itj", Xsq, var_f)  # (I,T,J)
    noise_part = (s_true ** 2)[:, None, None] * Psi_true[None, None, :]  # (I,1,J)
    return factor_part + noise_part
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
    seed_data = int(main_seed)
    seed_split = 123
    seed_model = 0

    # Align defaults with current notebook settings (override by editing here or adding args later):
    # - kappa_true log-normal with (mu=1, sd=1)
    # - C_true zeros
    # - missingness 0.3..0.6
    # - vol_proxy zeros (=> near-linear warp in the simulator)
    kappa_true = np.exp(rng.normal(1.0, 1.0, size=I))
    C_true = np.zeros((I, r, M_ctrl), dtype=float)
    miss_prob_per_channel = np.linspace(0.3, 0.6, J)
    vol_proxy = np.zeros((I, T, r), dtype=float)

    Y, M, info = simulate_factor_data_bspline(
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
        vol_proxy=vol_proxy,
    )

    # Build flattened indices for observed/test from M
    obs_idx = np.flatnonzero(np.transpose(M, (0, 2, 1)).reshape(-1))  # M is (I,T,J) -> (I,J,T) -> flat
    test_idx = np.flatnonzero(~np.transpose(M, (0, 2, 1)).reshape(-1))

    rng_split = np.random.default_rng(int(seed_split))
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

    # run_em_like has evolved over time; support multiple tuple layouts.
    if isinstance(em_result, tuple) or isinstance(em_result, list):
        if len(em_result) == 9:
            L_hat, _a2_hat, psi_hat, s_hat, kappa_hat, C_hat, u_tilde_hat, _F_hat, _hist = em_result
        elif len(em_result) == 8:
            # No history
            L_hat, _a2_hat, psi_hat, s_hat, kappa_hat, C_hat, u_tilde_hat, _F_hat = em_result
        elif len(em_result) == 7:
            # Legacy layout: no explicit a2/kappa return
            L_hat, psi_hat, s_hat, C_hat, u_tilde_hat, _F_hat, _hist = em_result
            kappa_hat = torch.ones((I,), dtype=L_hat.dtype, device=L_hat.device)
        else:
            raise RuntimeError(f"Unexpected run_em_like return length: {len(em_result)}")
    else:
        raise RuntimeError(f"Unexpected run_em_like return type: {type(em_result)}")

    # --- Build sigma for multiple methods (fit / oracle / avg) ---
    knots = make_open_uniform_knots(M_ctrl, degree, device=device)
    var_fitted = local_var_from_fit(L_hat, psi_hat, s_hat, kappa_hat, C_hat, u_tilde_hat, knots, degree).cpu().numpy()  # (I,T,J)
    sigma_fitted = np.sqrt(np.maximum(var_fitted, 1e-12))
    sigma_fit_flat = np.transpose(sigma_fitted, (0, 2, 1)).reshape(-1)  # (I*J*T,)

    local_var_Y = local_var_Y_from_info(info)  # (I,T,J)
    sigma_oracle = np.sqrt(np.maximum(local_var_Y, 1e-12))
    sigma_oracle_flat = np.transpose(sigma_oracle, (0, 2, 1)).reshape(-1)  # (I*J*T,)

    sigma_avg_flat = 0.5 * (sigma_fit_flat + sigma_oracle_flat)

    sigma_cal = sigma_fit_flat[cal_idx]
    sigma_test = sigma_fit_flat[test_idx]
    sigma_cal_oracle = sigma_oracle_flat[cal_idx]
    sigma_test_oracle = sigma_oracle_flat[test_idx]
    sigma_cal_avg = sigma_avg_flat[cal_idx]
    sigma_test_avg = sigma_avg_flat[test_idx]

    # --- TWFV scaled CP (fit sigma) ---
    lower_tmfv, upper_tmfv, q_tmfv = conformal_scaled_abs(
        Y_cal_true, Y_cal_fit, sigma_cal, Y_test_fit, sigma_test, alpha=alpha, sigma_min=1e-6
    )
    covered_tmfv = (Y_test_true >= lower_tmfv) & (Y_test_true <= upper_tmfv)
    cov_tmfv = float(np.mean(covered_tmfv)) if covered_tmfv.size > 0 else float("nan")

    # --- Oracle scaled CP (oracle sigma) ---
    lower_oracle, upper_oracle, q_oracle = conformal_scaled_abs(
        Y_cal_true, Y_cal_fit, sigma_cal_oracle, Y_test_fit, sigma_test_oracle, alpha=alpha, sigma_min=1e-6
    )
    covered_oracle = (Y_test_true >= lower_oracle) & (Y_test_true <= upper_oracle)
    cov_oracle = float(np.mean(covered_oracle)) if covered_oracle.size > 0 else float("nan")

    # --- Average sigma scaled CP (avg of fit + oracle) ---
    lower_avg, upper_avg, q_avg = conformal_scaled_abs(
        Y_cal_true, Y_cal_fit, sigma_cal_avg, Y_test_fit, sigma_test_avg, alpha=alpha, sigma_min=1e-6
    )
    covered_avg = (Y_test_true >= lower_avg) & (Y_test_true <= upper_avg)
    cov_avg = float(np.mean(covered_avg)) if covered_avg.size > 0 else float("nan")

    # --- SCP (unscaled absolute residual CP) ---
    cp = NaiveAbsoluteResidualCP(shape_ref=Y, J=J, alpha=alpha)
    cp.calibrate_from_fit(cal_idx, Y_cal_true, Y_cal_fit)
    scp_interval = cp.predict_interval_from_fit(test_idx, Y_test_fit, alpha=alpha)  # (n_test,2)
    lower_scp = scp_interval[:, 0]
    upper_scp = scp_interval[:, 1]
    covered_scp = (Y_test_true >= lower_scp) & (Y_test_true <= upper_scp)
    cov_scp = float(np.mean(covered_scp)) if covered_scp.size > 0 else float("nan")

    # --- Conditional reports for later plotting (no plotting during batch run) ---
    # Flatten convention: k=i*(J*T)+j*T+t, so time is k % T, channel is (k//T)%J, subject is k//(J*T)
    t_test = (test_idx % T).astype(int)
    subj_test = (test_idx // (J * T)).astype(int)
    chan_test = ((test_idx // T) % J).astype(int)

    report_tmfv = conditional_interval_report(
        y_true=Y_test_true, lower=lower_tmfv, upper=upper_tmfv, x=t_test, y_hat=Y_test_fit, n_bins=10
    )
    report_oracle = conditional_interval_report(
        y_true=Y_test_true, lower=lower_oracle, upper=upper_oracle, x=t_test, y_hat=Y_test_fit, n_bins=10
    )
    report_avg = conditional_interval_report(
        y_true=Y_test_true, lower=lower_avg, upper=upper_avg, x=t_test, y_hat=Y_test_fit, n_bins=10
    )
    report_scp = conditional_interval_report(
        y_true=Y_test_true, lower=lower_scp, upper=upper_scp, x=t_test, y_hat=Y_test_fit, n_bins=10
    )

    # --- Save per-run artifacts needed for later diagnostics/plots ---
    artifacts = dict(
        main_seed=int(main_seed),
        seed_data=int(seed_data),
        seed_split=int(seed_split),
        seed_model=int(seed_model),
        I=int(I),
        T=int(T),
        J=int(J),
        r=int(r),
        M_ctrl=int(M_ctrl),
        degree=int(degree),
        alpha=float(alpha),
        train_idx=train_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        # For cal_test_scaled_residual_diagnostics:
        # - Y_flat_true is the full flattened Y aligned to k=i*(J*T)+j*T+t
        # - Y_flat_fit is 0 everywhere in this script (innovations); store flag instead of full array
        Y_flat_true=Y_flat,
        yhat_is_zero=True,
        sigma_fit_flat=sigma_fit_flat,
        sigma_oracle_flat=sigma_oracle_flat,
        sigma_avg_flat=sigma_avg_flat,
        q_tmfv=float(q_tmfv),
        q_oracle=float(q_oracle),
        q_avg=float(q_avg),
        q_scp=float(cp.q_abs),
        # Test-point arrays (for quick reuse)
        t_test=t_test,
        subj_test=subj_test,
        chan_test=chan_test,
        Y_test_true=Y_test_true,
        Y_test_fit=Y_test_fit,
        sigma_test=sigma_test,
        sigma_test_oracle=sigma_test_oracle,
        sigma_test_avg=sigma_test_avg,
        lower_tmfv=lower_tmfv,
        upper_tmfv=upper_tmfv,
        lower_oracle=lower_oracle,
        upper_oracle=upper_oracle,
        lower_avg=lower_avg,
        upper_avg=upper_avg,
        lower_scp=lower_scp,
        upper_scp=upper_scp,
        covered_tmfv=covered_tmfv,
        covered_oracle=covered_oracle,
        covered_avg=covered_avg,
        covered_scp=covered_scp,
        # For plot_compare_conditional_reports:
        report_tmfv=report_tmfv,
        report_oracle=report_oracle,
        report_avg=report_avg,
        report_scp=report_scp,
    )

    return {
        "main_seed": int(main_seed),
        "seed_data": seed_data,
        "seed_split": seed_split,
        "seed_model": seed_model,
        "coverage_tmfv": cov_tmfv,
        "coverage_scp": cov_scp,
        "coverage_oracle": cov_oracle,
        "coverage_avg": cov_avg,
        "q_tmfv": float(q_tmfv),
        "q_scp": float(cp.q_abs),
        "q_oracle": float(q_oracle),
        "q_avg": float(q_avg),
        "n_cal": int(cal_idx.size),
        "n_test": int(test_idx.size),
        "_artifacts": artifacts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-seeds", type=int, nargs="+", default=list(range(5,10)))
    p.add_argument("--I", type=int, default=30)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--J", type=int, default=5)
    p.add_argument("--r", type=int, default=1)
    p.add_argument("--M-ctrl", type=int, default=20)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--max-outer", type=int, default=100)
    p.add_argument("--grad-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--variance-tied-missing", action="store_true")
    p.add_argument("--out-csv", type=str, default="batch_results.csv")
    p.add_argument("--artifacts-dir", type=str, default="batch_artifacts")
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
        # Save artifact bundle per run (pickle) for later plotting/diagnostics.
        art = res.pop("_artifacts", None)
        if art is not None:
            out_dir = Path(args.artifacts_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"run_seed_{int(ms)}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(art, f, protocol=pickle.HIGHEST_PROTOCOL)
        results.append(res)
        print(
            f"main_seed={ms} cov_tmfv={res['coverage_tmfv']:.3f} cov_scp={res['coverage_scp']:.3f} "
            f"cov_oracle={res['coverage_oracle']:.3f} n_test={res['n_test']}"
        )


    fieldnames = list(results[0].keys())
    file_exists = os.path.exists(args.out_csv)
    needs_header = (not file_exists) or (os.path.getsize(args.out_csv) == 0)
    mode = "a" if file_exists else "w"

    with open(args.out_csv, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            w.writeheader()
        w.writerows(results)

    covs = np.array([r["coverage_tmfv"] for r in results], dtype=float)
    print(f"coverage_tmfv mean={np.nanmean(covs):.3f} std={np.nanstd(covs):.3f}")

if __name__ == "__main__":
    main()