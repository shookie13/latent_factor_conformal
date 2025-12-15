"""
Demo script: generate synthetic data, fit a simple mean predictor on observed entries,
calibrate conformal prediction, and evaluate on masked (missing) entries.

Train/calibration data: observed entries (M == True)
Test data: masked/missing entries (M == False)
"""

from __future__ import annotations

import numpy as np

from twfv.simulate import simulate_factor_data_bspline
from twfv.cp import NaiveAbsoluteResidualCP, unflatten


def flatten(i: int, j: int, t: int, I: int, J: int, T: int) -> int:
    return i * (J * T) + j * T + t


def main():
    # --- 1) Simulate data ---
    I, T, J, r, M_ctrl = 5, 40, 8, 2, 6
    Y, M, info = simulate_factor_data_bspline(
        I, T, J, r, M_ctrl=M_ctrl, degree=3, seed=0, use_variance_tied_missing=True
    )
    # Y: (I,T,J) full innovations; M: bool observed mask

    # --- 2) Build indices ---
    obs_idx_list = []
    test_idx_list = []
    for i in range(I):
        for t in range(T):
            for j in range(J):
                k = flatten(i, j, t, I, J, T)
                if M[i, t, j]:
                    obs_idx_list.append(k)
                else:
                    test_idx_list.append(k)
    obs_idx = np.array(obs_idx_list, dtype=int)
    test_idx = np.array(test_idx_list, dtype=int)

    # Shuffle observed indices and split into train/cal
    rng = np.random.default_rng(123)
    rng.shuffle(obs_idx)
    split = int(0.7 * obs_idx.size)
    train_idx = obs_idx[:split]
    cal_idx = obs_idx[split:]

    # --- 3) Simple mean predictor (per-channel mean over train observed) ---
    channel_sums = np.zeros(J, dtype=float)
    channel_counts = np.zeros(J, dtype=float)
    for k in train_idx:
        i, j, t = unflatten(int(k), I, J, T)
        channel_sums[j] += Y[i, t, j]
        channel_counts[j] += 1.0
    channel_means = channel_sums / np.clip(channel_counts, 1.0, None)

    def predict_mean(idx_array: np.ndarray) -> np.ndarray:
        preds = np.zeros(idx_array.size, dtype=float)
        for n, k in enumerate(idx_array.tolist()):
            _, j, _ = unflatten(int(k), I, J, T)
            preds[n] = channel_means[j]
        return preds

    # Calibration true and fitted
    Y_cal_true = np.zeros(cal_idx.size, dtype=float)
    for n, k in enumerate(cal_idx.tolist()):
        i, j, t = unflatten(int(k), I, J, T)
        Y_cal_true[n] = Y[i, t, j]
    Y_cal_fit = predict_mean(cal_idx)

    # Test true and fitted
    Y_test_true = np.zeros(test_idx.size, dtype=float)
    for n, k in enumerate(test_idx.tolist()):
        i, j, t = unflatten(int(k), I, J, T)
        Y_test_true[n] = Y[i, t, j]
    Y_test_fit = predict_mean(test_idx)

    # --- 4) Conformal calibration and prediction ---
    cp = NaiveAbsoluteResidualCP(shape_ref=Y, J=J, alpha=0.1)
    cp.calibrate_from_fit(cal_idx, Y_cal_true, Y_cal_fit)
    intervals = cp.predict_interval_from_fit(test_idx, Y_test_fit, alpha=0.1)

    # --- 5) Evaluate coverage on masked entries ---
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    covered = (Y_test_true >= lower) & (Y_test_true <= upper)
    cov = float(np.mean(covered)) if covered.size > 0 else float("nan")

    print("Synthetic CP demo")
    print(f"I={I}, T={T}, J={J}, r={r}, M_ctrl={M_ctrl}")
    print(f"Observed entries: {obs_idx.size} (train={train_idx.size}, cal={cal_idx.size})")
    print(f"Test (masked) entries: {test_idx.size}")
    print(f"Per-channel mean baseline coverage on test: {cov:.3f}")


if __name__ == "__main__":
    main()

