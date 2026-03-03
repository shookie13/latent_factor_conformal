from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow running as a script from the repo root or from the scripts/ directory on Windows.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from twfv import cal_test_scaled_residual_diagnostics  # noqa: E402


Split = Literal["cal", "test"]
Method = Literal["tmfv", "oracle", "avg"]


def _load_artifacts(p: Path) -> dict:
    with p.open("rb") as f:
        return pickle.load(f)


def _select_sigma_and_q(art: dict, method: Method) -> tuple[np.ndarray, float]:
    if method == "tmfv":
        return np.asarray(art["sigma_fit_flat"], dtype=float), float(art["q_tmfv"])
    if method == "oracle":
        return np.asarray(art["sigma_oracle_flat"], dtype=float), float(art["q_oracle"])
    if method == "avg":
        return np.asarray(art["sigma_avg_flat"], dtype=float), float(art["q_avg"])
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize tail-rate gap curves across batch runs.")
    p.add_argument("--artifacts-dir", type=str, default="batch_artifacts", help="Directory containing run_seed_*.pkl files.")
    p.add_argument("--method", type=str, default="tmfv", choices=["tmfv", "oracle", "avg"])
    p.add_argument("--split", type=str, default="test", choices=["cal", "test"])
    p.add_argument("--n-time-bins", type=int, default=10)
    p.add_argument("--alpha", type=float, default=None, help="Override alpha; default uses alpha stored in artifact.")
    p.add_argument("--abs-gap", action="store_true", help="Plot |tail-alpha| instead of tail-alpha.")
    p.add_argument("--no-individual", action="store_true", help="Hide per-seed curves; only show mean±sd.")
    p.add_argument("--out", type=str, default="batch_tail_gap.png")
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    files = sorted(artifacts_dir.glob("run_seed_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No artifacts found in {artifacts_dir}. Expected run_seed_*.pkl")

    method: Method = args.method  # type: ignore[assignment]
    split: Split = args.split  # type: ignore[assignment]
    n_bins = int(args.n_time_bins)

    gaps = []
    centers_ref = None
    seeds = []

    for fp in files:
        art = _load_artifacts(fp)
        I = int(art["I"])
        J = int(art["J"])
        T = int(art["T"])

        alpha = float(art["alpha"]) if args.alpha is None else float(args.alpha)
        sigma_flat, q = _select_sigma_and_q(art, method)

        cal_idx = np.asarray(art["cal_idx"], dtype=int)
        test_idx = np.asarray(art["test_idx"], dtype=int)
        Y_flat_true = np.asarray(art["Y_flat_true"], dtype=float)

        # yhat is 0 in these experiments; reconstruct the flat fit array.
        Y_flat_fit = np.zeros_like(Y_flat_true)

        diag = cal_test_scaled_residual_diagnostics(
            cal_idx=cal_idx,
            test_idx=test_idx,
            Y_flat_true=Y_flat_true,
            Y_flat_fit=Y_flat_fit,
            sigma_flat=sigma_flat,
            I=I,
            J=J,
            T=T,
            q=q,
            n_time_bins=n_bins,
        )
        d = diag.cal if split == "cal" else diag.test
        centers = np.asarray(d["centers"], dtype=float)
        tail = np.asarray(d["tail"], dtype=float)
        gap = tail - alpha
        if args.abs_gap:
            gap = np.abs(gap)

        if centers_ref is None:
            centers_ref = centers
        else:
            # Ensure consistent bins across runs
            if centers_ref.shape != centers.shape or not np.allclose(centers_ref, centers):
                raise RuntimeError("Inconsistent time-bin centers across artifacts; check that all runs share the same T and binning.")

        gaps.append(gap)
        seeds.append(int(art.get("main_seed", -1)))

    G = np.stack(gaps, axis=0)  # (n_runs, n_bins)
    mean = np.nanmean(G, axis=0)
    sd = np.nanstd(G, axis=0)

    centers = np.asarray(centers_ref, dtype=float)
    ylabel = "|tail-alpha|" if args.abs_gap else "tail - alpha"
    title = f"Tail-rate gap by time bin ({method}, {split}); n_runs={G.shape[0]}"

    plt.figure(figsize=(11, 3.6))
    if not args.no_individual:
        for i in range(G.shape[0]):
            plt.plot(centers, G[i], color="C0", alpha=0.18, linewidth=1)

    plt.plot(centers, mean, color="C0", linewidth=2.5, label="mean across runs")
    plt.fill_between(centers, mean - sd, mean + sd, color="C0", alpha=0.22, label="±1 sd across runs")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("time (bin centers)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    # Also print a compact numeric summary: per-run average gap magnitude across bins.
    per_run = np.nanmean(np.abs(G), axis=1)
    print(f"Per-run mean(|gap|) across bins: mean={float(np.mean(per_run)):.4f} sd={float(np.std(per_run)):.4f}")
    # Show a few runs to help spot outliers
    order = np.argsort(per_run)[::-1]
    top = min(5, per_run.size)
    for k in range(top):
        idx = int(order[k])
        print(f"  seed={seeds[idx]} mean(|gap|)={float(per_run[idx]):.4f}")


if __name__ == "__main__":
    main()


