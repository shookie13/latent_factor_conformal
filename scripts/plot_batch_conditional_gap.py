from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(p: Path) -> dict:
    with p.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregated conditional plots (abs coverage gap) across batch artifacts.")
    p.add_argument("--artifacts-dir", type=str, default="batch_artifacts")
    p.add_argument("--a", type=str, default="tmfv", choices=["tmfv", "scp", "oracle", "avg"])
    p.add_argument("--b", type=str, default="scp", choices=["tmfv", "scp", "oracle", "avg"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--key", type=str, default="X")
    p.add_argument("--title", type=str, default="Batch aggregated conditional plots")
    p.add_argument("--save-dir", type=str, default="result_img")
    p.add_argument("--save-basename", type=str, default="batch_agg")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--heatmap-style", type=str, default="contourf", choices=["imshow", "contourf"])
    p.add_argument("--heatmap-levels", type=int, default=20)
    p.add_argument("--heatmap-upsample", type=int, default=2)
    args = p.parse_args()

    # Ensure repo root is on sys.path when running from scripts/
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from twfv.plot import plot_compare_conditional_reports_aggregated  # noqa: E402

    artifacts_dir = Path(args.artifacts_dir)
    files = sorted(artifacts_dir.glob("run_seed_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No run_seed_*.pkl found in {artifacts_dir}")
    artifacts = [_load(fp) for fp in files]

    def map_key(m: str) -> tuple[str, str]:
        return f"report_{m}", f"covered_{m}"

    report_key_a, covered_key_a = map_key(args.a)
    report_key_b, covered_key_b = map_key(args.b)

    plot_compare_conditional_reports_aggregated(
        artifacts,
        report_key_a=report_key_a,
        report_key_b=report_key_b,
        covered_key_a=covered_key_a,
        covered_key_b=covered_key_b,
        channel_key="chan_test",
        subject_key="subj_test",
        label_a=args.a.upper(),
        label_b=args.b.upper(),
        key=args.key,
        alpha_target=float(args.alpha),
        title=args.title,
        show=not args.no_show,
        save_plots=not args.no_save,
        save_dir=args.save_dir,
        save_basename=args.save_basename,
        heatmap_style=args.heatmap_style,
        heatmap_levels=int(args.heatmap_levels),
        heatmap_upsample=int(args.heatmap_upsample),
    )


if __name__ == "__main__":
    main()


