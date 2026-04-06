"""
Run all High Demand Problem experiments (40-46) sequentially.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/run_high_demand.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

EXPERIMENTS = [
    ("40_tweedie",           "40_tweedie/run.py"),
    ("41_log_target",        "41_log_target/run.py"),
    ("42_asymmetric_loss",   "42_asymmetric_loss/run.py"),
    ("43_quantile",          "43_quantile/run.py"),
    ("44_mixture_of_experts","44_mixture_of_experts/run.py"),
    ("45_log_residual",      "45_log_residual/run.py"),
    ("46_variance_weighted", "46_variance_weighted/run.py"),
]

EXP_DIR = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("  HIGH DEMAND PROBLEM: Running experiments 40-46")
    print("=" * 60)
    t_start = time.time()

    results = []
    for i, (name, script) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{len(EXPERIMENTS)}] {name}")
        print(f"{'#' * 60}\n")

        script_path = str(EXP_DIR / script)
        t0 = time.time()

        ret = subprocess.run(
            [PYTHON, script_path],
            cwd=str(ROOT),
        )

        elapsed = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"FAIL (code={ret.returncode})"
        results.append((name, status, elapsed))
        print(f"\n  >>> {name}: {status} ({elapsed:.0f}s)")

    # Summary
    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({total:.0f}s total, {total/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  {'Experiment':<25} {'Status':<12} {'Time':>8}")
    print(f"  {'-'*47}")
    for name, status, elapsed in results:
        print(f"  {name:<25} {status:<12} {elapsed:>7.0f}s")

    # Load and compare metrics
    print(f"\n  === METRICS COMPARISON ===")
    print(f"  {'Experiment':<25} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'R2':>8}")
    print(f"  {'-'*59}")

    import json
    baseline = {"mae": 2.2904, "wmape": 25.91}
    print(f"  {'01_baseline_8m':<25} {baseline['mae']:>8.4f} {baseline['wmape']:>7.2f}%  {'--':>8} {'--':>8}")

    for name, _, _ in results:
        metrics_path = EXP_DIR / name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            mae_delta = m["mae"] - baseline["mae"]
            print(f"  {name:<25} {m['mae']:>8.4f} {m['wmape']:>7.2f}% {m.get('bias', 0):>+8.4f} {m.get('r2', 0):>8.4f}  ({mae_delta:+.4f})")
        else:
            print(f"  {name:<25} {'no metrics':>8}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
