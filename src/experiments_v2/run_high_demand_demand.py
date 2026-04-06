"""
Run all High Demand Problem experiments on DEMAND target (50-56) sequentially.

These mirror experiments 40-46 but train on Spros (demand) instead of Prodano (sales).
Baseline: exp 03 Model B (MAE 2.9923, WMAPE 28.19% vs Spros).

Usage:
  .venv/Scripts/python.exe src/experiments_v2/run_high_demand_demand.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

EXPERIMENTS = [
    ("50_tweedie_demand",           "50_tweedie_demand/run.py"),
    ("51_log_target_demand",        "51_log_target_demand/run.py"),
    ("52_asymmetric_loss_demand",   "52_asymmetric_loss_demand/run.py"),
    ("53_quantile_demand",          "53_quantile_demand/run.py"),
    ("54_mixture_of_experts_demand","54_mixture_of_experts_demand/run.py"),
    ("55_log_residual_demand",      "55_log_residual_demand/run.py"),
    ("56_variance_weighted_demand", "56_variance_weighted_demand/run.py"),
]

EXP_DIR = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("  HIGH DEMAND PROBLEM (DEMAND TARGET): Running experiments 50-56")
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
    print(f"  {'Experiment':<30} {'Status':<12} {'Time':>8}")
    print(f"  {'-'*52}")
    for name, status, elapsed in results:
        print(f"  {name:<30} {status:<12} {elapsed:>7.0f}s")

    # Load and compare metrics
    print(f"\n  === METRICS COMPARISON (vs Spros) ===")
    print(f"  {'Experiment':<30} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'R2':>8}")
    print(f"  {'-'*64}")

    import json
    baseline = {"mae": 2.9923, "wmape": 28.19}
    print(f"  {'03_demand_baseline':<30} {baseline['mae']:>8.4f} {baseline['wmape']:>7.2f}%  {'--':>8} {'--':>8}")

    for name, _, _ in results:
        metrics_path = EXP_DIR / name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            mae_delta = m["mae"] - baseline["mae"]
            print(f"  {name:<30} {m['mae']:>8.4f} {m['wmape']:>7.2f}% {m.get('bias', 0):>+8.4f} {m.get('r2', 0):>8.4f}  ({mae_delta:+.4f})")
        else:
            print(f"  {name:<30} {'no metrics':>8}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
