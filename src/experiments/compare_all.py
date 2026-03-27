"""
Compare All Architecture Experiments
Reads summary CSVs from each experiment, prints final comparison table.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np

from src.experiments.common import BASELINE_MAE


SUMMARY_FILES = {
    "A: Per-Category":    "reports/exp_a_summary.csv",
    "B: Stacking":        "reports/exp_b_summary.csv",
    "C: Store Clusters":  "reports/exp_c_summary.csv",
    "D: Two-Stage":       "reports/exp_d_summary.csv",
    "E: Residual":        "reports/exp_e_summary.csv",
    "F: Outlier Treatment":  "reports/exp_f_summary.csv",
    "G: Noise Filtering":    "reports/exp_g_summary.csv",
    "H: Censored Demand":    "reports/exp_h_summary.csv",
    "I: Combined Best":      "reports/exp_i_summary.csv",
}


def main():
    print("=" * 70)
    print("  SRAVNENIE VSEKH EKSPERIMENTOV (A-I)")
    print("=" * 70)

    rows = []

    # Baseline row
    rows.append({
        "Experiment": "Baseline (v6-best)",
        "MAE": BASELINE_MAE,
        "WMAPE": None,
        "Bias": None,
        "Delta": 0.0,
    })

    # Load each experiment summary
    found = 0
    for name, path in SUMMARY_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                row = df.iloc[0]
                rows.append({
                    "Experiment": name,
                    "MAE": row["mae"],
                    "WMAPE": row.get("wmape", None),
                    "Bias": row.get("bias", None),
                    "Delta": row["mae"] - BASELINE_MAE,
                })
                found += 1
        else:
            print(f"  [!] Ne najden: {path}")
            rows.append({
                "Experiment": name,
                "MAE": None,
                "WMAPE": None,
                "Bias": None,
                "Delta": None,
            })

    print(f"\n  Najdeno eksperimentov: {found}/{len(SUMMARY_FILES)}")

    # Summary table
    results = pd.DataFrame(rows)
    results_sorted = results.dropna(subset=["MAE"]).sort_values("MAE")

    print("\n" + "=" * 70)
    print(f"  {'Experiment':<25} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'Delta':>8}")
    print(f"  {'-' * 62}")

    for _, row in results_sorted.iterrows():
        mae_str = f"{row['MAE']:.4f}" if pd.notna(row['MAE']) else "  --  "
        wmape_str = f"{row['WMAPE']:.2f}%" if pd.notna(row['WMAPE']) else "  --  "
        bias_str = f"{row['Bias']:+.4f}" if pd.notna(row['Bias']) else "  --  "
        delta_str = f"{row['Delta']:+.4f}" if pd.notna(row['Delta']) else "  --  "

        marker = ""
        if pd.notna(row['Delta']) and row['Delta'] < 0:
            marker = " <--"

        print(f"  {row['Experiment']:<25} {mae_str:>8} {wmape_str:>8} {bias_str:>8} {delta_str:>8}{marker}")

    print(f"  {'-' * 62}")

    # Winner
    valid = results_sorted[results_sorted["MAE"].notna() & (results_sorted["Experiment"] != "Baseline (v6-best)")]
    if len(valid) > 0:
        best = valid.iloc[0]
        print(f"\n  Luchshij eksperiment: {best['Experiment']}")
        print(f"  MAE = {best['MAE']:.4f} (delta vs baseline: {best['Delta']:+.4f})")

        if best["Delta"] < 0:
            improvement_pct = abs(best["Delta"]) / BASELINE_MAE * 100
            print(f"  Uluchshenie: {improvement_pct:.2f}% vs baseline")
        else:
            print(f"  Baseline ostaetsya luchshim")

    # Save combined results
    results_sorted.to_csv("reports/all_experiments.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Tablitsa sokhranena: reports/all_experiments.csv")

    print("\nGotovo!")


if __name__ == "__main__":
    main()
