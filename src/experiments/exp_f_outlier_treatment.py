"""
Experiment F: Outlier Treatment
Clip the TARGET (Prodano) in train only, test stays unclipped.

Strategies:
  1. clip_p99 -- clip train target at p99 (101)
  2. clip_p95 -- clip train target at p95 (49)
  3. clip_per_cat_p95 -- compute p95 per category, clip per-category
  4. winsorize_p95 -- replace values above p95 with p95 value
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def clip_target(y, threshold):
    """Clip target values at threshold."""
    return np.clip(y, 0, threshold)


def main():
    print("=" * 65)
    print("  EXPERIMENT F: OUTLIER TREATMENT (target clipping)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # Compute percentiles from train target
    p95 = np.percentile(y_train, 95)
    p99 = np.percentile(y_train, 99)
    print(f"\n  Train target stats:")
    print(f"    mean={y_train.mean():.1f}, std={y_train.std():.1f}")
    print(f"    p95={p95:.1f}, p99={p99:.1f}, max={y_train.max():.1f}")

    # Per-category p95
    train_df = pd.DataFrame({"target": y_train.values, "cat": X_train["Категория"].values})
    cat_p95 = train_df.groupby("cat")["target"].quantile(0.95).to_dict()
    print(f"    Per-category p95: {cat_p95}")

    # --- Baseline ---
    print("\n--- Baseline (no clipping) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    # --- Strategies ---
    strategies = {
        "clip_p99": lambda y, cats: clip_target(y, p99),
        "clip_p95": lambda y, cats: clip_target(y, p95),
        "clip_per_cat_p95": lambda y, cats: np.array([
            min(v, cat_p95.get(c, p95)) for v, c in zip(y, cats)
        ]),
        "winsorize_p95": lambda y, cats: clip_target(y, p95),  # same as clip_p95
    }

    results = []

    for name, transform_fn in strategies.items():
        print(f"\n--- {name} ---")

        # Clip train target
        y_train_clipped = transform_fn(
            y_train.values, X_train["Категория"].values
        )
        n_clipped = np.sum(y_train_clipped != y_train.values)
        print(f"  Obrezano strok: {n_clipped:,} ({n_clipped / len(y_train) * 100:.2f}%)")
        print(f"  Train target posle: mean={y_train_clipped.mean():.2f}, "
              f"max={y_train_clipped.max():.1f}")

        # Train and predict
        model = train_lgbm(X_train, y_train_clipped)
        pred = predict_clipped(model, X_test)

        mae, wm, bias = print_metrics(name, y_test, pred)

        print_category_metrics(y_test, pred, X_test["Категория"])

        results.append({
            "strategy": name,
            "mae": mae,
            "wmape": wm,
            "bias": bias,
            "n_clipped": int(n_clipped),
            "delta": mae - BASELINE_MAE,
        })

        save_predictions(
            X_test, y_test, pred,
            f"reports/exp_f_{name}_predictions.csv",
        )

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  ITOGI EKSPERIMENTA F")
    print("=" * 65)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    print(f"\n  {'Strategy':<22} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'Delta':>8} {'Clipped':>8}")
    print(f"  {'-' * 58}")

    # Baseline row
    print(f"  {'baseline':<22} {bl_mae:>8.4f} {'':>8} {'':>8} {'0.0000':>8} {'0':>8}")

    for _, row in results_df.iterrows():
        marker = " <--" if row["delta"] < 0 else ""
        print(f"  {row['strategy']:<22} {row['mae']:>8.4f} {row['wmape']:>7.2f}% "
              f"{row['bias']:>+8.4f} {row['delta']:>+8.4f} {row['n_clipped']:>8}{marker}")

    # Best result
    best = results_df.iloc[0]
    print(f"\n  Luchshaya strategiya: {best['strategy']}")
    print(f"  MAE = {best['mae']:.4f} (delta vs baseline: {best['delta']:+.4f})")

    # Save summary (best row for compare_all)
    summary = pd.DataFrame([{
        "experiment": f"F_{best['strategy']}",
        "mae": best["mae"],
        "wmape": best["wmape"],
        "bias": best["bias"],
        "baseline_mae": BASELINE_MAE,
        "delta": best["delta"],
    }])
    summary.to_csv("reports/exp_f_summary.csv", index=False)

    # Save full results
    results_df.to_csv("reports/exp_f_all_results.csv", index=False, encoding="utf-8-sig")
    print(f"  Polnye rezul'taty: reports/exp_f_all_results.csv")

    print("\nGotovo!")


if __name__ == "__main__":
    main()
