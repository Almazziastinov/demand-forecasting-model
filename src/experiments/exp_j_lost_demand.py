"""
Experiment J: Lost Demand Correction
Uses 16-day "Upushchennye" data to build a lookup table of lost demand
by (Category x Prodano_bucket x Ostatok_01), then applies correction
to 3-month train target: target_adj = Prodano + lost_qty_estimate.

Strategies:
  J1: full correction (mean lost from lookup)
  J2: partial correction x0.5
  J3: median-based correction (more conservative)а
  J4: correction only where Ostatok==0 (censored only)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from src.experiments.common import (
    BASELINE_MAE,
    FEATURES,
    MODEL_PARAMS,
    TARGET,
    load_data,
    predict_clipped,
    print_category_metrics,
    print_metrics,
    save_predictions,
    train_lgbm,
    wmape,
)

LOOKUP_PATH = "data/processed/lost_qty_lookup.csv"


def build_lookup():
    """Load lookup table from lost demand analysis."""
    lk = pd.read_csv(LOOKUP_PATH)
    # Build dict: (Category, prod_bucket, ost01) -> (mean_lost, median_lost)
    lookup_mean = {}
    lookup_median = {}
    for _, row in lk.iterrows():
        key = (row["Category"], row["prod_bucket"], int(row["ost01"]))
        lookup_mean[key] = row["lost_mean"]
        lookup_median[key] = row["lost_median"]
    return lookup_mean, lookup_median


def assign_prod_bucket(prodano):
    """Assign Prodano to bucket matching the lookup."""
    if prodano <= 2:
        return "0-2"
    elif prodano <= 5:
        return "3-5"
    elif prodano <= 10:
        return "6-10"
    elif prodano <= 20:
        return "11-20"
    elif prodano <= 50:
        return "21-50"
    else:
        return "51+"


def apply_correction(
    train_full, features, lookup, use_median=False, scale=1.0, censored_only=False
):
    """Apply lost demand correction to train target.

    Returns: X_train, y_train_adjusted, n_adjusted, avg_shift
    """
    y_orig = train_full[TARGET].values.copy().astype(float)
    y_adj = y_orig.copy()

    cats = train_full["Категория"].values
    prodano = train_full[TARGET].values
    ostatok = train_full["Остаток"].values

    n_adjusted = 0
    total_shift = 0.0

    for i in range(len(y_adj)):
        ost01 = 1 if ostatok[i] == 0 else 0

        if censored_only and ost01 == 0:
            continue

        bucket = assign_prod_bucket(prodano[i])
        key = (cats[i], bucket, ost01)
        correction = lookup.get(key, 0.0)

        if correction > 0:
            y_adj[i] = y_orig[i] + correction * scale
            n_adjusted += 1
            total_shift += correction * scale

    avg_shift = total_shift / max(n_adjusted, 1)

    X_train = train_full[features]
    return X_train, y_adj, n_adjusted, avg_shift


def main():
    print("=" * 65)
    print("  EXPERIMENT J: LOST DEMAND CORRECTION")
    print("  (based on 16-day 'Upushchennye' data)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train_full = df[df["Дата"] < test_start].copy()

    # Load lookup
    lookup_mean, lookup_median = build_lookup()
    print(f"\n  Lookup: {len(lookup_mean)} entries (Category x ProdBucket x Ost01)")

    # --- Baseline ---
    print("\n--- Baseline (bez korrektsii) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    results = []

    # ========================================
    # J1: Full mean correction
    # ========================================
    print("\n--- J1: Full mean correction (scale=1.0) ---")
    X_j1, y_j1, n1, shift1 = apply_correction(
        train_full, features, lookup_mean, scale=1.0
    )
    print(f"  Skorrektirovano: {n1:,} strok, srednij sdvig: +{shift1:.3f}")
    print(f"  Target: mean {y_train.mean():.2f} -> {y_j1.mean():.2f}")

    model_j1 = train_lgbm(X_j1, y_j1)
    pred_j1 = predict_clipped(model_j1, X_test)
    mae_j1, wm_j1, bias_j1 = print_metrics("J1_full_mean", y_test, pred_j1)
    print_category_metrics(y_test, pred_j1, X_test["Категория"])

    results.append(
        {
            "strategy": "J1_full_mean",
            "mae": mae_j1,
            "wmape": wm_j1,
            "bias": bias_j1,
            "delta": mae_j1 - BASELINE_MAE,
        }
    )
    save_predictions(X_test, y_test, pred_j1, "reports/exp_j1_predictions.csv")

    # ========================================
    # J2: Partial correction x0.5
    # ========================================
    print("\n--- J2: Partial mean correction (scale=0.5) ---")
    X_j2, y_j2, n2, shift2 = apply_correction(
        train_full, features, lookup_mean, scale=0.5
    )
    print(f"  Skorrektirovano: {n2:,} strok, srednij sdvig: +{shift2:.3f}")
    print(f"  Target: mean {y_train.mean():.2f} -> {y_j2.mean():.2f}")

    model_j2 = train_lgbm(X_j2, y_j2)
    pred_j2 = predict_clipped(model_j2, X_test)
    mae_j2, wm_j2, bias_j2 = print_metrics("J2_partial_0.5", y_test, pred_j2)
    print_category_metrics(y_test, pred_j2, X_test["Категория"])

    results.append(
        {
            "strategy": "J2_partial_0.5",
            "mae": mae_j2,
            "wmape": wm_j2,
            "bias": bias_j2,
            "delta": mae_j2 - BASELINE_MAE,
        }
    )
    save_predictions(X_test, y_test, pred_j2, "reports/exp_j2_predictions.csv")

    # ========================================
    # J3: Median correction (conservative)
    # ========================================
    print("\n--- J3: Median correction (bolee konservativno) ---")
    X_j3, y_j3, n3, shift3 = apply_correction(
        train_full, features, lookup_median, use_median=True, scale=1.0
    )
    print(f"  Skorrektirovano: {n3:,} strok, srednij sdvig: +{shift3:.3f}")
    print(f"  Target: mean {y_train.mean():.2f} -> {y_j3.mean():.2f}")

    model_j3 = train_lgbm(X_j3, y_j3)
    pred_j3 = predict_clipped(model_j3, X_test)
    mae_j3, wm_j3, bias_j3 = print_metrics("J3_median", y_test, pred_j3)
    print_category_metrics(y_test, pred_j3, X_test["Категория"])

    results.append(
        {
            "strategy": "J3_median",
            "mae": mae_j3,
            "wmape": wm_j3,
            "bias": bias_j3,
            "delta": mae_j3 - BASELINE_MAE,
        }
    )
    save_predictions(X_test, y_test, pred_j3, "reports/exp_j3_predictions.csv")

    # ========================================
    # J4: Correction only where Ostatok==0
    # ========================================
    print("\n--- J4: Mean correction ONLY where Ostatok==0 ---")
    X_j4, y_j4, n4, shift4 = apply_correction(
        train_full, features, lookup_mean, scale=1.0, censored_only=True
    )
    print(f"  Skorrektirovano: {n4:,} strok, srednij sdvig: +{shift4:.3f}")
    print(f"  Target: mean {y_train.mean():.2f} -> {y_j4.mean():.2f}")

    model_j4 = train_lgbm(X_j4, y_j4)
    pred_j4 = predict_clipped(model_j4, X_test)
    mae_j4, wm_j4, bias_j4 = print_metrics("J4_censored_only", y_test, pred_j4)
    print_category_metrics(y_test, pred_j4, X_test["Категория"])

    results.append(
        {
            "strategy": "J4_censored_only",
            "mae": mae_j4,
            "wmape": wm_j4,
            "bias": bias_j4,
            "delta": mae_j4 - BASELINE_MAE,
        }
    )
    save_predictions(X_test, y_test, pred_j4, "reports/exp_j4_predictions.csv")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  ITOGI EKSPERIMENTA J")
    print("=" * 65)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    print(f"\n  {'Strategy':<22} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'Delta':>8}")
    print(f"  {'-' * 50}")
    print(f"  {'baseline':<22} {bl_mae:>8.4f} {'':>8} {'':>8} {'0.0000':>8}")

    for _, row in results_df.iterrows():
        marker = " <--" if row["delta"] < 0 else ""
        print(
            f"  {row['strategy']:<22} {row['mae']:>8.4f} {row['wmape']:>7.2f}% "
            f"{row['bias']:>+8.4f} {row['delta']:>+8.4f}{marker}"
        )

    best = results_df.iloc[0]
    print(f"\n  Luchshaya strategiya: {best['strategy']}")
    print(f"  MAE = {best['mae']:.4f} (delta vs baseline: {best['delta']:+.4f})")

    # Save summary
    summary = pd.DataFrame(
        [
            {
                "experiment": f"J_{best['strategy']}",
                "mae": best["mae"],
                "wmape": best["wmape"],
                "bias": best["bias"],
                "baseline_mae": BASELINE_MAE,
                "delta": best["delta"],
            }
        ]
    )
    summary.to_csv("reports/exp_j_summary.csv", index=False)

    results_df.to_csv(
        "reports/exp_j_all_results.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  Polnye rezul'taty: reports/exp_j_all_results.csv")

    print("\nGotovo!")


if __name__ == "__main__":
    main()
