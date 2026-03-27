"""
Experiment H: Censored Demand Correction
When stock ran out (Ostatok == 0), observed sales understate true demand.

Strategies:
  1. demand_correction -- where Ostatok==0: adjusted = max(Prodano, sales_roll_mean7)
  2. exclude_censored -- remove train rows where Ostatok==0
  3. deficit_correction -- where Ostatok==0: adjusted = Prodano + stock_deficit * 0.5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from src.experiments.common import (
    load_data, FEATURES, TARGET, MODEL_PARAMS, BASELINE_MAE,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def main():
    print("=" * 65)
    print("  EXPERIMENT H: CENSORED DEMAND CORRECTION")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # Access full train df for Ostatok, stock_deficit, sales_roll_mean7
    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train_full = df[df["Дата"] < test_start].copy()

    # Censored demand stats
    censored_mask = train_full["Остаток"] == 0
    n_censored = censored_mask.sum()
    print(f"\n  Tsenzurirovannye nablyudeniya v traine (Ostatok==0):")
    print(f"    {n_censored:,} iz {len(train_full):,} ({n_censored / len(train_full) * 100:.1f}%)")
    print(f"    Srednie prodazhi pri Ostatok==0: {train_full.loc[censored_mask, 'Продано'].mean():.2f}")
    print(f"    Srednie prodazhi pri Ostatok>0:  {train_full.loc[~censored_mask, 'Продано'].mean():.2f}")

    # Check stock_deficit
    if "stock_deficit" in train_full.columns:
        deficit_vals = train_full.loc[censored_mask, "stock_deficit"]
        print(f"    stock_deficit pri Ostatok==0: mean={deficit_vals.mean():.2f}, "
              f"median={deficit_vals.median():.2f}")

    # --- Baseline ---
    print("\n--- Baseline (bez korrektsii) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    results = []

    # ========================================
    # H1: demand_correction (max of actual vs rolling mean7)
    # ========================================
    print("\n--- H1: demand_correction (adjusted = max(Prodano, sales_roll_mean7)) ---")

    y_train_h1 = y_train.copy()
    roll_mean7 = train_full["sales_roll_mean7"].values
    censored_idx = censored_mask.values

    y_h1_vals = y_train_h1.values.copy().astype(float)
    y_h1_vals[censored_idx] = np.maximum(
        y_h1_vals[censored_idx],
        np.nan_to_num(roll_mean7[censored_idx], nan=0.0)
    )
    n_adjusted = np.sum(y_h1_vals != y_train.values)
    print(f"  Skorrektировано strok: {n_adjusted:,}")
    print(f"  Srednij sdvig targeta: {(y_h1_vals - y_train.values).mean():+.3f}")

    model_h1 = train_lgbm(X_train, y_h1_vals)
    pred_h1 = predict_clipped(model_h1, X_test)
    mae_h1, wm_h1, bias_h1 = print_metrics("H1_demand_correction", y_test, pred_h1)
    print_category_metrics(y_test, pred_h1, X_test["Категория"])

    results.append({
        "strategy": "H1_demand_correction",
        "mae": mae_h1, "wmape": wm_h1, "bias": bias_h1,
        "delta": mae_h1 - BASELINE_MAE,
    })
    save_predictions(X_test, y_test, pred_h1, "reports/exp_h1_predictions.csv")

    # ========================================
    # H2: exclude_censored (remove Ostatok==0 rows)
    # ========================================
    print("\n--- H2: exclude_censored (udalenie strok s Ostatok==0) ---")

    train_uncensored = train_full[~censored_mask]
    X_train_h2 = train_uncensored[features]
    y_train_h2 = train_uncensored[TARGET]
    print(f"  Train posle udaleniya: {len(X_train_h2):,} strok "
          f"(udaleno {n_censored:,})")

    model_h2 = train_lgbm(X_train_h2, y_train_h2)
    pred_h2 = predict_clipped(model_h2, X_test)
    mae_h2, wm_h2, bias_h2 = print_metrics("H2_exclude_censored", y_test, pred_h2)
    print_category_metrics(y_test, pred_h2, X_test["Категория"])

    results.append({
        "strategy": "H2_exclude_censored",
        "mae": mae_h2, "wmape": wm_h2, "bias": bias_h2,
        "delta": mae_h2 - BASELINE_MAE,
    })
    save_predictions(X_test, y_test, pred_h2, "reports/exp_h2_predictions.csv")

    # ========================================
    # H3: deficit_correction (Prodano + stock_deficit * 0.5)
    # ========================================
    print("\n--- H3: deficit_correction (adjusted = Prodano + stock_deficit * 0.5) ---")

    y_h3_vals = y_train.values.copy().astype(float)
    if "stock_deficit" in train_full.columns:
        deficit = train_full["stock_deficit"].values
        y_h3_vals[censored_idx] = y_h3_vals[censored_idx] + \
            np.nan_to_num(deficit[censored_idx], nan=0.0) * 0.5
        n_adj_h3 = np.sum(y_h3_vals != y_train.values)
        print(f"  Skorrektirovano strok: {n_adj_h3:,}")
        print(f"  Srednij sdvig targeta: {(y_h3_vals - y_train.values).mean():+.3f}")
    else:
        print("  [!] stock_deficit ne najden, propuskayem")

    model_h3 = train_lgbm(X_train, y_h3_vals)
    pred_h3 = predict_clipped(model_h3, X_test)
    mae_h3, wm_h3, bias_h3 = print_metrics("H3_deficit_correction", y_test, pred_h3)
    print_category_metrics(y_test, pred_h3, X_test["Категория"])

    results.append({
        "strategy": "H3_deficit_correction",
        "mae": mae_h3, "wmape": wm_h3, "bias": bias_h3,
        "delta": mae_h3 - BASELINE_MAE,
    })
    save_predictions(X_test, y_test, pred_h3, "reports/exp_h3_predictions.csv")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  ITOGI EKSPERIMENTA H")
    print("=" * 65)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    print(f"\n  {'Strategy':<28} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'Delta':>8}")
    print(f"  {'-' * 56}")
    print(f"  {'baseline':<28} {bl_mae:>8.4f} {'':>8} {'':>8} {'0.0000':>8}")

    for _, row in results_df.iterrows():
        marker = " <--" if row["delta"] < 0 else ""
        print(f"  {row['strategy']:<28} {row['mae']:>8.4f} {row['wmape']:>7.2f}% "
              f"{row['bias']:>+8.4f} {row['delta']:>+8.4f}{marker}")

    best = results_df.iloc[0]
    print(f"\n  Luchshaya strategiya: {best['strategy']}")
    print(f"  MAE = {best['mae']:.4f} (delta vs baseline: {best['delta']:+.4f})")

    # Save summary (best row)
    summary = pd.DataFrame([{
        "experiment": f"H_{best['strategy']}",
        "mae": best["mae"],
        "wmape": best["wmape"],
        "bias": best["bias"],
        "baseline_mae": BASELINE_MAE,
        "delta": best["delta"],
    }])
    summary.to_csv("reports/exp_h_summary.csv", index=False)

    # Save full results
    results_df.to_csv("reports/exp_h_all_results.csv", index=False, encoding="utf-8-sig")
    print(f"  Polnye rezul'taty: reports/exp_h_all_results.csv")

    print("\nGotovo!")


if __name__ == "__main__":
    main()
