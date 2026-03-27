"""
Experiment G: Noise Filtering
G1: Remove false zeros (Prodano==0 AND Vypusk==0) from train
G2: Robust rolling stats (median instead of mean for 3/7-day windows)
G3: G1 + G2 combined
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from src.experiments.common import (
    load_data, FEATURES, CATEGORICAL_COLS, TARGET, MODEL_PARAMS, BASELINE_MAE,
    DATA_PATH,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def load_data_with_median_rolling():
    """Load data and recompute rolling features using median for 3/7-day windows."""
    print(f"Zagruzka {DATA_PATH} (s pereschyotom rolling median)...")
    df = pd.read_csv(DATA_PATH)
    df["Дата"] = pd.to_datetime(df["Дата"])

    # Recompute rolling median features
    grouped = df.groupby(["Пекарня", "Номенклатура"])
    df["sales_roll_median3"] = grouped["Продано"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).median()
    )
    df["sales_roll_median7"] = grouped["Продано"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).median()
    )

    # Replace mean3/mean7 with median versions in feature list
    features_median = []
    for f in FEATURES:
        if f == "sales_roll_mean3":
            features_median.append("sales_roll_median3")
        elif f == "sales_roll_mean7":
            features_median.append("sales_roll_median7")
        else:
            features_median.append(f)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    available = [f for f in features_median if f in df.columns]

    # Split
    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} strok ({train['Дата'].nunique()} dnej)")
    print(f"  Test:  {len(test):,} strok")
    print(f"  Priznakov: {len(available)}")

    X_train = train[available]
    y_train = train[TARGET]
    X_test = test[available]
    y_test = test[TARGET]

    return df, X_train, y_train, X_test, y_test, available


def main():
    print("=" * 65)
    print("  EXPERIMENT G: NOISE FILTERING")
    print("=" * 65)

    # Load standard data for G1 and baseline
    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline ---
    print("\n--- Baseline (no filtering) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    results = []

    # ========================================
    # G1: Remove false zeros
    # ========================================
    print("\n--- G1: Udalenie lozhnykh nulej (Prodano==0 AND Vypusk==0) ---")

    # Get the full df to access Vypusk column
    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train_full = df[df["Дата"] < test_start].copy()

    false_zero_mask = (train_full["Продано"] == 0) & (train_full["Выпуск"] == 0)
    n_false_zeros = false_zero_mask.sum()
    print(f"  Lozhnye nuli v traine: {n_false_zeros:,} ({n_false_zeros / len(train_full) * 100:.2f}%)")

    # Filter train
    train_filtered = train_full[~false_zero_mask]
    X_train_g1 = train_filtered[features]
    y_train_g1 = train_filtered[TARGET]
    print(f"  Train posle fil'tratsii: {len(X_train_g1):,} strok")

    model_g1 = train_lgbm(X_train_g1, y_train_g1)
    pred_g1 = predict_clipped(model_g1, X_test)
    mae_g1, wm_g1, bias_g1 = print_metrics("G1_false_zeros", y_test, pred_g1)
    print_category_metrics(y_test, pred_g1, X_test["Категория"])

    results.append({
        "strategy": "G1_remove_false_zeros",
        "mae": mae_g1, "wmape": wm_g1, "bias": bias_g1,
        "delta": mae_g1 - BASELINE_MAE,
    })
    save_predictions(X_test, y_test, pred_g1, "reports/exp_g1_predictions.csv")

    # ========================================
    # G2: Robust rolling stats (median)
    # ========================================
    print("\n--- G2: Robust rolling stats (median vmesto mean dlya 3/7 dnej) ---")

    df_m, X_train_g2, y_train_g2, X_test_g2, y_test_g2, features_g2 = load_data_with_median_rolling()

    model_g2 = train_lgbm(X_train_g2, y_train_g2)
    pred_g2 = predict_clipped(model_g2, X_test_g2)
    mae_g2, wm_g2, bias_g2 = print_metrics("G2_robust_rolling", y_test_g2, pred_g2)

    # For category metrics, need to get categories from df_m
    test_start_m = df_m["Дата"].max() - pd.Timedelta(days=2)
    test_m = df_m[df_m["Дата"] >= test_start_m]
    print_category_metrics(y_test_g2, pred_g2, test_m["Категория"])

    results.append({
        "strategy": "G2_robust_rolling",
        "mae": mae_g2, "wmape": wm_g2, "bias": bias_g2,
        "delta": mae_g2 - BASELINE_MAE,
    })
    save_predictions(X_test_g2, y_test_g2, pred_g2, "reports/exp_g2_predictions.csv")

    # ========================================
    # G3: G1 + G2 combined
    # ========================================
    print("\n--- G3: G1 + G2 combined (false zeros removed + median rolling) ---")

    train_full_m = df_m[df_m["Дата"] < test_start_m].copy()
    false_zero_mask_m = (train_full_m["Продано"] == 0) & (train_full_m["Выпуск"] == 0)
    train_filtered_m = train_full_m[~false_zero_mask_m]

    X_train_g3 = train_filtered_m[features_g2]
    y_train_g3 = train_filtered_m[TARGET]
    print(f"  Train posle vsekh filtrov: {len(X_train_g3):,} strok")

    model_g3 = train_lgbm(X_train_g3, y_train_g3)
    pred_g3 = predict_clipped(model_g3, X_test_g2)
    mae_g3, wm_g3, bias_g3 = print_metrics("G3_combined", y_test_g2, pred_g3)
    print_category_metrics(y_test_g2, pred_g3, test_m["Категория"])

    results.append({
        "strategy": "G3_combined",
        "mae": mae_g3, "wmape": wm_g3, "bias": bias_g3,
        "delta": mae_g3 - BASELINE_MAE,
    })
    save_predictions(X_test_g2, y_test_g2, pred_g3, "reports/exp_g3_predictions.csv")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  ITOGI EKSPERIMENTA G")
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
        "experiment": f"G_{best['strategy']}",
        "mae": best["mae"],
        "wmape": best["wmape"],
        "bias": best["bias"],
        "baseline_mae": BASELINE_MAE,
        "delta": best["delta"],
    }])
    summary.to_csv("reports/exp_g_summary.csv", index=False)

    # Save full results
    results_df.to_csv("reports/exp_g_all_results.csv", index=False, encoding="utf-8-sig")
    print(f"  Polnye rezul'taty: reports/exp_g_all_results.csv")

    print("\nGotovo!")


if __name__ == "__main__":
    main()
