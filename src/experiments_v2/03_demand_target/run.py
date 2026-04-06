"""
Experiment 03 v2: Demand Target -- fair comparison.

Two models, same data, same features, evaluated on the SAME metric:
  A) Baseline: train on Prodano, evaluate on Spros (demand)
  B) Demand:   train on Spros,   evaluate on Spros (demand)

If B beats A, then training on demand gives better demand predictions.
We also report metrics vs Prodano for reference.

Input:  data/processed/daily_sales_8m_demand.csv
Output: src/experiments_v2/03_demand_target/metrics.json
        src/experiments_v2/03_demand_target/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/03_demand_target/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
DEMAND_CSV = Path(ROOT) / "data" / "processed" / "daily_sales_8m_demand.csv"

DEMAND_TARGET = "Спрос"

# Demand features (lags/rolling on demand)
DEMAND_FEATURES = [
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
    "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30",
    "demand_roll_std7",
]
ALL_FEATURES = FEATURES_V2 + DEMAND_FEATURES


def main():
    print("=" * 60)
    print("  EXPERIMENT 03 v2: Demand Target (fair comparison)")
    print("=" * 60)
    t_start = time.time()

    # --- Load ---
    print(f"\n[1/6] Loading data from {DEMAND_CSV}...")
    if not DEMAND_CSV.exists():
        print(f"  ERROR: {DEMAND_CSV} not found!")
        print("  Run build_demand_profiles.py first.")
        return
    df = pd.read_csv(str(DEMAND_CSV), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    print(f"  Days: {df['Дата'].nunique()}, Bakeries: {df['Пекарня'].nunique()}, "
          f"Products: {df['Номенклатура'].nunique()}")

    mean_sold = df[TARGET].mean()
    mean_demand = df[DEMAND_TARGET].mean()
    uplift_pct = (mean_demand - mean_sold) / mean_sold * 100
    censored_pct = 100 * df["is_censored"].mean()
    print(f"\n  Demand stats:")
    print(f"    mean({TARGET}):  {mean_sold:.4f}")
    print(f"    mean({DEMAND_TARGET}):   {mean_demand:.4f} ({uplift_pct:+.1f}%)")
    print(f"    censored: {censored_pct:.1f}%")

    # --- Features ---
    print(f"\n[2/6] Selecting features...")
    # Model A (baseline on sales): only FEATURES_V2 (sales lags)
    features_a = [f for f in FEATURES_V2 if f in df.columns]
    # Model B (demand): FEATURES_V2 + demand lags
    features_b = [f for f in ALL_FEATURES if f in df.columns]
    print(f"  Model A (sales):  {len(features_a)} features")
    print(f"  Model B (demand): {len(features_b)} features")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Split ---
    print(f"\n[3/6] Train/test split (last {TEST_DAYS} days)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    y_test_sold = test[TARGET].values
    y_test_demand = test[DEMAND_TARGET].values

    # --- Model A: train on Prodano ---
    print(f"\n[4/6] Model A: train on '{TARGET}' (sales baseline)...")
    t_train = time.time()
    model_a = train_lgbm(train[features_a], train[TARGET])
    time_a = time.time() - t_train
    print(f"  Training time: {time_a:.0f}s")

    pred_a = predict_clipped(model_a, test[features_a])

    # --- Model B: train on Spros ---
    print(f"\n[5/6] Model B: train on '{DEMAND_TARGET}' (demand target)...")
    t_train = time.time()
    model_b = train_lgbm(train[features_b], train[DEMAND_TARGET])
    time_b = time.time() - t_train
    print(f"  Training time: {time_b:.0f}s")

    pred_b = predict_clipped(model_b, test[features_b])

    # --- Evaluation ---
    print(f"\n[6/6] Evaluation...")

    # Both evaluated vs DEMAND (fair comparison)
    mae_a_demand = mean_absolute_error(y_test_demand, pred_a)
    wm_a_demand = wmape(y_test_demand, pred_a)
    bias_a_demand = np.mean(y_test_demand - pred_a)

    mae_b_demand = mean_absolute_error(y_test_demand, pred_b)
    wm_b_demand = wmape(y_test_demand, pred_b)
    bias_b_demand = np.mean(y_test_demand - pred_b)

    print(f"\n  {'='*60}")
    print(f"  FAIR COMPARISON: both evaluated vs {DEMAND_TARGET}")
    print(f"  {'='*60}")
    print(f"  {'Metric':<12} {'A (sales)':<15} {'B (demand)':<15} {'Delta':>10}")
    print(f"  {'-'*52}")
    print(f"  {'MAE':<12} {mae_a_demand:<15.4f} {mae_b_demand:<15.4f} {mae_b_demand - mae_a_demand:>+10.4f}")
    print(f"  {'WMAPE':<12} {wm_a_demand:<14.2f}% {wm_b_demand:<14.2f}% {wm_b_demand - wm_a_demand:>+10.2f}%")
    print(f"  {'Bias':<12} {bias_a_demand:<+15.4f} {bias_b_demand:<+15.4f} {bias_b_demand - bias_a_demand:>+10.4f}")

    winner = "B (demand)" if mae_b_demand < mae_a_demand else "A (sales)"
    print(f"\n  Winner: {winner}")

    # Also report vs Prodano (for reference / comparison with exp 01)
    mae_a_sold = mean_absolute_error(y_test_sold, pred_a)
    mae_b_sold = mean_absolute_error(y_test_sold, pred_b)
    wm_a_sold = wmape(y_test_sold, pred_a)
    wm_b_sold = wmape(y_test_sold, pred_b)

    print(f"\n  Reference: evaluated vs {TARGET} (sales)")
    print(f"  {'Metric':<12} {'A (sales)':<15} {'B (demand)':<15}")
    print(f"  {'-'*42}")
    print(f"  {'MAE':<12} {mae_a_sold:<15.4f} {mae_b_sold:<15.4f}")
    print(f"  {'WMAPE':<12} {wm_a_sold:<14.2f}% {wm_b_sold:<14.2f}%")

    # High demand analysis (vs demand target)
    print(f"\n  High demand analysis (vs {DEMAND_TARGET}):")
    for threshold in [15, 50, 100]:
        mask = y_test_demand >= threshold
        if mask.sum() > 0:
            ma_a = mean_absolute_error(y_test_demand[mask], pred_a[mask])
            ma_b = mean_absolute_error(y_test_demand[mask], pred_b[mask])
            bi_a = np.mean(y_test_demand[mask] - pred_a[mask])
            bi_b = np.mean(y_test_demand[mask] - pred_b[mask])
            print(f"    >= {threshold}: N={mask.sum()}, "
                  f"A MAE={ma_a:.2f} Bias={bi_a:+.2f} | "
                  f"B MAE={ma_b:.2f} Bias={bi_b:+.2f}")

    # Per-category (Model B vs demand)
    print(f"\n  === Per-category metrics (Model B vs {DEMAND_TARGET}) ===")
    print_category_metrics(y_test_demand, pred_b, test["Категория"].values)

    # Feature importance (Model B)
    importance_b = pd.DataFrame({
        "feature": features_b, "importance": model_b.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 15 features (Model B):")
    for _, row in importance_b.head(15).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>8.0f}")

    demand_imp = importance_b[importance_b["feature"].str.startswith("demand_")]
    sales_imp = importance_b[importance_b["feature"].str.startswith("sales_")]
    print(f"\n  Demand features total: {demand_imp['importance'].sum():.0f}")
    print(f"  Sales features total:  {sales_imp['importance'].sum():.0f}")

    # --- Save ---
    rmse_b = np.sqrt(mean_squared_error(y_test_demand, pred_b))
    r2_b = r2_score(y_test_demand, pred_b)

    metrics = {
        "experiment": "03_demand_target_v2",
        "comparison": "both evaluated vs Spros (demand)",
        # Fair comparison vs demand
        "mae_A_vs_demand": round(mae_a_demand, 4),
        "mae_B_vs_demand": round(mae_b_demand, 4),
        "wmape_A_vs_demand": round(wm_a_demand, 2),
        "wmape_B_vs_demand": round(wm_b_demand, 2),
        "bias_A_vs_demand": round(bias_a_demand, 4),
        "bias_B_vs_demand": round(bias_b_demand, 4),
        "winner": winner,
        "mae_improvement": round(mae_a_demand - mae_b_demand, 4),
        # Reference vs sales
        "mae_A_vs_sold": round(mae_a_sold, 4),
        "mae_B_vs_sold": round(mae_b_sold, 4),
        # Best model (B) full metrics vs demand
        "mae": round(mae_b_demand, 4),
        "wmape": round(wm_b_demand, 2),
        "rmse": round(rmse_b, 4),
        "bias": round(bias_b_demand, 4),
        "r2": round(r2_b, 4),
        # Dataset stats
        "demand_uplift_pct": round(uplift_pct, 2),
        "censored_pct": round(censored_pct, 1),
        "mean_sold": round(mean_sold, 4),
        "mean_demand": round(mean_demand, 4),
        "train_rows": len(train), "test_rows": len(test),
        "train_days": int(train["Дата"].nunique()),
        "test_days": int(test["Дата"].nunique()),
        "n_features_A": len(features_a),
        "n_features_B": len(features_b),
        "n_bakeries": int(df["Пекарня"].nunique()),
        "n_products": int(df["Номенклатура"].nunique()),
        "n_categories": int(df["Категория"].nunique()),
        "train_time_A_s": round(time_a, 1),
        "train_time_B_s": round(time_b, 1),
        "feature_importance_top10": importance_b.head(10)[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_sold": y_test_sold,
        "fact_demand": y_test_demand,
        "pred_A_sales": np.round(pred_a, 2),
        "pred_B_demand": np.round(pred_b, 2),
        "is_censored": test["is_censored"].values,
        "lost_qty": test["lost_qty"].values,
        "error_A_vs_demand": np.round(np.abs(y_test_demand - pred_a), 2),
        "error_B_vs_demand": np.round(np.abs(y_test_demand - pred_b), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
