"""
Experiment 60: Baseline V3 (new reference baseline).

Combines best findings from all previous experiments:
  - Target: Spros (demand) — from exp 03
  - Features: FEATURES_V3 (FEATURES_V2 + demand lags + price features) — from exp 09
  - Objective: Quantile P50 (median) — from exp 53
  - Also trains P25/P75 for prediction intervals

This is the new baseline for all future experiments.

Input:  data/processed/daily_sales_8m_demand.csv (with price features)
Output: src/experiments_v2/60_baseline_v3/metrics.json
        src/experiments_v2/60_baseline_v3/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/60_baseline_v3/run.py
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
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, print_metrics, print_category_metrics,
    train_quantile, train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent

# Previous baselines for comparison
BASELINE_V1_MAE = 2.2904   # exp 01: sales target, FEATURES_V2, MSE loss
BASELINE_V2_MAE = 2.9923   # exp 03 Model B: demand target, FEATURES_V2 + demand lags, MSE loss
BEST_DEMAND_MAE = 2.9364   # exp 09: demand target + price features, MSE loss

QUANTILES = [0.25, 0.50, 0.75]


def main():
    print("=" * 60)
    print("  EXPERIMENT 60: Baseline V3")
    print("  Target: Spros (demand)")
    print("  Features: FEATURES_V3 (V2 + demand lags + price)")
    print("  Objective: Quantile P50 (median)")
    print("=" * 60)
    t_start = time.time()

    # --- Load ---
    print(f"\n[1/5] Loading data from {DEMAND_8M_PATH.name}...")
    if not DEMAND_8M_PATH.exists():
        print(f"  ERROR: {DEMAND_8M_PATH} not found!")
        print("  Run merge_prices.py first.")
        return
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    print(f"  Days: {df['Дата'].nunique()}, Bakeries: {df['Пекарня'].nunique()}, "
          f"Products: {df['Номенклатура'].nunique()}, Categories: {df['Категория'].nunique()}")

    mean_demand = df[DEMAND_TARGET].mean()
    mean_sold = df[TARGET].mean()
    censored_pct = 100 * df["is_censored"].mean()
    print(f"\n  mean({TARGET}): {mean_sold:.4f}")
    print(f"  mean({DEMAND_TARGET}): {mean_demand:.4f} ({(mean_demand - mean_sold) / mean_sold * 100:+.1f}%)")
    print(f"  censored: {censored_pct:.1f}%")

    # --- Features ---
    print(f"\n[2/5] Selecting features...")
    available = [f for f in FEATURES_V3 if f in df.columns]
    missing = [f for f in FEATURES_V3 if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available)} of {len(FEATURES_V3)} features")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Split ---
    print(f"\n[3/5] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    X_train = train[available]
    y_train = train[DEMAND_TARGET]
    X_test = test[available]
    y_test_demand = test[DEMAND_TARGET].values
    y_test_sold = test[TARGET].values

    # --- Train quantile models ---
    print(f"\n[4/5] Training Quantile models (P25/P50/P75)...")
    models = {}
    preds = {}
    total_train_time = 0

    for q in QUANTILES:
        q_name = f"P{int(q * 100)}"
        print(f"\n  --- {q_name} (alpha={q}) ---")
        t_q = time.time()
        models[q_name] = train_quantile(X_train, y_train, alpha=q)
        tt = time.time() - t_q
        total_train_time += tt

        preds[q_name] = predict_clipped(models[q_name], X_test)
        mae = mean_absolute_error(y_test_demand, preds[q_name])
        wm = wmape(y_test_demand, preds[q_name])
        bias = np.mean(y_test_demand - preds[q_name])
        print(f"    MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}, Time={tt:.0f}s")

    # --- Also train MSE baseline for comparison ---
    print(f"\n  --- MSE baseline (for reference) ---")
    t_mse = time.time()
    model_mse = train_lgbm(X_train, y_train)
    time_mse = time.time() - t_mse
    pred_mse = predict_clipped(model_mse, X_test)
    mae_mse = mean_absolute_error(y_test_demand, pred_mse)
    wm_mse = wmape(y_test_demand, pred_mse)
    bias_mse = np.mean(y_test_demand - pred_mse)
    print(f"    MAE={mae_mse:.4f}, WMAPE={wm_mse:.2f}%, Bias={bias_mse:+.4f}, Time={time_mse:.0f}s")

    # --- Evaluation ---
    print(f"\n[5/5] Evaluation...")

    # Primary: P50
    y_p50 = preds["P50"]
    mae = mean_absolute_error(y_test_demand, y_p50)
    wm = wmape(y_test_demand, y_p50)
    rmse = np.sqrt(mean_squared_error(y_test_demand, y_p50))
    bias = np.mean(y_test_demand - y_p50)
    r2 = r2_score(y_test_demand, y_p50)

    print(f"\n  === BASELINE V3 RESULTS (P50 vs {DEMAND_TARGET}) ===")
    print(f"    MAE   = {mae:.4f}")
    print(f"    WMAPE = {wm:.2f}%")
    print(f"    RMSE  = {rmse:.4f}")
    print(f"    Bias  = {bias:+.4f}")
    print(f"    R2    = {r2:.4f}")

    # Comparison with previous baselines
    print(f"\n  === vs previous baselines ===")
    print(f"    {'Baseline':<35} {'MAE':>8} {'Delta':>10}")
    print(f"    {'-' * 55}")
    print(f"    {'v1 (exp 01, sales/MSE)':<35} {BASELINE_V1_MAE:>8.4f} {'(vs Продано)':>10}")
    print(f"    {'v2 (exp 03, demand/MSE)':<35} {BASELINE_V2_MAE:>8.4f} {mae - BASELINE_V2_MAE:>+10.4f}")
    print(f"    {'exp 09 (demand+price/MSE)':<35} {BEST_DEMAND_MAE:>8.4f} {mae - BEST_DEMAND_MAE:>+10.4f}")
    print(f"    {'exp 53 (demand/Quantile P50)':<35} {2.9437:>8.4f} {mae - 2.9437:>+10.4f}")
    print(f"    {'MSE same features (this run)':<35} {mae_mse:>8.4f} {mae - mae_mse:>+10.4f}")
    print(f"    {'>>> V3 (demand+price/P50) <<<':<35} {mae:>8.4f} {'NEW':>10}")

    # Interval analysis
    p25 = preds["P25"]
    p75 = preds["P75"]
    width = p75 - p25
    coverage = np.mean((y_test_demand >= p25) & (y_test_demand <= p75))

    print(f"\n  === Prediction intervals ===")
    print(f"    Mean width (P75-P25): {width.mean():.2f}")
    print(f"    Coverage [P25,P75]:   {coverage * 100:.1f}% (expected ~50%)")

    # High demand
    print(f"\n  === High demand analysis ===")
    print(f"    {'Demand':<12} {'N':>6} {'MAE_P50':>8} {'MAE_MSE':>8} {'Bias':>8} {'Width':>8}")
    print(f"    {'-' * 54}")
    for threshold in [15, 50, 100]:
        mask = y_test_demand >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_test_demand[mask], y_p50[mask])
            mae_h_mse = mean_absolute_error(y_test_demand[mask], pred_mse[mask])
            bias_h = np.mean(y_test_demand[mask] - y_p50[mask])
            width_h = width[mask].mean()
            print(f"    >= {threshold:<6} {mask.sum():>6} {mae_h:>8.2f} {mae_h_mse:>8.2f} "
                  f"{bias_h:>+8.2f} {width_h:>8.1f}")

    # Per-category
    print(f"\n  === Per-category metrics (P50 vs {DEMAND_TARGET}) ===")
    print_category_metrics(y_test_demand, y_p50, test["Категория"].values)

    # Feature importance (P50 model)
    print(f"\n  === Feature importance (P50 model, top 20) ===")
    importance = pd.DataFrame({
        "feature": available,
        "importance": models["P50"].feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(20).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>8.0f}")

    # Also report vs Prodano (for P&L / business comparison)
    mae_vs_sold = mean_absolute_error(y_test_sold, y_p50)
    wm_vs_sold = wmape(y_test_sold, y_p50)
    print(f"\n  === Reference: P50 vs {TARGET} (for business comparison) ===")
    print(f"    MAE = {mae_vs_sold:.4f}, WMAPE = {wm_vs_sold:.2f}%")

    # --- Save ---
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "60_baseline_v3",
        "target": DEMAND_TARGET,
        "objective": "quantile P50",
        "features": "FEATURES_V3 (V2 + demand lags + price)",
        "mae": round(mae, 4),
        "wmape": round(wm, 2),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "mae_P25": round(mean_absolute_error(y_test_demand, preds["P25"]), 4),
        "mae_P50": round(mae, 4),
        "mae_P75": round(mean_absolute_error(y_test_demand, preds["P75"]), 4),
        "mae_mse_same_features": round(mae_mse, 4),
        "interval_mean_width": round(width.mean(), 4),
        "coverage_pct": round(coverage * 100, 2),
        "mae_vs_prodano": round(mae_vs_sold, 4),
        "wmape_vs_prodano": round(wm_vs_sold, 2),
        "previous_baselines": {
            "v1_exp01_sales_mse": BASELINE_V1_MAE,
            "v2_exp03_demand_mse": BASELINE_V2_MAE,
            "exp09_demand_price_mse": BEST_DEMAND_MAE,
            "exp53_demand_quantile": 2.9437,
        },
        "train_rows": len(train),
        "test_rows": len(test),
        "train_days": int(train["Дата"].nunique()),
        "test_days": int(test["Дата"].nunique()),
        "n_features": len(available),
        "n_bakeries": int(df["Пекарня"].nunique()),
        "n_products": int(df["Номенклатура"].nunique()),
        "n_categories": int(df["Категория"].nunique()),
        "censored_pct": round(censored_pct, 1),
        "train_time_quantiles_s": round(total_train_time, 1),
        "train_time_mse_s": round(time_mse, 1),
        "feature_importance_top15": importance.head(15)[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_demand": y_test_demand,
        "fact_sold": y_test_sold,
        "pred_P25": np.round(preds["P25"], 2),
        "pred_P50": np.round(preds["P50"], 2),
        "pred_P75": np.round(preds["P75"], 2),
        "pred_MSE": np.round(pred_mse, 2),
        "is_censored": test["is_censored"].values,
        "abs_error_P50": np.round(np.abs(y_test_demand - y_p50), 2),
        "abs_error_MSE": np.round(np.abs(y_test_demand - pred_mse), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
