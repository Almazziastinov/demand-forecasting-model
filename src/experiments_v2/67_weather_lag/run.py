"""
Experiment 67: Weather-lag features to break lag contamination.

Hypothesis (confirmed on bakery "Sibirsky Trakt 25"):
  Bad weather on D0 -> actual sales drop -> lag1 carries low value into D+1
  -> model under-forecasts D+1 even if D+1 weather is normal.
  The model has no signal that lag1 was depressed by weather, not by real demand drop.

Fix: add weather-lag features so the model can discount a low lag1 when
yesterday was bad weather but today is normal:
  - is_bad_weather_lag1: was yesterday bad weather? (binary)
  - precipitation_lag1: yesterday's precipitation mm
  - temp_mean_lag1: yesterday's temp (captures cold-weather depressed demand)
  - bad_weather_x_lag1: interaction term (bad_weather_lag1 * demand_lag1)
  - bad_weather_x_lag1_demand: same for demand_lag1

Baseline: exp 60 (V3, P50), MAE=2.8816
Best so far: exp 63 (combined), MAE=2.8540

Input:  data/processed/daily_sales_8m_demand.csv
Output: src/experiments_v2/67_weather_lag/metrics.json
        src/experiments_v2/67_weather_lag/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/67_weather_lag/run.py
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

from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, print_metrics, print_category_metrics,
    train_quantile, train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent

BASELINE_V3_MAE  = 2.8816  # exp 60
BEST_MAE         = 2.8540  # exp 63

# New weather-lag feature names
WEATHER_LAG_FEATURES = [
    "is_bad_weather_lag1",
    "precipitation_lag1",
    "temp_mean_lag1",
    "bad_x_demand_lag1",   # interaction: is_bad_weather_lag1 * demand_lag1
]

FEATURES_V3_WLAG = FEATURES_V3 + WEATHER_LAG_FEATURES
QUANTILES = [0.25, 0.50, 0.75]


def build_weather_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged weather features shifted by 1 day per bakery group."""
    df = df.copy()
    df["Дата"] = pd.to_datetime(df["Дата"])
    df = df.sort_values(["Пекарня", "Дата"])

    grp = df.groupby("Пекарня")
    df["is_bad_weather_lag1"] = grp["is_bad_weather"].shift(1).fillna(0).astype(int)
    df["precipitation_lag1"]  = grp["precipitation"].shift(1).fillna(0)
    df["temp_mean_lag1"]       = grp["temp_mean"].shift(1).fillna(df["temp_mean"].mean())

    # Interaction: bad weather lag1 × demand_lag1
    # Tells model: "lag1 was low AND yesterday was bad weather"
    df["bad_x_demand_lag1"] = df["is_bad_weather_lag1"] * df["demand_lag1"].fillna(0)

    return df


def main():
    print("=" * 60)
    print("  EXPERIMENT 67: Weather-lag features")
    print("  Hypothesis: lag1 contaminated by bad-weather depression")
    print("  Target: Spros (demand), Objective: Quantile P50")
    print("=" * 60)
    t_start = time.time()

    # --- Load ---
    print(f"\n[1/5] Loading data...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")

    # --- Build weather-lag features ---
    print(f"\n[2/5] Building weather-lag features...")
    df = build_weather_lag_features(df)

    # Show distribution of new features
    bad_lag1_rate = df["is_bad_weather_lag1"].mean()
    print(f"  is_bad_weather_lag1 = 1: {bad_lag1_rate*100:.1f}% of rows")
    print(f"  precipitation_lag1 mean: {df['precipitation_lag1'].mean():.2f} mm")
    print(f"  bad_x_demand_lag1  mean: {df['bad_x_demand_lag1'].mean():.3f}")

    # Verify the Sibirsky Trakt 25 case is in the data
    target_col = DEMAND_TARGET
    available = [f for f in FEATURES_V3_WLAG if f in df.columns]
    missing = [f for f in FEATURES_V3_WLAG if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available)} features (V3={len(FEATURES_V3)} + weather_lag={len(WEATHER_LAG_FEATURES)})")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Split ---
    print(f"\n[3/5] Train/test split...")
    from src.config import TEST_DAYS
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test  = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    X_train = train[available]
    y_train = train[target_col]
    X_test  = test[available]
    y_test  = test[target_col].values
    y_sold  = test["Продано"].values

    # --- Train ---
    print(f"\n[4/5] Training models...")
    models, preds = {}, {}
    total_t = 0
    for q in QUANTILES:
        qn = f"P{int(q*100)}"
        t0 = time.time()
        models[qn] = train_quantile(X_train, y_train, alpha=q)
        tt = time.time() - t0
        total_t += tt
        preds[qn] = predict_clipped(models[qn], X_test)
        mae = mean_absolute_error(y_test, preds[qn])
        print(f"  {qn}: MAE={mae:.4f}, Time={tt:.0f}s")

    # MSE baseline (same features) for fair comparison
    t0 = time.time()
    model_mse = train_lgbm(X_train, y_train)
    time_mse = time.time() - t0
    pred_mse = predict_clipped(model_mse, X_test)
    mae_mse = mean_absolute_error(y_test, pred_mse)
    print(f"  MSE (same feats): MAE={mae_mse:.4f}, Time={time_mse:.0f}s")

    # --- Eval ---
    print(f"\n[5/5] Evaluation...")
    y_p50 = preds["P50"]
    mae   = mean_absolute_error(y_test, y_p50)
    wm    = wmape(y_test, y_p50)
    rmse  = np.sqrt(mean_squared_error(y_test, y_p50))
    bias  = np.mean(y_test - y_p50)
    r2    = r2_score(y_test, y_p50)

    print(f"\n  === EXP 67 RESULTS (P50 vs {target_col}) ===")
    print(f"    MAE   = {mae:.4f}")
    print(f"    WMAPE = {wm:.2f}%")
    print(f"    RMSE  = {rmse:.4f}")
    print(f"    Bias  = {bias:+.4f}")
    print(f"    R2    = {r2:.4f}")

    print(f"\n  === vs baselines ===")
    print(f"    {'Exp':<35} {'MAE':>8} {'Delta':>10}")
    print(f"    {'-'*55}")
    print(f"    {'exp 60 (V3, P50)':<35} {BASELINE_V3_MAE:>8.4f} {mae-BASELINE_V3_MAE:>+10.4f}")
    print(f"    {'exp 63 (combined, best)':<35} {BEST_MAE:>8.4f} {mae-BEST_MAE:>+10.4f}")
    print(f"    {'>>> exp 67 (weather-lag, P50) <<<':<35} {mae:>8.4f} {'NEW':>10}")

    # Feature importance
    imp = pd.DataFrame({
        "feature":    available,
        "importance": models["P50"].feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  === Feature importance — new weather-lag features ===")
    new_feats = imp[imp["feature"].isin(WEATHER_LAG_FEATURES)]
    total_imp = imp["importance"].sum()
    for _, r in new_feats.iterrows():
        rank = imp.index.get_loc(r.name) + 1
        pct  = r["importance"] / total_imp * 100
        print(f"    #{rank:3d}  {r['feature']:<30} {r['importance']:>10.0f}  ({pct:.2f}%)")

    print(f"\n  === Top 15 features overall ===")
    for i, (_, r) in enumerate(imp.head(15).iterrows()):
        pct = r["importance"] / total_imp * 100
        new = " *NEW*" if r["feature"] in WEATHER_LAG_FEATURES else ""
        print(f"    #{i+1:2d}  {r['feature']:<30} {pct:>6.2f}%{new}")

    # Bad-weather lag day breakdown
    print(f"\n  === Bad-weather lag effect (days after is_bad_weather_lag1=1 vs 0) ===")
    test_df = test.copy()
    test_df["pred_P50"] = y_p50
    test_df["abs_err"]  = np.abs(y_test - y_p50)
    test_df["bias"]     = y_p50 - y_test

    for lag_val, label in [(0, "normal yesterday"), (1, "bad weather yesterday")]:
        mask = test_df["is_bad_weather_lag1"] == lag_val
        sub = test_df[mask]
        if len(sub) == 0:
            continue
        mae_sub  = sub["abs_err"].mean()
        bias_sub = sub["bias"].mean()
        print(f"  {label:25s} N={mask.sum():5d}  MAE={mae_sub:.4f}  Bias={bias_sub:+.4f}")

    # Per-category
    print(f"\n  === Per-category (P50 vs {target_col}) ===")
    print_category_metrics(y_test, y_p50, test["Категория"].values)

    # --- Save ---
    p25, p75 = preds["P25"], preds["P75"]
    width    = p75 - p25
    coverage = np.mean((y_test >= p25) & (y_test <= p75))

    metrics = {
        "experiment":   "67_weather_lag",
        "target":       target_col,
        "objective":    "quantile P50",
        "features":     f"FEATURES_V3 + weather_lag ({len(available)} total)",
        "mae":          round(mae, 4),
        "wmape":        round(wm, 2),
        "rmse":         round(rmse, 4),
        "bias":         round(bias, 4),
        "r2":           round(r2, 4),
        "mae_P25":      round(mean_absolute_error(y_test, p25), 4),
        "mae_P50":      round(mae, 4),
        "mae_P75":      round(mean_absolute_error(y_test, p75), 4),
        "mae_mse_same_features": round(mae_mse, 4),
        "interval_mean_width": round(width.mean(), 4),
        "coverage_pct": round(coverage * 100, 2),
        "mae_vs_prodano": round(mean_absolute_error(y_sold, y_p50), 4),
        "delta_vs_v3":  round(mae - BASELINE_V3_MAE, 4),
        "delta_vs_best": round(mae - BEST_MAE, 4),
        "new_features": WEATHER_LAG_FEATURES,
        "bad_lag1_rate_pct": round(bad_lag1_rate * 100, 1),
        "train_rows":   len(train),
        "test_rows":    len(test),
        "n_features":   len(available),
        "train_time_s": round(total_t, 1),
        "feature_importance_top15": imp.head(15)[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата":              test["Дата"].values,
        "Пекарня":           test["Пекарня"].values,
        "Номенклатура":      test["Номенклатура"].values,
        "Категория":         test["Категория"].values,
        "fact_demand":       y_test,
        "fact_sold":         y_sold,
        "pred_P50":          np.round(y_p50, 2),
        "pred_MSE":          np.round(pred_mse, 2),
        "is_bad_weather_lag1": test["is_bad_weather_lag1"].values,
        "abs_error_P50":     np.round(np.abs(y_test - y_p50), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    print(f"\n  Total time: {time.time()-t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
