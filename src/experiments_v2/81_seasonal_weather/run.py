"""
Experiment 81: Seasonal weather features on new stg dataset (Jun 2025 - Jun 2026).

Hypothesis: training data now covers full summer 2025 + winter 2025-2026,
so the model can learn the seasonal asymmetry:
  - Rain in summer (Apr-Sep) -> demand drops
  - Snow/cold in winter (Oct-Mar) -> demand neutral or slightly up

New features added to FEATURES_STG (base without price/demand lags):
  - is_warm_season     : month in [4..9]
  - precip_warm        : precipitation * is_warm_season
  - precip_cold        : precipitation * (1 - is_warm_season)
  - bad_weather_warm   : is_bad_weather * is_warm_season
  - bad_weather_cold   : is_bad_weather * (1 - is_warm_season)

Variants:
  A: baseline_stg       - FEATURES_STG, no seasonal split, target=Продано
  B: seasonal_split     - FEATURES_STG + seasonal features
  C: seasonal_lag1      - B + is_bad_weather_lag1 * is_warm_season interaction

Input:  data/processed/daily_sales_stg.csv
Output: src/experiments_v2/81_seasonal_weather/metrics.json
        src/experiments_v2/81_seasonal_weather/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/81_seasonal_weather/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

EXP_DIR = Path(__file__).resolve().parent

TARGET = "Продано"
TEST_DAYS = 7

FEATURES_STG = [
    "Пекарня", "Номенклатура", "Категория", "Город",
    "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
    "sales_lag14", "sales_lag30",
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    "is_holiday", "is_pre_holiday", "is_post_holiday", "is_payday_week",
    "is_month_start", "is_month_end",
    "temp_mean", "temp_range",
    "precipitation", "rain", "snowfall", "windspeed_max",
    "is_rainy", "is_snowy", "is_cold", "is_warm",
    "is_windy", "is_bad_weather", "weather_cat_code",
]

CATEGORICAL_COLS = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]

MODEL_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "objective": "quantile",
    "alpha": 0.5,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

out = sys.stdout.buffer
def p(s): out.write((s + '\n').encode('utf-8')); out.flush()


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_warm_season"] = df["Месяц"].isin([4, 5, 6, 7, 8, 9]).astype(int)
    df["precip_warm"] = df["precipitation"] * df["is_warm_season"]
    df["precip_cold"] = df["precipitation"] * (1 - df["is_warm_season"])
    df["bad_weather_warm"] = df["is_bad_weather"] * df["is_warm_season"]
    df["bad_weather_cold"] = df["is_bad_weather"] * (1 - df["is_warm_season"])
    return df


def add_weather_lag1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grp = df.groupby(["Пекарня", "Номенклатура"])
    df["is_bad_weather_lag1"] = grp["is_bad_weather"].shift(1).fillna(0)
    df["precipitation_lag1"] = grp["precipitation"].shift(1).fillna(0)
    df["bad_warm_lag1"] = df["is_bad_weather_lag1"] * df["is_warm_season"]
    return df


def train_and_eval(X_train, y_train, X_test, y_test, features, label):
    cats = [c for c in CATEGORICAL_COLS if c in features]
    X_tr = X_train[features].copy()
    X_te = X_test[features].copy()
    for col in cats:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(X_tr, y_train, categorical_feature=cats)
    pred = np.clip(model.predict(X_te), 0, None)
    mae = mean_absolute_error(y_test, pred)
    bias = float(np.mean(pred - y_test))
    wmape = float(np.sum(np.abs(y_test - pred)) / np.sum(y_test)) * 100
    p(f"  {label}: MAE={mae:.4f}  Bias={bias:+.2f}  WMAPE={wmape:.2f}%")
    return mae, pred, model


def main():
    p("=" * 70)
    p("Exp 81: Seasonal weather features on stg dataset")
    p("=" * 70)

    DATA_PATH = Path(ROOT) / "data" / "processed" / "daily_sales_stg.csv"
    p(f"Loading {DATA_PATH} ...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    df["Дата"] = pd.to_datetime(df["Дата"])
    p(f"  rows: {len(df):,}  dates: {df['Дата'].min().date()} .. {df['Дата'].max().date()}")
    p(f"  loaded in {time.time()-t0:.1f}s")

    # Train/test split
    cutoff = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < cutoff].copy()
    test  = df[df["Дата"] >= cutoff].copy()
    p(f"  train: {len(train):,} rows  test: {len(test):,} rows  cutoff: {cutoff.date()}")

    # Check no NaN in required features
    missing_feats = [f for f in FEATURES_STG if f not in df.columns]
    if missing_feats:
        p(f"  MISSING features: {missing_feats}")
        return

    y_train = train[TARGET]
    y_test  = test[TARGET]

    # --- Variant A: baseline_stg (FEATURES_STG, no seasonal) ---
    p("\n--- Variant A: baseline_stg ---")
    mae_a, pred_a, _ = train_and_eval(train, y_train, test, y_test, FEATURES_STG, "A")

    # --- Variant B: seasonal_split ---
    p("\n--- Variant B: seasonal_split ---")
    train_b = add_seasonal_features(train)
    test_b  = add_seasonal_features(test)
    features_b = FEATURES_STG + ["is_warm_season", "precip_warm", "precip_cold",
                                  "bad_weather_warm", "bad_weather_cold"]
    mae_b, pred_b, _ = train_and_eval(train_b, y_train, test_b, y_test, features_b, "B")

    # --- Variant C: seasonal + weather lag1 ---
    p("\n--- Variant C: seasonal + bad_weather_lag1 interaction ---")
    train_c = add_weather_lag1(add_seasonal_features(train))
    test_c  = add_weather_lag1(add_seasonal_features(test))
    # fill NaN from lag on first days
    lag_cols = ["is_bad_weather_lag1", "precipitation_lag1", "bad_warm_lag1"]
    for col in lag_cols:
        train_c[col] = train_c[col].fillna(0)
        test_c[col]  = test_c[col].fillna(0)
    features_c = features_b + lag_cols
    mae_c, pred_c, _ = train_and_eval(train_c, y_train, test_c, y_test, features_c, "C")

    # --- Summary ---
    p("\n" + "=" * 70)
    p("SUMMARY")
    p("=" * 70)
    p(f"  A baseline_stg         MAE={mae_a:.4f}")
    p(f"  B seasonal_split       MAE={mae_b:.4f}  delta={mae_b-mae_a:+.4f}")
    p(f"  C seasonal+lag1        MAE={mae_c:.4f}  delta={mae_c-mae_a:+.4f}")

    # Breakdown: summer vs winter test performance for best variant
    best_pred = pred_c if mae_c < mae_b else pred_b
    test_eval = test_c if mae_c < mae_b else test_b
    warm = test_eval["is_warm_season"] == 1
    cold = ~warm
    p(f"\n  Test period breakdown ({test_eval['Дата'].min().date()} .. {test_eval['Дата'].max().date()}):")
    if warm.any():
        p(f"    warm season rows: {warm.sum():,}  MAE={mean_absolute_error(y_test[warm], best_pred[warm]):.4f}")
    if cold.any():
        p(f"    cold season rows: {cold.sum():,}  MAE={mean_absolute_error(y_test[cold], best_pred[cold]):.4f}")

    # Bad weather breakdown for best variant
    bw = test_eval["is_bad_weather"] == 1
    bw_lag1 = test_c.get("is_bad_weather_lag1", pd.Series(0, index=test_c.index)) == 1 if "C" else None
    if bw.any():
        p(f"\n  Bad weather day (D0) rows: {bw.sum():,}  MAE={mean_absolute_error(y_test[bw], best_pred[bw]):.4f}")
        p(f"    Bias={float(np.mean(best_pred[bw] - y_test[bw])):+.2f}")

    # Save results
    import json
    results = {
        "experiment": "81_seasonal_weather",
        "dataset": str(DATA_PATH),
        "target": TARGET,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "variants": {
            "A_baseline_stg":    {"mae": round(float(mae_a), 4)},
            "B_seasonal_split":  {"mae": round(float(mae_b), 4), "delta_vs_A": round(float(mae_b-mae_a), 4)},
            "C_seasonal_lag1":   {"mae": round(float(mae_c), 4), "delta_vs_A": round(float(mae_c-mae_a), 4)},
        }
    }
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    p(f"\nSaved -> {EXP_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
