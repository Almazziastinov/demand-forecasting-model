"""
Experiment 54: Mixture of Experts on DEMAND target.

Same as exp 44 (split at product_avg >= 15) but trained on Spros (demand).
Data: daily_sales_8m_demand.csv with demand lags.
Baseline: exp 03 Model B (MAE 2.9923, WMAPE 28.19% vs Spros).

Usage:
  .venv/Scripts/python.exe src/experiments_v2/54_mixture_of_experts_demand/run.py
"""

import sys, os, time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
DEMAND_CSV = Path(ROOT) / "data" / "processed" / "daily_sales_8m_demand.csv"
DEMAND_TARGET = "Спрос"
BASELINE_MAE = 2.9923
BASELINE_WMAPE = 28.19
THRESHOLD = 15  # split point

DEMAND_FEATURES = [
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
    "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30",
    "demand_roll_std7",
]
ALL_FEATURES = FEATURES_V2 + DEMAND_FEATURES


def main():
    print("=" * 60)
    print(f"  EXPERIMENT 54: Mixture of Experts (split at avg={THRESHOLD}) (DEMAND target)")
    print("=" * 60)
    t_start = time.time()

    # Load
    if not DEMAND_CSV.exists():
        print(f"  ERROR: {DEMAND_CSV} not found!")
        return
    df = pd.read_csv(str(DEMAND_CSV), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}, Days: {df['Дата'].nunique()}")

    available = [f for f in ALL_FEATURES if f in df.columns]

    # Split train/test
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    # Compute product average from TRAIN only (on demand)
    product_avg = train.groupby("Номенклатура")[DEMAND_TARGET].mean()
    train["product_avg"] = train["Номенклатура"].map(product_avg).fillna(0)
    test["product_avg"] = test["Номенклатура"].map(product_avg).fillna(0)

    # Additional features for high demand
    product_std = train.groupby("Номенклатура")[DEMAND_TARGET].std().fillna(0)
    train["product_std"] = train["Номенклатура"].map(product_std).fillna(0)
    test["product_std"] = test["Номенклатура"].map(product_std).fillna(0)

    # Split into high/low
    train_high = train[train["product_avg"] >= THRESHOLD].copy()
    train_low = train[train["product_avg"] < THRESHOLD].copy()
    test_high = test[test["product_avg"] >= THRESHOLD].copy()
    test_low = test[test["product_avg"] < THRESHOLD].copy()

    print(f"\n  High demand (avg >= {THRESHOLD}):")
    print(f"    Train: {len(train_high):,} ({len(train_high)/len(train)*100:.1f}%)")
    print(f"    Test:  {len(test_high):,} ({len(test_high)/len(test)*100:.1f}%)")
    print(f"  Low demand (avg < {THRESHOLD}):")
    print(f"    Train: {len(train_low):,} ({len(train_low)/len(train)*100:.1f}%)")
    print(f"    Test:  {len(test_low):,} ({len(test_low)/len(test)*100:.1f}%)")

    # Features
    features_high = available + ["product_avg", "product_std"]
    features_low = available

    for col in CATEGORICAL_COLS_V2:
        for d in [train_high, train_low, test_high, test_low]:
            if col in d.columns:
                d[col] = d[col].astype("category")

    total_train_time = 0

    # --- Model HIGH ---
    print(f"\n  --- Training model_high (Huber, deeper) on {DEMAND_TARGET} ---")
    params_high = MODEL_PARAMS.copy()
    params_high["objective"] = "huber"
    params_high["huber_delta"] = 5.0
    params_high["max_depth"] = 9
    params_high["n_estimators"] = 3000
    params_high.pop("metric", None)

    t_train = time.time()
    model_high = LGBMRegressor(**params_high)
    model_high.fit(train_high[features_high], train_high[DEMAND_TARGET])
    tt = time.time() - t_train
    total_train_time += tt
    print(f"  Training time: {tt:.0f}s")

    # --- Model LOW ---
    print(f"\n  --- Training model_low (standard) on {DEMAND_TARGET} ---")
    t_train = time.time()
    params_low = MODEL_PARAMS.copy()
    model_low = LGBMRegressor(**params_low)
    model_low.fit(train_low[features_low], train_low[DEMAND_TARGET])
    tt = time.time() - t_train
    total_train_time += tt
    print(f"  Training time: {tt:.0f}s")

    # --- Predict ---
    pred_high = predict_clipped(model_high, test_high[features_high]) if len(test_high) > 0 else np.array([])
    pred_low = predict_clipped(model_low, test_low[features_low]) if len(test_low) > 0 else np.array([])

    # Merge predictions back
    test_high = test_high.copy()
    test_low = test_low.copy()
    if len(test_high) > 0:
        test_high["pred"] = pred_high
    if len(test_low) > 0:
        test_low["pred"] = pred_low

    test_merged = pd.concat([test_high, test_low], ignore_index=True)
    y_test_all = test_merged[DEMAND_TARGET].values
    y_pred_all = test_merged["pred"].values

    # Evaluate
    mae = mean_absolute_error(y_test_all, y_pred_all)
    wm = wmape(y_test_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
    bias = np.mean(y_test_all - y_pred_all)
    r2 = r2_score(y_test_all, y_pred_all)

    print(f"\n  === COMBINED RESULTS (vs {DEMAND_TARGET}) ===")
    print_metrics("54_moe_demand", y_test_all, y_pred_all, baseline_mae=BASELINE_MAE)
    print(f"    RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    print(f"    Delta vs baseline: MAE {mae - BASELINE_MAE:+.4f}, WMAPE {wm - BASELINE_WMAPE:+.2f}%")

    # Per-segment metrics
    if len(test_high) > 0:
        mae_h = mean_absolute_error(test_high[DEMAND_TARGET], pred_high)
        bias_h = np.mean(test_high[DEMAND_TARGET].values - pred_high)
        print(f"\n  High demand: MAE={mae_h:.4f}, Bias={bias_h:+.4f}")
    if len(test_low) > 0:
        mae_l = mean_absolute_error(test_low[DEMAND_TARGET], pred_low)
        bias_l = np.mean(test_low[DEMAND_TARGET].values - pred_low)
        print(f"  Low demand:  MAE={mae_l:.4f}, Bias={bias_l:+.4f}")

    # High demand buckets
    y_t = y_test_all.astype(float)
    y_p = y_pred_all.astype(float)
    for threshold in [15, 50, 100]:
        mask = y_t >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_t[mask], y_p[mask])
            bias_h = np.mean(y_t[mask] - y_p[mask])
            print(f"    Demand >= {threshold}: N={mask.sum()}, MAE={mae_h:.2f}, Bias={bias_h:+.2f}")

    # Per-category
    print(f"\n  === Per-category metrics ===")
    print_category_metrics(y_test_all, y_pred_all, test_merged["Категория"].values)

    # Save
    metrics = {
        "experiment": "54_mixture_of_experts_demand",
        "target": DEMAND_TARGET,
        "threshold": THRESHOLD,
        "high_objective": "huber",
        "high_max_depth": 9,
        "mae": round(mae, 4), "wmape": round(wm, 2),
        "rmse": round(rmse, 4), "bias": round(bias, 4), "r2": round(r2, 4),
        "train_rows": len(train), "test_rows": len(test),
        "n_high_train": len(train_high), "n_high_test": len(test_high),
        "n_low_train": len(train_low), "n_low_test": len(test_low),
        "n_features": len(available),
        "train_time_s": round(total_train_time, 1),
        "baseline_03_mae": BASELINE_MAE, "baseline_03_wmape": BASELINE_WMAPE,
    }
    predictions = pd.DataFrame({
        "Дата": test_merged["Дата"].values, "Пекарня": test_merged["Пекарня"].values,
        "Номенклатура": test_merged["Номенклатура"].values,
        "Категория": test_merged["Категория"].values,
        "fact": y_test_all, "pred": np.round(y_pred_all, 2),
        "abs_error": np.round(np.abs(y_test_all - y_pred_all), 2),
        "segment": ["high" if a >= THRESHOLD else "low" for a in test_merged["product_avg"].values],
    })
    save_results(EXP_DIR, metrics, predictions)

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
