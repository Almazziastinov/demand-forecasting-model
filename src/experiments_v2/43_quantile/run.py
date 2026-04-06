"""
Experiment 43: Quantile regression (P25, P50, P75).

Three models predicting different quantiles of the distribution.
P50 (median) is more robust to right tail than mean.
Business value: instead of "198" -> "160-200-240" interval.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/43_quantile/run.py
"""

import sys, os, time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DAILY_8M_PATH, FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
BASELINE_MAE = 2.2904
BASELINE_WMAPE = 25.91

QUANTILES = [0.25, 0.50, 0.75]


def main():
    print("=" * 60)
    print("  EXPERIMENT 43: Quantile regression (P25, P50, P75)")
    print("=" * 60)
    t_start = time.time()

    # Load
    df = pd.read_csv(str(DAILY_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}, Days: {df['Дата'].nunique()}")

    available = [f for f in FEATURES_V2 if f in df.columns]
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Split
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start]
    test = df[df["Дата"] >= test_start]
    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    X_train, y_train = train[available], train[TARGET]
    X_test, y_test = test[available], test[TARGET]

    results = {}
    all_preds = {
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact": y_test.values,
    }

    total_train_time = 0

    for q in QUANTILES:
        q_name = f"P{int(q*100)}"
        print(f"\n  --- Training {q_name} (alpha={q}) ---")

        params = MODEL_PARAMS.copy()
        params["objective"] = "quantile"
        params["alpha"] = q
        params.pop("metric", None)

        t_train = time.time()
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        tt = time.time() - t_train
        total_train_time += tt
        print(f"  Training time: {tt:.0f}s")

        y_pred = predict_clipped(model, X_test)
        mae = mean_absolute_error(y_test, y_pred)
        wm = wmape(y_test, y_pred)
        bias = np.mean(np.asarray(y_test) - y_pred)

        print(f"  {q_name}: MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}")

        results[q_name] = {"mae": mae, "wmape": wm, "bias": bias}
        all_preds[f"pred_{q_name}"] = np.round(y_pred, 2)

    # Primary metric: P50 (median)
    y_pred_p50 = all_preds["pred_P50"]
    mae = mean_absolute_error(y_test, y_pred_p50)
    wm = wmape(y_test, y_pred_p50)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_p50))
    bias = np.mean(np.asarray(y_test) - y_pred_p50)
    r2 = r2_score(y_test, y_pred_p50)

    print(f"\n  === PRIMARY METRIC (P50 median) ===")
    print_metrics("43_quantile_P50", y_test, y_pred_p50, baseline_mae=BASELINE_MAE)
    print(f"    RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    print(f"    Delta vs baseline: MAE {mae - BASELINE_MAE:+.4f}, WMAPE {wm - BASELINE_WMAPE:+.2f}%")

    # Interval width analysis
    p25 = np.asarray(all_preds["pred_P25"], dtype=float)
    p75 = np.asarray(all_preds["pred_P75"], dtype=float)
    width = p75 - p25
    print(f"\n  === Interval width (P75 - P25) ===")
    print(f"    Mean: {width.mean():.2f}, Median: {np.median(width):.2f}, "
          f"Max: {width.max():.2f}")

    # Coverage: how often actual falls within [P25, P75]
    y_t = np.asarray(y_test, dtype=float)
    coverage = np.mean((y_t >= p25) & (y_t <= p75))
    print(f"    Coverage (actual in [P25, P75]): {coverage*100:.1f}%  (expected ~50%)")

    # High demand
    y_p = np.asarray(y_pred_p50, dtype=float)
    for threshold in [15, 50, 100]:
        mask = y_t >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_t[mask], y_p[mask])
            bias_h = np.mean(y_t[mask] - y_p[mask])
            width_h = (p75[mask] - p25[mask]).mean()
            print(f"    Demand >= {threshold}: N={mask.sum()}, MAE={mae_h:.2f}, "
                  f"Bias={bias_h:+.2f}, Interval={width_h:.1f}")

    # Per-category
    print(f"\n  === Per-category metrics (P50) ===")
    print_category_metrics(y_test, y_pred_p50, test["Категория"].values)

    # Save
    metrics = {
        "experiment": "43_quantile",
        "quantiles": QUANTILES,
        "primary_quantile": "P50",
        "mae_P25": round(results["P25"]["mae"], 4),
        "mae_P50": round(results["P50"]["mae"], 4),
        "mae_P75": round(results["P75"]["mae"], 4),
        "wmape_P50": round(results["P50"]["wmape"], 2),
        "mae": round(mae, 4), "wmape": round(wm, 2),
        "rmse": round(rmse, 4), "bias": round(bias, 4), "r2": round(r2, 4),
        "interval_mean_width": round(width.mean(), 4),
        "coverage_pct": round(coverage * 100, 2),
        "train_rows": len(train), "test_rows": len(test),
        "train_time_s": round(total_train_time, 1),
        "baseline_01_mae": BASELINE_MAE, "baseline_01_wmape": BASELINE_WMAPE,
    }
    predictions = pd.DataFrame(all_preds)
    predictions["abs_error"] = np.round(np.abs(y_test.values - y_pred_p50), 2)
    save_results(EXP_DIR, metrics, predictions)

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
