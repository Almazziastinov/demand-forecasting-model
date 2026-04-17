"""
Experiment 69: Quantile intervals on demand target.

Trains three LightGBM quantile models on the demand target:
  - P10: lower tail
  - P50: median / main forecast
  - P90: upper tail

The goal is to expose forecast uncertainty explicitly and evaluate:
  - point accuracy of P50
  - pinball loss for all quantiles
  - interval coverage for [P10, P90]
  - asymmetry of the prediction interval

Usage:
  .venv/Scripts/python.exe src/experiments_v2/69_quantile_intervals_v3/run.py
"""

from __future__ import annotations

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
    CATEGORICAL_COLS_V2,
    DEMAND_8M_PATH,
    DEMAND_TARGET,
    FEATURES_V3,
    TARGET,
    predict_clipped,
    print_category_metrics,
    print_metrics,
    save_results,
    train_quantile,
    wmape,
)


EXP_DIR = Path(__file__).resolve().parent
QUANTILES = [("P10", 0.10), ("P50", 0.50), ("P90", 0.90)]
TEST_DAYS = 30

DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"
CATEGORY_COL = "Категория"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball loss for a single quantile."""
    errors = y_true - y_pred
    return float(np.mean(np.maximum(alpha * errors, (alpha - 1) * errors)))


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Evaluate the prediction interval formed by lower and upper bounds."""
    width = upper - lower
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return {
        "coverage_pct": float(coverage * 100),
        "width_mean": float(width.mean()),
        "width_median": float(np.median(width)),
        "width_p90": float(np.quantile(width, 0.9)),
    }


def main() -> None:
    print("=" * 60)
    print("  EXPERIMENT 69: Quantile intervals on demand target")
    print("  Models: P10 / P50 / P90")
    print("=" * 60)
    t_start = time.time()

    if not DEMAND_8M_PATH.exists():
        print(f"  ERROR: {DEMAND_8M_PATH} not found!")
        return

    print(f"\n[1/5] Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(
        f"  Shape: {df.shape}, Days: {df[DATE_COL].nunique()}, "
        f"Bakeries: {df[BAKERY_COL].nunique()}, Products: {df[PRODUCT_COL].nunique()}"
    )

    available = [f for f in FEATURES_V3 if f in df.columns]
    missing = [f for f in FEATURES_V3 if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing {len(missing)} features: {missing[:10]}")
    print(f"  Using {len(available)} of {len(FEATURES_V3)} FEATURES_V3 columns")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"\n[2/5] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df[DATE_COL].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df[DATE_COL] < test_start].copy()
    test = df[df[DATE_COL] >= test_start].copy()
    print(f"  Train: {len(train):,} rows, {train[DATE_COL].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test[DATE_COL].nunique()} days")

    X_train = train[available]
    y_train = train[DEMAND_TARGET].astype(float)
    X_test = test[available]
    y_test = test[DEMAND_TARGET].astype(float)
    y_test_sold = test[TARGET].astype(float) if TARGET in test.columns else None

    all_preds = pd.DataFrame(
        {
            DATE_COL: test[DATE_COL].values,
            BAKERY_COL: test[BAKERY_COL].values,
            PRODUCT_COL: test[PRODUCT_COL].values,
            CATEGORY_COL: test[CATEGORY_COL].values if CATEGORY_COL in test.columns else None,
            "fact_demand": y_test.values,
        }
    )
    if y_test_sold is not None:
        all_preds["fact_sold"] = y_test_sold.values

    print(f"\n[3/5] Training quantile models...")
    total_train_time = 0.0
    models: dict[str, object] = {}
    raw_preds: dict[str, np.ndarray] = {}
    summary_rows: list[dict] = []

    for q_name, alpha in QUANTILES:
        print(f"\n  --- {q_name} (alpha={alpha}) ---")
        t_q = time.time()
        model = train_quantile(X_train, y_train, alpha=alpha)
        train_time = time.time() - t_q
        total_train_time += train_time
        models[q_name] = model

        y_pred = predict_clipped(model, X_test)
        raw_preds[q_name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        bias = float(np.mean(y_test.values - y_pred))
        wm = wmape(y_test, y_pred)
        pbl = pinball_loss(y_test.values, y_pred, alpha)

        summary_rows.append(
            {
                "quantile": q_name,
                "alpha": alpha,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "wmape": round(wm, 2),
                "bias": round(bias, 4),
                "pinball_loss": round(pbl, 6),
                "train_time_s": round(train_time, 1),
            }
        )
        print(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}")
        print(f"    Pinball={pbl:.6f}, Train time={train_time:.0f}s")

        all_preds[f"pred_{q_name}"] = np.round(y_pred, 2)

    print(f"\n[4/5] Evaluation...")
    y_p10 = np.asarray(raw_preds["P10"], dtype=float)
    y_p50 = np.asarray(raw_preds["P50"], dtype=float)
    y_p90 = np.asarray(raw_preds["P90"], dtype=float)
    y_true = y_test.values.astype(float)

    lower = np.minimum(y_p10, y_p90)
    upper = np.maximum(y_p10, y_p90)
    interval = interval_metrics(y_true, lower, upper)

    mae_p50 = mean_absolute_error(y_true, y_p50)
    mse_p50 = mean_squared_error(y_true, y_p50)
    rmse_p50 = np.sqrt(mse_p50)
    wmape_p50 = wmape(y_true, y_p50)
    bias_p50 = float(np.mean(y_true - y_p50))
    r2_p50 = np.nan
    if len(y_true) >= 2 and not np.allclose(y_true, y_true[0]):
        r2_p50 = float(r2_score(y_true, y_p50))

    crossing_p10_p50 = float(np.mean(y_p10 > y_p50) * 100)
    crossing_p50_p90 = float(np.mean(y_p50 > y_p90) * 100)
    crossing_any = float(np.mean((y_p10 > y_p50) | (y_p50 > y_p90)) * 100)
    asymmetry = float(np.mean((y_p90 - y_p50) - (y_p50 - y_p10)))

    print("\n  === PRIMARY METRIC (P50) ===")
    print_metrics("69_quantile_intervals_P50", y_true, y_p50)
    print(f"    RMSE = {rmse_p50:.4f}, R2 = {r2_p50:.4f}")
    print(f"    WMAPE = {wmape_p50:.2f}%")

    print("\n  === Interval metrics [P10, P90] ===")
    print(f"    Coverage = {interval['coverage_pct']:.2f}% (expected ~80%)")
    print(f"    Width mean = {interval['width_mean']:.2f}")
    print(f"    Width median = {interval['width_median']:.2f}")
    print(f"    Width P90 = {interval['width_p90']:.2f}")
    print(f"    Asymmetry mean = {asymmetry:+.2f}")
    print(f"    Crossing P10>P50 = {crossing_p10_p50:.2f}%")
    print(f"    Crossing P50>P90 = {crossing_p50_p90:.2f}%")
    print(f"    Any crossing = {crossing_any:.2f}%")

    print(f"\n  === Per-category metrics (P50) ===")
    if CATEGORY_COL in test.columns:
        print_category_metrics(y_true, y_p50, test[CATEGORY_COL].values)

    if y_test_sold is not None:
        mae_vs_sold = mean_absolute_error(y_test_sold.values, y_p50)
        wmape_vs_sold = wmape(y_test_sold.values, y_p50)
        print(f"\n  === Reference vs sold ===")
        print(f"    MAE = {mae_vs_sold:.4f}, WMAPE = {wmape_vs_sold:.2f}%")

    all_preds["interval_lower"] = np.round(lower, 2)
    all_preds["interval_upper"] = np.round(upper, 2)
    all_preds["interval_width"] = np.round(upper - lower, 2)
    all_preds["in_interval"] = ((y_true >= lower) & (y_true <= upper)).astype(int)
    all_preds["abs_error_P50"] = np.round(np.abs(y_true - y_p50), 2)
    all_preds["error_P50"] = np.round(y_true - y_p50, 2)

    metrics = {
        "experiment": "69_quantile_intervals_v3",
        "target": DEMAND_TARGET,
        "features": "FEATURES_V3",
        "quantiles": [q for q, _ in QUANTILES],
        "n_features": len(available),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_days": int(test[DATE_COL].nunique()),
        "train_time_s": round(total_train_time, 1),
        "p10_mae": next(row["mae"] for row in summary_rows if row["quantile"] == "P10"),
        "p50_mae": next(row["mae"] for row in summary_rows if row["quantile"] == "P50"),
        "p90_mae": next(row["mae"] for row in summary_rows if row["quantile"] == "P90"),
        "p10_pinball_loss": next(row["pinball_loss"] for row in summary_rows if row["quantile"] == "P10"),
        "p50_pinball_loss": next(row["pinball_loss"] for row in summary_rows if row["quantile"] == "P50"),
        "p90_pinball_loss": next(row["pinball_loss"] for row in summary_rows if row["quantile"] == "P90"),
        "mae": round(mae_p50, 4),
        "mse": round(mse_p50, 4),
        "rmse": round(rmse_p50, 4),
        "wmape": round(wmape_p50, 2),
        "bias": round(bias_p50, 4),
        "r2": round(r2_p50, 4) if np.isfinite(r2_p50) else None,
        "coverage_pct": round(interval["coverage_pct"], 2),
        "interval_width_mean": round(interval["width_mean"], 4),
        "interval_width_median": round(interval["width_median"], 4),
        "interval_width_p90": round(interval["width_p90"], 4),
        "interval_asymmetry_mean": round(asymmetry, 4),
        "crossing_p10_p50_pct": round(crossing_p10_p50, 2),
        "crossing_p50_p90_pct": round(crossing_p50_p90, 2),
        "crossing_any_pct": round(crossing_any, 2),
    }

    predictions = all_preds.copy()
    save_results(EXP_DIR, metrics, predictions)

    print("\n[5/5] Done!")
    print(f"  Total time: {time.time() - t_start:.0f}s")
    print(f"  Saved predictions and metrics in: {EXP_DIR}")


if __name__ == "__main__":
    main()
