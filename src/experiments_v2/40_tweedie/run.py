"""
Experiment 40: Tweedie regression.

objective='tweedie', tweedie_variance_power=1.5
Var(y) ~ mean^p -- automatically accounts for variance growing with mean.
Between Poisson (p=1) and Gamma (p=2).

Hypothesis: MSE loss trains conditional mean, which "averages" when distribution
is wide (high demand products). Tweedie loss naturally handles heteroscedasticity.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/40_tweedie/run.py
"""

import sys, os, time, json
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


def main():
    print("=" * 60)
    print("  EXPERIMENT 40: Tweedie regression")
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

    # Train with Tweedie
    params = MODEL_PARAMS.copy()
    params["objective"] = "tweedie"
    params["tweedie_variance_power"] = 1.5
    # Remove metric conflict
    params.pop("metric", None)

    print(f"  Training Tweedie (power=1.5)...")
    t_train = time.time()
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - t_train
    print(f"  Training time: {train_time:.0f}s")

    # Evaluate
    y_pred = predict_clipped(model, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    wm = wmape(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(np.asarray(y_test) - y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  === RESULTS ===")
    print_metrics("40_tweedie", y_test, y_pred, baseline_mae=BASELINE_MAE)
    print(f"    RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    print(f"    Delta vs baseline: MAE {mae - BASELINE_MAE:+.4f}, WMAPE {wm - BASELINE_WMAPE:+.2f}%")

    # High demand analysis
    y_t = np.asarray(y_test, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    for threshold in [15, 50, 100]:
        mask = y_t >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_t[mask], y_p[mask])
            bias_h = np.mean(y_t[mask] - y_p[mask])
            print(f"    Demand >= {threshold}: N={mask.sum()}, MAE={mae_h:.2f}, Bias={bias_h:+.2f}")

    # Per-category
    print(f"\n  === Per-category metrics ===")
    print_category_metrics(y_test, y_pred, test["Категория"].values)

    # Feature importance
    importance = pd.DataFrame({
        "feature": available, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<25} {row['importance']:>8.0f}")

    # Save
    metrics = {
        "experiment": "40_tweedie",
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "mae": round(mae, 4), "wmape": round(wm, 2),
        "rmse": round(rmse, 4), "bias": round(bias, 4), "r2": round(r2, 4),
        "train_rows": len(train), "test_rows": len(test),
        "train_time_s": round(train_time, 1),
        "baseline_01_mae": BASELINE_MAE, "baseline_01_wmape": BASELINE_WMAPE,
        "feature_importance_top10": importance.head(10)[["feature", "importance"]].to_dict("records"),
    }
    predictions = pd.DataFrame({
        "Дата": test["Дата"].values, "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values, "Категория": test["Категория"].values,
        "fact": y_test.values, "pred": np.round(y_pred, 2),
        "abs_error": np.round(np.abs(y_test.values - y_pred), 2),
    })
    save_results(EXP_DIR, metrics, predictions)

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
