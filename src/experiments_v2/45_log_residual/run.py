"""
Experiment 45: Log-target + Residual correction (two-stage).

Stage 1: LightGBM on log1p(y) -> base prediction (good for high demand)
Stage 2: LightGBM on (y - expm1(base)) -> correction (fixes low demand)
Final:   expm1(base) + correction

Best of both worlds: log handles scale, residual corrects bias.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/45_log_residual/run.py
"""

import sys, os, time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DAILY_8M_PATH, FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
BASELINE_MAE = 2.2904
BASELINE_WMAPE = 25.91


def main():
    print("=" * 60)
    print("  EXPERIMENT 45: Log-target + Residual correction")
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
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    X_train, y_train = train[available], train[TARGET].values
    X_test, y_test = test[available], test[TARGET].values

    total_train_time = 0

    # --- Stage 1: Log model ---
    print(f"\n  --- Stage 1: LightGBM on log1p(y) ---")
    y_train_log = np.log1p(y_train)

    t_train = time.time()
    model_log = train_lgbm(X_train, y_train_log)
    tt = time.time() - t_train
    total_train_time += tt
    print(f"  Training time: {tt:.0f}s")

    # Stage 1 predictions (on train for residual, on test for eval)
    base_train = np.expm1(model_log.predict(X_train))
    base_test = np.expm1(model_log.predict(X_test))

    mae_stage1 = mean_absolute_error(y_test, np.maximum(base_test, 0))
    print(f"  Stage 1 MAE (test): {mae_stage1:.4f}")

    # --- Stage 2: Residual model ---
    print(f"\n  --- Stage 2: LightGBM on residual (y - base) ---")
    residual_train = y_train - base_train

    print(f"  Residual stats: mean={residual_train.mean():.4f}, "
          f"std={residual_train.std():.4f}, "
          f"min={residual_train.min():.2f}, max={residual_train.max():.2f}")

    t_train = time.time()
    model_residual = train_lgbm(X_train, residual_train)
    tt = time.time() - t_train
    total_train_time += tt
    print(f"  Training time: {tt:.0f}s")

    # --- Final prediction ---
    correction_test = model_residual.predict(X_test)
    y_pred = np.maximum(base_test + correction_test, 0)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    wm = wmape(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(y_test - y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  === RESULTS (Stage 1 + Stage 2) ===")
    print_metrics("45_log_residual", y_test, y_pred, baseline_mae=BASELINE_MAE)
    print(f"    RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    print(f"    Stage 1 only:  MAE={mae_stage1:.4f}")
    print(f"    Stage 1+2:     MAE={mae:.4f}")
    print(f"    Correction:    {mae_stage1 - mae:+.4f} MAE")
    print(f"    Delta vs baseline: MAE {mae - BASELINE_MAE:+.4f}, WMAPE {wm - BASELINE_WMAPE:+.2f}%")

    # High demand
    y_t = y_test.astype(float)
    y_p = y_pred.astype(float)
    for threshold in [15, 50, 100]:
        mask = y_t >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_t[mask], y_p[mask])
            bias_h = np.mean(y_t[mask] - y_p[mask])
            print(f"    Demand >= {threshold}: N={mask.sum()}, MAE={mae_h:.2f}, Bias={bias_h:+.2f}")

    # Per-category
    print(f"\n  === Per-category metrics ===")
    print_category_metrics(y_test, y_pred, test["Категория"].values)

    # Save
    importance_log = pd.DataFrame({
        "feature": available, "importance": model_log.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics = {
        "experiment": "45_log_residual", "stages": 2,
        "mae_stage1": round(mae_stage1, 4),
        "mae": round(mae, 4), "wmape": round(wm, 2),
        "rmse": round(rmse, 4), "bias": round(bias, 4), "r2": round(r2, 4),
        "train_rows": len(train), "test_rows": len(test),
        "train_time_s": round(total_train_time, 1),
        "baseline_01_mae": BASELINE_MAE, "baseline_01_wmape": BASELINE_WMAPE,
        "feature_importance_top10": importance_log.head(10)[["feature", "importance"]].to_dict("records"),
    }
    predictions = pd.DataFrame({
        "Дата": test["Дата"].values, "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values, "Категория": test["Категория"].values,
        "fact": y_test, "pred": np.round(y_pred, 2),
        "pred_stage1": np.round(np.maximum(base_test, 0), 2),
        "correction": np.round(correction_test, 2),
        "abs_error": np.round(np.abs(y_test - y_pred), 2),
    })
    save_results(EXP_DIR, metrics, predictions)

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
