"""
Experiment 01: Baseline 8 months.

Train LightGBM on 8 months of daily aggregated check data.
Compare with v6 baseline (MAE 3.33, WMAPE 23.6% on 3 months, 5 categories).

Note: not a 1-to-1 comparison -- 8m data has 28 categories and no stock features.

Input:  data/processed/daily_sales_8m.csv
Output: src/experiments_v2/01_baseline_8m/metrics.json
        src/experiments_v2/01_baseline_8m/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/01_baseline_8m/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DAILY_8M_PATH, FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
BASELINE_MAE = 3.33
BASELINE_WMAPE = 23.6


def main():
    print("=" * 60)
    print("  EXPERIMENT 01: Baseline 8 months")
    print("=" * 60)
    t_start = time.time()

    # --- Load data ---
    print(f"\n[1/6] Loading data from {DAILY_8M_PATH}...")
    df = pd.read_csv(str(DAILY_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    print(f"  Days: {df['Дата'].nunique()}, Bakeries: {df['Пекарня'].nunique()}, "
          f"Products: {df['Номенклатура'].nunique()}, Categories: {df['Категория'].nunique()}")

    # --- Select features ---
    print(f"\n[2/6] Selecting features...")
    available = [f for f in FEATURES_V2 if f in df.columns]
    missing = [f for f in FEATURES_V2 if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available)} of {len(FEATURES_V2)} features")

    # Convert categorical columns
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Train/test split ---
    print(f"\n[3/6] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days "
          f"({train['Дата'].min().date()} -- {train['Дата'].max().date()})")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days "
          f"({test['Дата'].min().date()} -- {test['Дата'].max().date()})")

    X_train = train[available]
    y_train = train[TARGET]
    X_test = test[available]
    y_test = test[TARGET]

    # --- Train ---
    print(f"\n[4/6] Training LightGBM (v6 params)...")
    print(f"  Params: n_estimators={MODEL_PARAMS['n_estimators']}, "
          f"lr={MODEL_PARAMS['learning_rate']:.4f}, "
          f"num_leaves={MODEL_PARAMS['num_leaves']}, "
          f"max_depth={MODEL_PARAMS['max_depth']}")

    t_train = time.time()
    model = train_lgbm(X_train, y_train)
    train_time = time.time() - t_train
    print(f"  Training time: {train_time:.0f}s")

    # --- Evaluate ---
    print(f"\n[5/6] Evaluation...")
    y_pred = predict_clipped(model, X_test)

    mae = mean_absolute_error(y_test, y_pred)
    wm = wmape(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(np.asarray(y_test) - y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  === RESULTS ===")
    print_metrics("01_baseline_8m", y_test, y_pred, baseline_mae=BASELINE_MAE)
    print(f"    RMSE  = {rmse:.4f}")
    print(f"    R2    = {r2:.4f}")

    # Delta vs v6 baseline
    print(f"\n  === vs v6 baseline (3 months, 5 categories) ===")
    print(f"    MAE:   {mae:.4f}  vs  {BASELINE_MAE:.4f}  (delta: {mae - BASELINE_MAE:+.4f})")
    print(f"    WMAPE: {wm:.2f}%  vs  {BASELINE_WMAPE:.1f}%  (delta: {wm - BASELINE_WMAPE:+.2f}%)")
    print(f"    NOTE: not 1-to-1 (28 vs 5 categories, no stock features)")

    # Per-category breakdown
    print(f"\n  === Per-category metrics ===")
    print_category_metrics(y_test, y_pred, test["Категория"].values)

    # --- Feature importance ---
    print(f"\n[6/6] Feature importance (top 20)...")
    importance = pd.DataFrame({
        "feature": available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(20).iterrows():
        print(f"    {row['feature']:<25} {row['importance']:>8.0f}")

    # --- Save results ---
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "01_baseline_8m",
        "mae": round(mae, 4),
        "wmape": round(wm, 2),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_days": int(train["Дата"].nunique()),
        "test_days": int(test["Дата"].nunique()),
        "n_features": len(available),
        "n_bakeries": int(df["Пекарня"].nunique()),
        "n_products": int(df["Номенклатура"].nunique()),
        "n_categories": int(df["Категория"].nunique()),
        "train_time_s": round(train_time, 1),
        "baseline_v6_mae": BASELINE_MAE,
        "baseline_v6_wmape": BASELINE_WMAPE,
        "feature_importance_top10": importance.head(10)[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact": y_test.values,
        "pred": np.round(y_pred, 2),
        "abs_error": np.round(np.abs(y_test.values - y_pred), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    # --- MLflow ---
    print(f"\n  Logging to MLflow...")
    mlflow.set_experiment("experiments_v2")
    with mlflow.start_run(run_name="01_baseline_8m"):
        # Params
        mlflow.log_params({
            "experiment": "01_baseline_8m",
            "n_features": len(available),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_days": int(train["Дата"].nunique()),
            "test_days": TEST_DAYS,
            "n_bakeries": int(df["Пекарня"].nunique()),
            "n_products": int(df["Номенклатура"].nunique()),
            "n_categories": int(df["Категория"].nunique()),
        })
        mlflow.log_params({f"lgbm_{k}": v for k, v in MODEL_PARAMS.items()})

        # Metrics
        mlflow.log_metrics({
            "mae": mae,
            "wmape": wm,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "train_time_s": train_time,
        })

        # Artifacts
        mlflow.log_artifact(str(EXP_DIR / "metrics.json"))
        mlflow.log_artifact(str(EXP_DIR / "predictions.csv"))

        # Model
        mlflow.lightgbm.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run_id: {run_id}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  MLflow UI: mlflow ui --port 5001")
    print("  Done!")


if __name__ == "__main__":
    main()
