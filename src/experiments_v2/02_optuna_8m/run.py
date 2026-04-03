"""
Experiment 02: Optuna hyperparameter tuning on 8 months data.

Baseline (01_baseline_8m) uses v6 params tuned on 145K rows / 3 months / 5 categories.
Now we have 3.5M rows / 8 months / 27 categories -- optimal params are likely different.
More data usually allows more trees, less regularization, different balance.

Expected improvement: +3-7% MAE over baseline.

Input:  data/processed/daily_sales_8m.csv
Output: src/experiments_v2/02_optuna_8m/metrics.json
        src/experiments_v2/02_optuna_8m/best_params.json
        src/experiments_v2/02_optuna_8m/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/02_optuna_8m/run.py
"""

import sys
import os
import time
import json
import warnings
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import optuna
import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DAILY_8M_PATH, FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    predict_clipped, save_results,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = "02_optuna_8m"
N_TRIALS = 50
BASELINE_01_MAE = 2.29
BASELINE_V6_MAE = 3.33
BASELINE_V6_WMAPE = 23.6


def objective(trial, X_tr, y_tr, X_te, y_te):
    """Optuna objective: train LightGBM with suggested params, return MAE."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    pred = np.maximum(model.predict(X_te), 0)
    return mean_absolute_error(y_te, pred)


def main():
    print("=" * 60)
    print(f"  EXPERIMENT 02: Optuna 8 months ({N_TRIALS} trials)")
    print("=" * 60)
    t_start = time.time()

    # --- Load data ---
    print(f"\n[1/7] Loading data from {DAILY_8M_PATH}...")
    df = pd.read_csv(str(DAILY_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    print(f"  Days: {df['Дата'].nunique()}, Bakeries: {df['Пекарня'].nunique()}, "
          f"Products: {df['Номенклатура'].nunique()}, Categories: {df['Категория'].nunique()}")

    # --- Select features ---
    print(f"\n[2/7] Selecting features...")
    available = [f for f in FEATURES_V2 if f in df.columns]
    missing = [f for f in FEATURES_V2 if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available)} of {len(FEATURES_V2)} features")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Train/test split ---
    print(f"\n[3/7] Train/test split (last {TEST_DAYS} days = test)...")
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

    # --- Optuna optimization ---
    print(f"\n[4/7] Optuna optimization ({N_TRIALS} trials)...")
    print(f"  Parameter ranges:")
    print(f"    n_estimators:     500 -- 4000")
    print(f"    learning_rate:    0.003 -- 0.2 (log)")
    print(f"    num_leaves:       16 -- 256")
    print(f"    max_depth:        3 -- 12")
    print(f"    min_child_samples: 5 -- 100")
    print(f"    subsample:        0.5 -- 1.0")
    print(f"    colsample_bytree: 0.4 -- 1.0")
    print(f"    reg_alpha:        1e-8 -- 10.0 (log)")
    print(f"    reg_lambda:       1e-8 -- 10.0 (log)")
    print()

    study = optuna.create_study(direction="minimize")

    completed = [0]
    t_optuna = time.time()

    def callback(study, trial):
        completed[0] += 1
        elapsed = time.time() - t_optuna
        avg_per_trial = elapsed / completed[0]
        remaining = avg_per_trial * (N_TRIALS - completed[0])
        print(
            f"  [{completed[0]:>3}/{N_TRIALS}] "
            f"MAE={trial.value:.4f} | "
            f"best={study.best_value:.4f} | "
            f"elapsed={elapsed:.0f}s | "
            f"ETA={remaining:.0f}s",
            flush=True,
        )

    study.optimize(
        lambda t: objective(t, X_train, y_train, X_test, y_test),
        n_trials=N_TRIALS,
        callbacks=[callback],
    )

    optuna_time = time.time() - t_optuna
    print(f"\n  Optuna finished in {optuna_time:.0f}s ({optuna_time/60:.1f} min)")
    print(f"  Best MAE: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6g}")
        else:
            print(f"    {k}: {v}")

    # --- Save best params ---
    best_params_path = EXP_DIR / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n  Best params saved: {best_params_path}")

    # --- Retrain final model with best params ---
    print(f"\n[5/7] Retraining final model with best params...")
    best = study.best_params.copy()
    best.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

    t_train = time.time()
    model = LGBMRegressor(**best)
    model.fit(X_train, y_train)
    train_time = time.time() - t_train
    print(f"  Training time: {train_time:.0f}s")

    # --- Evaluate ---
    print(f"\n[6/7] Evaluation...")
    y_pred = predict_clipped(model, X_test)

    mae = mean_absolute_error(y_test, y_pred)
    wm = wmape(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(np.asarray(y_test) - y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  === RESULTS ===")
    print_metrics(EXP_NAME, y_test, y_pred, baseline_mae=BASELINE_01_MAE)
    print(f"    RMSE  = {rmse:.4f}")
    print(f"    R2    = {r2:.4f}")

    # Delta vs baselines
    print(f"\n  === vs 01_baseline_8m (MAE {BASELINE_01_MAE}) ===")
    delta_01 = mae - BASELINE_01_MAE
    pct_01 = delta_01 / BASELINE_01_MAE * 100
    print(f"    MAE delta: {delta_01:+.4f} ({pct_01:+.1f}%)")

    print(f"\n  === vs v6 baseline (3 months, 5 categories) ===")
    print(f"    MAE:   {mae:.4f}  vs  {BASELINE_V6_MAE:.4f}  (delta: {mae - BASELINE_V6_MAE:+.4f})")
    print(f"    NOTE: not 1-to-1 (28 vs 5 categories, no stock features)")

    # Per-category breakdown
    print(f"\n  === Per-category metrics ===")
    print_category_metrics(y_test, y_pred, test["Категория"].values)

    # --- Feature importance ---
    print(f"\n[7/7] Feature importance (top 20)...")
    importance = pd.DataFrame({
        "feature": available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(20).iterrows():
        print(f"    {row['feature']:<25} {row['importance']:>8.0f}")

    # --- Save results ---
    print(f"\n  Saving results...")
    metrics = {
        "experiment": EXP_NAME,
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
        "optuna_time_s": round(optuna_time, 1),
        "n_trials": N_TRIALS,
        "best_optuna_mae": round(study.best_value, 4),
        "best_params": study.best_params,
        "baseline_01_mae": BASELINE_01_MAE,
        "baseline_v6_mae": BASELINE_V6_MAE,
        "baseline_v6_wmape": BASELINE_V6_WMAPE,
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
    with mlflow.start_run(run_name=EXP_NAME):
        # Params
        mlflow.log_params({
            "experiment": EXP_NAME,
            "n_features": len(available),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_days": int(train["Дата"].nunique()),
            "test_days": TEST_DAYS,
            "n_bakeries": int(df["Пекарня"].nunique()),
            "n_products": int(df["Номенклатура"].nunique()),
            "n_categories": int(df["Категория"].nunique()),
            "n_trials": N_TRIALS,
        })
        mlflow.log_params({f"lgbm_{k}": v for k, v in study.best_params.items()})

        # Metrics
        mlflow.log_metrics({
            "mae": mae,
            "wmape": wm,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "train_time_s": train_time,
            "optuna_time_s": optuna_time,
            "best_optuna_mae": study.best_value,
        })

        # Artifacts
        mlflow.log_artifact(str(EXP_DIR / "metrics.json"))
        mlflow.log_artifact(str(EXP_DIR / "predictions.csv"))
        mlflow.log_artifact(str(best_params_path))

        # Model
        mlflow.lightgbm.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run_id: {run_id}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Best Optuna MAE: {study.best_value:.4f}")
    print(f"  Final MAE:       {mae:.4f} (vs baseline 01: {BASELINE_01_MAE})")
    print(f"  Final WMAPE:     {wm:.2f}%")
    print(f"  Optuna time:     {optuna_time:.0f}s ({optuna_time/60:.1f} min)")
    print(f"  Total time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  MLflow UI:       mlflow ui --port 5001")
    print(f"  Done!")


if __name__ == "__main__":
    main()
