"""
Experiment 03: Demand Target (demand profiles).

Train LightGBM on corrected demand target (Spros = Prodano + lost_qty)
instead of raw sales (Prodano). Demand is estimated via cumulative
hourly profiles from full days (see build_demand_profiles.py).

Input:  data/processed/daily_sales_8m_demand.csv
Output: src/experiments_v2/03_demand_target/metrics.json
        src/experiments_v2/03_demand_target/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/03_demand_target/run.py
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
from sklearn.metrics import mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    FEATURES_V2, CATEGORICAL_COLS_V2,
    print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
DEMAND_CSV = Path(ROOT) / "data" / "processed" / "daily_sales_8m_demand.csv"

DEMAND_TARGET = "Спрос"

# Baseline from experiment 01
BASELINE_01_MAE = 2.29
BASELINE_01_WMAPE = 25.9

# Features: FEATURES_V2 + demand lags/rolling
DEMAND_FEATURES = [
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
    "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30",
    "demand_roll_std7",
]
ALL_FEATURES = FEATURES_V2 + DEMAND_FEATURES


def main():
    print("=" * 60)
    print("  EXPERIMENT 03: Demand Target (demand profiles)")
    print("=" * 60)
    t_start = time.time()

    # --- Load data ---
    print(f"\n[1/7] Loading data from {DEMAND_CSV}...")
    if not DEMAND_CSV.exists():
        print(f"  ERROR: {DEMAND_CSV} not found!")
        print("  Run build_demand_profiles.py first.")
        return
    df = pd.read_csv(str(DEMAND_CSV), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    print(f"  Days: {df['Дата'].nunique()}, Bakeries: {df['Пекарня'].nunique()}, "
          f"Products: {df['Номенклатура'].nunique()}, Categories: {df['Категория'].nunique()}")

    # Demand stats
    mean_sold = df[TARGET].mean()
    mean_demand = df[DEMAND_TARGET].mean()
    uplift_pct = (mean_demand - mean_sold) / mean_sold * 100
    censored_pct = 100 * df["is_censored"].mean()
    mean_lost = df["lost_qty"].mean()

    print("\n  Demand stats:")
    print(f"    mean({TARGET}):     {mean_sold:.4f}")
    print(f"    mean({DEMAND_TARGET}):      {mean_demand:.4f}")
    print(f"    Demand uplift:    {uplift_pct:+.2f}%")
    print(f"    % censored:       {censored_pct:.1f}%")
    print(f"    mean(lost_qty):   {mean_lost:.4f}")

    # --- Select features ---
    print("\n[2/7] Selecting features...")
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available)} of {len(ALL_FEATURES)} features "
          f"({len(FEATURES_V2)} base + {len([f for f in DEMAND_FEATURES if f in df.columns])} demand)")

    # Convert categorical columns
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
    y_train_demand = train[DEMAND_TARGET]
    X_test = test[available]
    y_test_demand = test[DEMAND_TARGET]
    y_test_sold = test[TARGET]

    # --- Train on demand target ---
    print(f"\n[4/7] Training LightGBM on '{DEMAND_TARGET}' target...")
    print(f"  Params: n_estimators={MODEL_PARAMS['n_estimators']}, "
          f"lr={MODEL_PARAMS['learning_rate']:.4f}, "
          f"num_leaves={MODEL_PARAMS['num_leaves']}, "
          f"max_depth={MODEL_PARAMS['max_depth']}")

    t_train = time.time()
    model = train_lgbm(X_train, y_train_demand)
    train_time = time.time() - t_train
    print(f"  Training time: {train_time:.0f}s")

    # --- Evaluate ---
    print("\n[5/7] Evaluation...")
    y_pred = predict_clipped(model, X_test)

    # Metrics vs DEMAND (how well we predict true demand)
    print(f"\n  === vs {DEMAND_TARGET} (demand accuracy) ===")
    mae_demand, wm_demand, bias_demand = print_metrics(
        f"03_demand (vs {DEMAND_TARGET})", y_test_demand, y_pred
    )
    rmse_demand = np.sqrt(mean_squared_error(y_test_demand, y_pred))
    r2_demand = r2_score(y_test_demand, y_pred)
    print(f"    RMSE  = {rmse_demand:.4f}")
    print(f"    R2    = {r2_demand:.4f}")

    # Metrics vs SOLD (for comparison with baseline-01)
    print(f"\n  === vs {TARGET} (sales comparison with baseline) ===")
    mae_sold, wm_sold, bias_sold = print_metrics(
        f"03_demand (vs {TARGET})", y_test_sold, y_pred, baseline_mae=BASELINE_01_MAE
    )
    rmse_sold = np.sqrt(mean_squared_error(y_test_sold, y_pred))
    r2_sold = r2_score(y_test_sold, y_pred)
    print(f"    RMSE  = {rmse_sold:.4f}")
    print(f"    R2    = {r2_sold:.4f}")

    # Delta vs baseline-01
    print(f"\n  === vs baseline-01 (MAE {BASELINE_01_MAE}, WMAPE {BASELINE_01_WMAPE}%) ===")
    print(f"    MAE (vs {TARGET}):   {mae_sold:.4f}  vs  {BASELINE_01_MAE:.4f}  "
          f"(delta: {mae_sold - BASELINE_01_MAE:+.4f})")
    print(f"    WMAPE (vs {TARGET}): {wm_sold:.2f}%  vs  {BASELINE_01_WMAPE:.1f}%  "
          f"(delta: {wm_sold - BASELINE_01_WMAPE:+.2f}%)")
    print(f"    NOTE: model trained on {DEMAND_TARGET}, evaluated on {TARGET}")

    # Test censored stats
    test_censored = test["is_censored"].mean() * 100
    test_lost = test[test["is_censored"] == 1]["lost_qty"].mean()
    print("\n  Test set demand stats:")
    print(f"    % censored in test: {test_censored:.1f}%")
    print(f"    Mean lost_qty (censored): {test_lost:.2f}")

    # Per-category breakdown
    print(f"\n  === Per-category metrics (vs {TARGET}) ===")
    print_category_metrics(y_test_sold, y_pred, test["Категория"].values)

    # --- Feature importance ---
    print("\n[6/7] Feature importance (top 20)...")
    importance = pd.DataFrame({
        "feature": available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(20).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>8.0f}")

    # Highlight demand features
    demand_imp = importance[importance["feature"].str.startswith("demand_")]
    sales_imp = importance[importance["feature"].str.startswith("sales_")]
    print(f"\n  Demand features total importance: {demand_imp['importance'].sum():.0f}")
    print(f"  Sales features total importance:  {sales_imp['importance'].sum():.0f}")

    # --- Save results ---
    print("\n[7/7] Saving results...")
    metrics = {
        "experiment": "03_demand_target",
        "target": DEMAND_TARGET,
        # Demand metrics
        "mae_demand": round(mae_demand, 4),
        "wmape_demand": round(wm_demand, 2),
        "rmse_demand": round(rmse_demand, 4),
        "bias_demand": round(bias_demand, 4),
        "r2_demand": round(r2_demand, 4),
        # Sales metrics (comparable with baseline)
        "mae_sold": round(mae_sold, 4),
        "wmape_sold": round(wm_sold, 2),
        "rmse_sold": round(rmse_sold, 4),
        "bias_sold": round(bias_sold, 4),
        "r2_sold": round(r2_sold, 4),
        # Demand stats
        "demand_uplift_pct": round(uplift_pct, 2),
        "censored_pct": round(censored_pct, 1),
        "mean_lost_qty": round(mean_lost, 4),
        "mean_sold": round(mean_sold, 4),
        "mean_demand": round(mean_demand, 4),
        # Dataset
        "train_rows": len(train),
        "test_rows": len(test),
        "train_days": int(train["Дата"].nunique()),
        "test_days": int(test["Дата"].nunique()),
        "n_features": len(available),
        "n_demand_features": len([f for f in DEMAND_FEATURES if f in df.columns]),
        "n_bakeries": int(df["Пекарня"].nunique()),
        "n_products": int(df["Номенклатура"].nunique()),
        "n_categories": int(df["Категория"].nunique()),
        "train_time_s": round(train_time, 1),
        # Baselines
        "baseline_01_mae": BASELINE_01_MAE,
        "baseline_01_wmape": BASELINE_01_WMAPE,
        "feature_importance_top10": importance.head(10)[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_sold": y_test_sold.values,
        "fact_demand": y_test_demand.values,
        "pred": np.round(y_pred, 2),
        "is_censored": test["is_censored"].values,
        "lost_qty": test["lost_qty"].values,
        "abs_error_sold": np.round(np.abs(y_test_sold.values - y_pred), 2),
        "abs_error_demand": np.round(np.abs(y_test_demand.values - y_pred), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    # --- MLflow ---
    print("\n  Logging to MLflow...")
    mlflow.set_experiment("experiments_v2")
    with mlflow.start_run(run_name="03_demand_target"):
        mlflow.log_params({
            "experiment": "03_demand_target",
            "target": DEMAND_TARGET,
            "n_features": len(available),
            "n_demand_features": len([f for f in DEMAND_FEATURES if f in df.columns]),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_days": int(train["Дата"].nunique()),
            "test_days": TEST_DAYS,
            "n_bakeries": int(df["Пекарня"].nunique()),
            "n_products": int(df["Номенклатура"].nunique()),
            "n_categories": int(df["Категория"].nunique()),
            "demand_uplift_pct": round(uplift_pct, 2),
            "censored_pct": round(censored_pct, 1),
        })
        mlflow.log_params({f"lgbm_{k}": v for k, v in MODEL_PARAMS.items()})

        mlflow.log_metrics({
            "mae_demand": mae_demand,
            "wmape_demand": wm_demand,
            "mae_sold": mae_sold,
            "wmape_sold": wm_sold,
            "rmse_demand": rmse_demand,
            "rmse_sold": rmse_sold,
            "bias_demand": bias_demand,
            "bias_sold": bias_sold,
            "r2_demand": r2_demand,
            "r2_sold": r2_sold,
            "train_time_s": train_time,
        })

        mlflow.log_artifact(str(EXP_DIR / "metrics.json"))
        mlflow.log_artifact(str(EXP_DIR / "predictions.csv"))
        mlflow.lightgbm.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run_id: {run_id}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  MLflow UI: mlflow ui --port 5001")
    print("  Done!")


if __name__ == "__main__":
    main()
