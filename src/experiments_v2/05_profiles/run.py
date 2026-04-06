"""
Experiment 05: Product & Bakery profiles.

Add aggregate features computed from training data only (no leakage):
  - bakery_avg_sales     -- mean daily sales per bakery (scale of the store)
  - product_avg_sales    -- mean daily sales per product (popularity)
  - product_cv           -- coefficient of variation per product (stability)
  - bakery_x_product_avg -- mean daily sales per bakery x product pair (specificity)
  - category_avg_sales   -- mean daily sales per category
  - bakery_product_count -- number of unique products sold at bakery

Hypothesis: LightGBM encodes categorical IDs but doesn't directly know the
"magnitude" behind each ID.  Explicit profile features give the model scale
context and should improve predictions, especially for rare combinations.

Input:  data/processed/daily_sales_8m.csv
Output: src/experiments_v2/05_profiles/metrics.json
        src/experiments_v2/05_profiles/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/05_profiles/run.py
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
BASELINE_MAE = 2.2904
BASELINE_WMAPE = 25.91

# New profile features to add
PROFILE_FEATURES = [
    "bakery_avg_sales",
    "product_avg_sales",
    "product_cv",
    "bakery_x_product_avg",
    "category_avg_sales",
    "bakery_product_count",
]


def build_profiles(train_df):
    """Compute profile features from training data only."""
    profiles = {}

    # 1. bakery_avg_sales: mean sales per bakery
    bakery_avg = train_df.groupby("Пекарня")[TARGET].mean()
    profiles["bakery_avg_sales"] = bakery_avg.to_dict()

    # 2. product_avg_sales: mean sales per product
    product_avg = train_df.groupby("Номенклатура")[TARGET].mean()
    profiles["product_avg_sales"] = product_avg.to_dict()

    # 3. product_cv: std / mean per product (coefficient of variation)
    product_stats = train_df.groupby("Номенклатура")[TARGET].agg(["mean", "std"])
    product_stats["cv"] = product_stats["std"] / (product_stats["mean"] + 1e-8)
    profiles["product_cv"] = product_stats["cv"].to_dict()

    # 4. bakery_x_product_avg: mean sales per bakery x product pair
    bp_avg = train_df.groupby(["Пекарня", "Номенклатура"])[TARGET].mean()
    profiles["bakery_x_product_avg"] = bp_avg.to_dict()

    # 5. category_avg_sales: mean sales per category
    cat_avg = train_df.groupby("Категория")[TARGET].mean()
    profiles["category_avg_sales"] = cat_avg.to_dict()

    # 6. bakery_product_count: number of unique products sold at each bakery
    bp_count = train_df.groupby("Пекарня")["Номенклатура"].nunique()
    profiles["bakery_product_count"] = bp_count.to_dict()

    return profiles


def apply_profiles(df, profiles):
    """Map profile features onto DataFrame."""
    df["bakery_avg_sales"] = df["Пекарня"].map(profiles["bakery_avg_sales"])
    df["product_avg_sales"] = df["Номенклатура"].map(profiles["product_avg_sales"])
    df["product_cv"] = df["Номенклатура"].map(profiles["product_cv"])

    # bakery x product pair
    bp_keys = list(zip(df["Пекарня"], df["Номенклатура"]))
    bp_map = profiles["bakery_x_product_avg"]
    df["bakery_x_product_avg"] = [bp_map.get(k, np.nan) for k in bp_keys]

    df["category_avg_sales"] = df["Категория"].map(profiles["category_avg_sales"])
    df["bakery_product_count"] = df["Пекарня"].map(profiles["bakery_product_count"])

    # Fill NaN for unseen combinations in test
    for col in PROFILE_FEATURES:
        n_null = df[col].isna().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"    {col}: filled {n_null} NaN with median {median_val:.4f}")

    return df


def main():
    print("=" * 60)
    print("  EXPERIMENT 05: Product & Bakery Profiles")
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

    # --- Train/test split (BEFORE computing profiles!) ---
    print(f"\n[2/7] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days "
          f"({train['Дата'].min().date()} -- {train['Дата'].max().date()})")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days "
          f"({test['Дата'].min().date()} -- {test['Дата'].max().date()})")

    # --- Build profiles from TRAIN only ---
    print(f"\n[3/7] Building profiles from training data...")
    profiles = build_profiles(train)

    print(f"  bakery_avg_sales:     {len(profiles['bakery_avg_sales'])} bakeries")
    print(f"  product_avg_sales:    {len(profiles['product_avg_sales'])} products")
    print(f"  product_cv:           {len(profiles['product_cv'])} products")
    print(f"  bakery_x_product_avg: {len(profiles['bakery_x_product_avg'])} pairs")
    print(f"  category_avg_sales:   {len(profiles['category_avg_sales'])} categories")
    print(f"  bakery_product_count: {len(profiles['bakery_product_count'])} bakeries")

    # --- Apply profiles to both train and test ---
    print(f"\n[4/7] Applying profiles...")
    print(f"  Train:")
    train = apply_profiles(train, profiles)
    print(f"  Test:")
    test = apply_profiles(test, profiles)

    # Profile feature stats
    print(f"\n  Profile feature stats (train):")
    for col in PROFILE_FEATURES:
        vals = train[col]
        print(f"    {col:<25} mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    # --- Select features ---
    print(f"\n[5/7] Selecting features...")
    all_features = FEATURES_V2 + PROFILE_FEATURES
    available = [f for f in all_features if f in train.columns]
    missing = [f for f in all_features if f not in train.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available)} features ({len(FEATURES_V2)} base + {len(PROFILE_FEATURES)} profile)")

    # Convert categorical columns
    for col in CATEGORICAL_COLS_V2:
        if col in train.columns:
            train[col] = train[col].astype("category")
            test[col] = test[col].astype("category")

    X_train = train[available]
    y_train = train[TARGET]
    X_test = test[available]
    y_test = test[TARGET]

    # --- Train ---
    print(f"\n[6/7] Training LightGBM (v6 params)...")
    print(f"  Params: n_estimators={MODEL_PARAMS['n_estimators']}, "
          f"lr={MODEL_PARAMS['learning_rate']:.4f}, "
          f"num_leaves={MODEL_PARAMS['num_leaves']}, "
          f"max_depth={MODEL_PARAMS['max_depth']}")

    t_train = time.time()
    model = train_lgbm(X_train, y_train)
    train_time = time.time() - t_train
    print(f"  Training time: {train_time:.0f}s")

    # --- Evaluate ---
    print(f"\n[7/7] Evaluation...")
    y_pred = predict_clipped(model, X_test)

    mae = mean_absolute_error(y_test, y_pred)
    wm = wmape(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(np.asarray(y_test) - y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  === RESULTS ===")
    print_metrics("05_profiles", y_test, y_pred, baseline_mae=BASELINE_MAE)
    print(f"    RMSE  = {rmse:.4f}")
    print(f"    R2    = {r2:.4f}")

    # Delta vs baseline 01
    print(f"\n  === vs 01_baseline_8m ===")
    print(f"    MAE:   {mae:.4f}  vs  {BASELINE_MAE:.4f}  (delta: {mae - BASELINE_MAE:+.4f})")
    print(f"    WMAPE: {wm:.2f}%  vs  {BASELINE_WMAPE:.2f}%  (delta: {wm - BASELINE_WMAPE:+.2f}%)")

    # Per-category breakdown
    print(f"\n  === Per-category metrics ===")
    print_category_metrics(y_test, y_pred, test["Категория"].values)

    # --- Feature importance ---
    print(f"\n  Feature importance (top 20)...")
    importance = pd.DataFrame({
        "feature": available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(20).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>8.0f}")

    # Profile features importance
    print(f"\n  Profile features importance:")
    profile_imp = importance[importance["feature"].isin(PROFILE_FEATURES)]
    for _, row in profile_imp.iterrows():
        rank = importance.index.tolist().index(row.name) + 1
        print(f"    #{rank:<3} {row['feature']:<30} {row['importance']:>8.0f}")

    # --- Save results ---
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "05_profiles",
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
        "n_profile_features": len(PROFILE_FEATURES),
        "profile_features": PROFILE_FEATURES,
        "n_bakeries": int(df["Пекарня"].nunique()),
        "n_products": int(df["Номенклатура"].nunique()),
        "n_categories": int(df["Категория"].nunique()),
        "train_time_s": round(train_time, 1),
        "baseline_01_mae": BASELINE_MAE,
        "baseline_01_wmape": BASELINE_WMAPE,
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
    with mlflow.start_run(run_name="05_profiles"):
        mlflow.log_params({
            "experiment": "05_profiles",
            "n_features": len(available),
            "n_profile_features": len(PROFILE_FEATURES),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_days": int(train["Дата"].nunique()),
            "test_days": TEST_DAYS,
            "n_bakeries": int(df["Пекарня"].nunique()),
            "n_products": int(df["Номенклатура"].nunique()),
            "n_categories": int(df["Категория"].nunique()),
        })
        mlflow.log_params({f"lgbm_{k}": v for k, v in MODEL_PARAMS.items()})

        mlflow.log_metrics({
            "mae": mae,
            "wmape": wm,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "train_time_s": train_time,
        })

        mlflow.log_artifact(str(EXP_DIR / "metrics.json"))
        mlflow.log_artifact(str(EXP_DIR / "predictions.csv"))
        mlflow.lightgbm.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run_id: {run_id}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Done!")


if __name__ == "__main__":
    main()
