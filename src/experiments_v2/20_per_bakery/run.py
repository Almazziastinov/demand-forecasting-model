"""
Experiment 20: Per-bakery models.

Train one model per bakery (or top-N) vs global baseline.
Compare MAE for specific bakeries.

Baseline: exp 60 V3 (MAE 2.8816, Quantile P50, 58 features FEATURES_V3)

Input:  data/processed/daily_sales_8m_demand.csv
Output: src/experiments_v2/20_per_bakery/metrics.json

Usage:
  .venv/Scripts/python.exe src/experiments_v2/20_per_bakery/run.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, print_metrics, print_category_metrics,
    train_quantile, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent

BASELINE_V3_MAE = 2.8816


def load_data():
    """Load demand data with features."""
    print(f"Loading {DEMAND_8M_PATH}...")
    df = pd.read_csv(DEMAND_8M_PATH, encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"], format="mixed", dayfirst=False)
    print(f"  {len(df):,} rows, {df['Пекарня'].nunique()} bakeries, {df['Номенклатура'].nunique()} products")
    return df


def add_features(df):
    """Add exp 61 features: censoring, DOW, trend."""
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура"])

    # Group A: Censoring (lag-safe)
    df["is_censored_lag1"] = grp["is_censored"].shift(1).fillna(0).astype(int)
    df["lost_qty_lag1"] = grp["lost_qty"].shift(1).fillna(0)
    df["pct_censored_7d"] = (
        grp["is_censored"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean() * 100)
    ).fillna(0)

    # Group B: DOW means
    grp_dow = df.groupby(["Пекарня", "Номенклатура", "ДеньНедели"])
    df["sales_dow_mean"] = (
        grp_dow["Продано"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)
    df["demand_dow_mean"] = (
        grp_dow[DEMAND_TARGET]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    # Group C: Trend & volatility
    df["demand_trend"] = df["demand_roll_mean7"] / (df["demand_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1)

    return df


# Full feature set with exp 61 features (66 features)
FEATURES_EXP61 = FEATURES_V3 + [
    "is_censored_lag1", "lost_qty_lag1", "pct_censored_7d",
    "sales_dow_mean", "demand_dow_mean",
    "demand_trend", "cv_7d",
    "stale_ratio_lag1",
]


def train_global_model(df, features):
    """Train one global model on all data (baseline)."""
    train = df[df["is_train"] == 1].copy()
    test = df[df["is_train"] == 0].copy()

    X_train = train[features].copy()
    y_train = train[DEMAND_TARGET]
    X_test = test[features].copy()
    y_test = test[DEMAND_TARGET]

    for col in CATEGORICAL_COLS_V2:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    print(f"  Training global model: {len(X_train):,} train, {len(X_test):,} test")
    model = train_quantile(X_train, y_train)

    y_pred = predict_clipped(model, X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  Global model MAE: {mae:.4f}")
    return model, mae


def train_per_bakery_models(df, features, top_n=10):
    """Train separate model per bakery for top-N bakeries."""
    bakery_counts = df.groupby("Пекарня").size().sort_values(ascending=False)
    top_bakeries = bakery_counts.head(top_n).index.tolist()

    print(f"\n  Training per-bakery models for top {top_n} bakeries:")
    print(f"    {top_bakeries[:3]}... ({len(top_bakeries)} total)")

    train = df[df["is_train"] == 1].copy()
    test = df[df["is_train"] == 0].copy()

    results = {}
    bakery_maes = {}

    for bakery in top_bakeries:
        train_b = train[train["Пекарня"] == bakery].copy()
        test_b = test[test["Пекарня"] == bakery].copy()

        if len(train_b) < 100 or len(test_b) < 20:
            print(f"    {bakery}: Skipping (insufficient data: {len(train_b)} train, {len(test_b)} test)")
            continue

        X_train = train_b[features].copy()
        y_train = train_b[DEMAND_TARGET]
        X_test = test_b[features].copy()
        y_test = test_b[DEMAND_TARGET]

        for col in CATEGORICAL_COLS_V2:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        try:
            model = train_quantile(X_train, y_train)
            y_pred = predict_clipped(model, X_test)
            mae = mean_absolute_error(y_test, y_pred)
            bakery_maes[bakery] = mae
            print(f"    {bakery}: MAE = {mae:.4f} ({len(train_b):,} train, {len(test_b):,} test)")
        except Exception as e:
            print(f"    {bakery}: Error - {e}")

    return bakery_maes


def compare_for_bakery(df, bakery, features):
    """Compare global vs per-bakery model for a specific bakery."""
    train = df[df["is_train"] == 1].copy()
    test = df[df["is_train"] == 0].copy()

    train_b = train[train["Пекарня"] == bakery].copy()
    test_b = test[test["Пекарня"] == bakery].copy()

    print(f"\n  Bakery: {bakery}")
    print(f"    Train: {len(train_b):,}, Test: {len(test_b):,}")

    # Global model predictions for this bakery
    X_train_all = train[features].copy()
    y_train_all = train[DEMAND_TARGET]
    X_test_all = train[train["Пекарня"] == bakery][features].copy()

    for col in CATEGORICAL_COLS_V2:
        X_train_all[col] = X_train_all[col].astype("category")
        X_test_all[col] = X_test_all[col].astype("category")

    global_model = train_quantile(X_train_all, y_train_all)
    global_pred = predict_clipped(global_model, X_test_all)
    global_mae = mean_absolute_error(test_b[DEMAND_TARGET], global_pred)

    # Per-bakery model
    X_train_b = train_b[features].copy()
    y_train_b = train_b[DEMAND_TARGET]
    X_test_b = test_b[features].copy()

    for col in CATEGORICAL_COLS_V2:
        X_train_b[col] = X_train_b[col].astype("category")
        X_test_b[col] = X_test_b[col].astype("category")

    local_model = train_quantile(X_train_b, y_train_b)
    local_pred = predict_clipped(local_model, X_test_b)
    local_mae = mean_absolute_error(test_b[DEMAND_TARGET], local_pred)

    print(f"    Global MAE:  {global_mae:.4f}")
    print(f"    Local MAE:    {local_mae:.4f}")
    print(f"    Delta:       {local_mae - global_mae:+.4f}")

    return {
        "bakery": bakery,
        "train_rows": len(train_b),
        "test_rows": len(test_b),
        "global_mae": global_mae,
        "local_mae": local_mae,
        "delta": local_mae - global_mae,
    }


def main():
    print("=" * 60)
    print("Experiment 20: Per-bakery models")
    print("=" * 60)

    start = time.time()

    # Load data
    df = load_data()

    # Add features
    df = add_features(df)

    # Check for stale_ratio (optional, may not exist)
    if "stale_ratio_lag1" not in df.columns:
        df["stale_ratio_lag1"] = 0
        print("  Note: stale_ratio_lag1 not found, using zeros")

    # Split train/test
    max_date = df["Дата"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS)
    df["is_train"] = (df["Дата"] < test_start).astype(int)

    print(f"\n  Train: {df['is_train'].sum():,}, Test: {(df['is_train'] == 0).sum():,}")
    print(f"  Test period: {test_start.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    features = FEATURES_EXP61

    # Top 10 bakeries by row count
    bakery_counts = df.groupby("Пекарня").size().sort_values(ascending=False)
    top_10 = bakery_counts.head(10).index.tolist()

    # Global baseline for all data
    print("\n" + "=" * 40)
    print("Global model (baseline)")
    print("=" * 40)
    global_model, global_mae = train_global_model(df, features)

    # Compare for each top bakery
    print("\n" + "=" * 40)
    print("Per-bakery comparison")
    print("=" * 40)

    comparisons = []
    for bakery in top_10:
        result = compare_for_bakery(df, bakery, features)
        comparisons.append(result)

    # Summary
    print("\n" + "=" * 40)
    print("Summary")
    print("=" * 40)

    global_maes = [c["global_mae"] for c in comparisons]
    local_maes = [c["local_mae"] for c in comparisons]
    deltas = [c["delta"] for c in comparisons]

    print(f"  Global MAE (avg top-10):  {np.mean(global_maes):.4f}")
    print(f"  Local MAE (avg top-10):   {np.mean(local_maes):.4f}")
    print(f"  Delta (avg):               {np.mean(deltas):+.4f}")

    better = sum(1 for d in deltas if d < 0)
    worse = sum(1 for d in deltas if d > 0)
    print(f"  Better: {better}, Worse: {worse}")

    # Save results
    metrics = {
        "experiment": 20,
        "name": "Per-bakery models",
        "baseline_mae": BASELINE_V3_MAE,
        "global_mae": global_mae,
        "top_bakeries": top_10,
        "comparisons": comparisons,
        "summary": {
            "avg_global_mae_top10": float(np.mean(global_maes)),
            "avg_local_mae_top10": float(np.mean(local_maes)),
            "avg_delta": float(np.mean(deltas)),
            "better_count": better,
            "worse_count": worse,
        },
    }

    save_results(EXP_DIR, metrics)

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
