"""
Experiment 62: Assortment availability (substitution effect).

Hypothesis: When products disappear from a bakery's assortment,
demand transfers to remaining products. Adding features about
assortment size helps the model predict these demand spikes.

Feature groups:
  A: items_in_bakery_today -- number of unique products available
  B: items_in_bakery_lag1  -- yesterday's assortment size
  C: items_change          -- change in assortment size vs yesterday
  D: items_in_category_today -- products available in same category
  E: category_items_change -- category-level assortment change

Baseline: exp 60 V3 (MAE 2.8816, Quantile P50, 58 features)
Best so far: exp 61 (MAE 2.8601)

Input:  data/processed/daily_sales_8m_demand.csv
Output: src/experiments_v2/62_assortment_availability/metrics.json

Usage:
  .venv/Scripts/python.exe src/experiments_v2/62_assortment_availability/run.py
"""

import sys
import os
import time
from pathlib import Path

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
EXP61_MAE = 2.8601


# -- Feature engineering ------------------------------------------------

def add_assortment_features(df):
    """Add assortment availability features (bakery-level and category-level)."""
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()

    # A: Number of unique products in bakery today
    df["items_in_bakery_today"] = (
        df.groupby(["Пекарня", "Дата"])["Номенклатура"]
        .transform("nunique")
    )

    # B: Yesterday's assortment size (lag 1 of bakery-day level)
    # Build bakery-day lookup first, then merge back
    bakery_day = (
        df.groupby(["Пекарня", "Дата"])["Номенклатура"]
        .nunique()
        .reset_index(name="__bakery_items")
        .sort_values(["Пекарня", "Дата"])
    )
    bakery_day["items_in_bakery_lag1"] = (
        bakery_day.groupby("Пекарня")["__bakery_items"].shift(1)
    )
    bakery_day["items_change"] = (
        bakery_day["__bakery_items"] - bakery_day["items_in_bakery_lag1"]
    )
    bakery_day.drop(columns=["__bakery_items"], inplace=True)
    df = df.merge(bakery_day, on=["Пекарня", "Дата"], how="left")

    # D: Number of unique products in same category today
    df["items_in_category_today"] = (
        df.groupby(["Пекарня", "Категория", "Дата"])["Номенклатура"]
        .transform("nunique")
    )

    # E: Category-level assortment change vs yesterday
    cat_day = (
        df.groupby(["Пекарня", "Категория", "Дата"])["Номенклатура"]
        .nunique()
        .reset_index(name="__cat_items")
        .sort_values(["Пекарня", "Категория", "Дата"])
    )
    cat_day["__cat_items_lag1"] = (
        cat_day.groupby(["Пекарня", "Категория"])["__cat_items"].shift(1)
    )
    cat_day["category_items_change"] = (
        cat_day["__cat_items"] - cat_day["__cat_items_lag1"]
    )
    cat_day.drop(columns=["__cat_items", "__cat_items_lag1"], inplace=True)
    df = df.merge(cat_day, on=["Пекарня", "Категория", "Дата"], how="left")

    # Fill NaN (first day has no lag)
    for col in FEATURES_ASSORTMENT:
        df[col] = df[col].fillna(0)

    return df


FEATURES_ASSORTMENT = [
    "items_in_bakery_today",
    "items_in_bakery_lag1",
    "items_change",
    "items_in_category_today",
    "category_items_change",
]


# -- Main ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EXPERIMENT 62: Assortment Availability")
    print("  Hypothesis: substitution effect when products disappear")
    print("  Baseline: exp 60 V3 (MAE 2.8816)")
    print("=" * 60)
    t_start = time.time()

    # [1/5] Load
    print(f"\n[1/5] Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")

    # [2/5] Build features
    print(f"\n[2/5] Building assortment features...")
    df = add_assortment_features(df)

    # Descriptive stats for new features
    print(f"\n  Feature stats:")
    for feat in FEATURES_ASSORTMENT:
        s = df[feat]
        print(f"    {feat:<30} mean={s.mean():.2f}, std={s.std():.2f}, "
              f"min={s.min():.0f}, max={s.max():.0f}")

    # [3/5] Categoricals
    print(f"\n[3/5] Converting categoricals...")
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # [4/5] Split
    print(f"\n[4/5] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    y_train = train[DEMAND_TARGET]
    y_test = test[DEMAND_TARGET].values

    # [5/5] Sequential evaluation
    print(f"\n[5/5] Training & evaluating...")

    base_features = [f for f in FEATURES_V3 if f in df.columns]

    variants = [
        ("baseline (V3)",            base_features),
        ("+bakery assortment",       base_features + ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"]),
        ("+category assortment",     base_features + ["items_in_category_today", "category_items_change"]),
        ("+all assortment",          base_features + FEATURES_ASSORTMENT),
    ]

    all_results = {}

    for name, feats in variants:
        available = [f for f in feats if f in df.columns]
        print(f"\n  --- {name} ({len(available)} features) ---")

        t0 = time.time()
        model = train_quantile(train[available], y_train, alpha=0.5)
        train_time = time.time() - t0

        y_pred = predict_clipped(model, test[available])
        mae = mean_absolute_error(y_test, y_pred)
        wm = wmape(y_test, y_pred)
        bias = float(np.mean(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)
        delta = mae - BASELINE_V3_MAE

        all_results[name] = {
            "mae": round(mae, 4),
            "wmape": round(wm, 2),
            "bias": round(bias, 4),
            "r2": round(r2, 4),
            "n_features": len(available),
            "delta_vs_v3": round(delta, 4),
            "delta_vs_exp61": round(mae - EXP61_MAE, 4),
            "train_time_s": round(train_time, 1),
        }

        print(f"    MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}, R2={r2:.4f}")
        print(f"    Delta vs V3: {delta:+.4f}, vs exp61: {mae - EXP61_MAE:+.4f}")
        print(f"    Train time: {train_time:.0f}s")

        # High demand breakdown
        for threshold in [15, 50, 100]:
            mask = y_test >= threshold
            if mask.sum() > 0:
                mae_h = mean_absolute_error(y_test[mask], y_pred[mask])
                bias_h = float(np.mean(y_test[mask] - y_pred[mask]))
                print(f"    demand>={threshold}: MAE={mae_h:.2f}, Bias={bias_h:+.2f}, N={mask.sum()}")

    # Feature importance for best variant
    best_name = min(all_results, key=lambda k: all_results[k]["mae"])
    best_feats = dict(variants)[best_name]
    best_available = [f for f in best_feats if f in df.columns]

    print(f"\n  === Feature importance ({best_name}) ===")
    best_model = train_quantile(train[best_available], y_train, alpha=0.5)
    importance = pd.DataFrame({
        "feature": best_available,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    new_feats = set(FEATURES_ASSORTMENT)
    for _, row in importance.head(25).iterrows():
        marker = " <<<" if row["feature"] in new_feats else ""
        print(f"    {row['feature']:<30} {row['importance']:>6.0f}{marker}")

    print(f"\n  New feature ranks:")
    for rank, (_, row) in enumerate(importance.iterrows(), 1):
        if row["feature"] in new_feats:
            print(f"    #{rank}: {row['feature']} ({row['importance']:.0f})")

    # Per-category for best
    best_pred = predict_clipped(best_model, test[best_available])
    print(f"\n  === Per-category ({best_name}) ===")
    print_category_metrics(y_test, best_pred, test["Категория"].values)

    # Summary
    print(f"\n  {'=' * 75}")
    print(f"  SUMMARY")
    print(f"  {'=' * 75}")
    print(f"  {'Variant':<25} {'Feats':>5} {'MAE':>8} {'WMAPE':>8} {'vs V3':>8} {'vs 61':>8}")
    print(f"  {'-' * 65}")
    for name, r in all_results.items():
        print(f"  {name:<25} {r['n_features']:>5} {r['mae']:>8.4f} "
              f"{r['wmape']:>7.2f}% {r['delta_vs_v3']:>+8.4f} {r['delta_vs_exp61']:>+8.4f}")

    # Save
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "62_assortment_availability",
        "target": DEMAND_TARGET,
        "hypothesis": "substitution effect when products disappear from assortment",
        "baseline_v3_mae": BASELINE_V3_MAE,
        "exp61_mae": EXP61_MAE,
        "results": all_results,
        "best_variant": best_name,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_importance_new": [
            {"feature": row["feature"], "importance": int(row["importance"]),
             "rank": i + 1}
            for i, (_, row) in enumerate(importance.iterrows())
            if row["feature"] in new_feats
        ],
    }
    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact": y_test,
        "pred": np.round(best_pred, 2),
        "abs_error": np.round(np.abs(y_test - best_pred), 2),
        "model": best_name,
    })
    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
