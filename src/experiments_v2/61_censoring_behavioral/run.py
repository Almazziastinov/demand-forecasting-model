"""
Experiment 61: Censoring & behavioral features.

Adds new feature groups to baseline V3 (exp 60):
  A: Censoring  -- is_censored_lag1, lost_qty_lag1, pct_censored_7d
  B: Day-of-week -- sales_dow_mean, demand_dow_mean
  C: Trend & volatility -- demand_trend, cv_7d
  D: Stale ratio -- stale_ratio_lag1 (from raw checks, Freshness column)

Sequential evaluation: baseline -> +A -> +A+B -> +A+B+C -> +A+B+C+D
Each group evaluated for delta MAE vs baseline V3.

Baseline: exp 60 V3 (MAE 2.8816, Quantile P50, 58 features FEATURES_V3)

Input:  data/processed/daily_sales_8m_demand.csv
        data/raw/sales_hrs_all.csv (for Group D only)
Output: src/experiments_v2/61_censoring_behavioral/metrics.json

Usage:
  .venv/Scripts/python.exe src/experiments_v2/61_censoring_behavioral/run.py
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
    SALES_HRS_PATH,
    wmape, print_metrics, print_category_metrics,
    train_quantile, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent

BASELINE_V3_MAE = 2.8816


# ── Group A: Censoring features ──────────────────────────────────

def add_censoring_features(df):
    """Lagged censoring signals: was product out of stock yesterday?"""
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура"])

    # Lag-1: yesterday's values (no leakage)
    df["is_censored_lag1"] = grp["is_censored"].shift(1).fillna(0).astype(int)
    df["lost_qty_lag1"] = grp["lost_qty"].shift(1).fillna(0)

    # Rolling: % of censored days in last 7 (on lagged data)
    df["pct_censored_7d"] = (
        grp["is_censored"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean() * 100)
    ).fillna(0)

    return df


FEATURES_A = ["is_censored_lag1", "lost_qty_lag1", "pct_censored_7d"]


# ── Group B: Day-of-week averages ────────────────────────────────

def add_dow_features(df):
    """Average sales/demand for this (bakery, product, day_of_week).

    Uses expanding mean on shifted data to prevent leakage:
    for each row, the mean is computed from all previous same-DOW days.
    """
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура", "ДеньНедели"])

    df["sales_dow_mean"] = (
        grp["Продано"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    df["demand_dow_mean"] = (
        grp[DEMAND_TARGET]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    return df


FEATURES_B = ["sales_dow_mean", "demand_dow_mean"]


# ── Group C: Trend & volatility ──────────────────────────────────

def add_trend_features(df):
    """Derived from existing rolling stats (already lag-safe)."""
    df["demand_trend"] = df["demand_roll_mean7"] / (df["demand_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1)
    return df


FEATURES_C = ["demand_trend", "cv_7d"]


# ── Group D: Stale ratio ─────────────────────────────────────────

def build_stale_ratio(sales_hrs_path, df):
    """Parse raw checks in chunks to compute stale_ratio per (bakery, product, date).

    stale_ratio = qty sold as 'Vcherashnij' / total qty sold.
    Then lag by 1 day.
    """
    if not Path(sales_hrs_path).exists():
        print(f"    WARNING: {sales_hrs_path} not found, skipping stale_ratio")
        return df

    print("    Parsing raw checks for stale_ratio (chunked)...")
    chunk_size = 2_000_000
    usecols = ["Дата продажи", "Касса.Торговая точка", "Номенклатура",
               "Свежесть", "Кол-во", "Вид события по кассе"]

    agg_list = []
    for i, chunk in enumerate(pd.read_csv(
        str(sales_hrs_path), encoding="utf-8-sig",
        usecols=usecols, chunksize=chunk_size,
    )):
        sales = chunk[chunk["Вид события по кассе"] == "Продажа"].copy()
        sales["is_stale"] = (sales["Свежесть"] == "Вчерашний").astype(int)
        sales["stale_qty"] = sales["is_stale"] * sales["Кол-во"]

        agg = (sales.groupby(["Дата продажи", "Касса.Торговая точка", "Номенклатура"])
               .agg(total_qty=("Кол-во", "sum"), stale_qty=("stale_qty", "sum"))
               .reset_index())
        agg_list.append(agg)

        if (i + 1) % 5 == 0:
            print(f"      chunk {i + 1}...")

    stale = pd.concat(agg_list, ignore_index=True)
    stale = (stale.groupby(["Дата продажи", "Касса.Торговая точка", "Номенклатура"])
             .agg(total_qty=("total_qty", "sum"), stale_qty=("stale_qty", "sum"))
             .reset_index())

    stale["stale_ratio"] = stale["stale_qty"] / (stale["total_qty"] + 1e-8)
    stale.rename(columns={
        "Дата продажи": "Дата",
        "Касса.Торговая точка": "Пекарня",
    }, inplace=True)
    stale["Дата"] = pd.to_datetime(stale["Дата"], format="%d.%m.%Y", errors="coerce")
    stale = stale[["Дата", "Пекарня", "Номенклатура", "stale_ratio"]].dropna(subset=["Дата"])

    print(f"    Stale ratio: {len(stale):,} rows, "
          f"mean={stale['stale_ratio'].mean():.3f}, "
          f"nonzero={( stale['stale_ratio'] > 0).mean() * 100:.1f}%")

    # Merge
    df = df.merge(stale, on=["Дата", "Пекарня", "Номенклатура"], how="left")
    df["stale_ratio"] = df["stale_ratio"].fillna(0)

    # Lag by 1 day
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"])
    df["stale_ratio_lag1"] = (
        df.groupby(["Пекарня", "Номенклатура"])["stale_ratio"]
        .shift(1).fillna(0)
    )
    df.drop(columns=["stale_ratio"], inplace=True)

    return df


FEATURES_D = ["stale_ratio_lag1"]


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXPERIMENT 61: Censoring & Behavioral Features")
    print("  Baseline: exp 60 V3 (MAE 2.8816, Quantile P50)")
    print("=" * 60)
    t_start = time.time()

    # [1/5] Load
    print(f"\n[1/5] Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")

    # [2/5] Build features
    print(f"\n[2/5] Building feature groups...")

    print("  Group A: censoring features...")
    df = add_censoring_features(df)

    print("  Group B: day-of-week averages...")
    df = add_dow_features(df)

    print("  Group C: trend & volatility...")
    df = add_trend_features(df)

    print("  Group D: stale ratio (from raw checks)...")
    df = build_stale_ratio(str(SALES_HRS_PATH), df)

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
        ("baseline (V3)",     base_features),
        ("+A (censoring)",    base_features + FEATURES_A),
        ("+A+B (dow)",        base_features + FEATURES_A + FEATURES_B),
        ("+A+B+C (trend)",    base_features + FEATURES_A + FEATURES_B + FEATURES_C),
    ]

    # Only add Group D if stale_ratio was computed
    if "stale_ratio_lag1" in df.columns:
        variants.append(
            ("+A+B+C+D (stale)", base_features + FEATURES_A + FEATURES_B + FEATURES_C + FEATURES_D)
        )

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
            "delta_mae": round(delta, 4),
            "train_time_s": round(train_time, 1),
        }

        print(f"    MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}, R2={r2:.4f}")
        print(f"    Delta vs V3: {delta:+.4f} MAE")
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

    # Show new features highlighted
    new_feats = set(FEATURES_A + FEATURES_B + FEATURES_C + FEATURES_D)
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

    # Summary table
    print(f"\n  {'=' * 70}")
    print(f"  SUMMARY")
    print(f"  {'=' * 70}")
    print(f"  {'Variant':<25} {'Features':>4} {'MAE':>8} {'WMAPE':>8} {'Delta':>8}")
    print(f"  {'-' * 58}")
    for name, r in all_results.items():
        print(f"  {name:<25} {r['n_features']:>4} {r['mae']:>8.4f} "
              f"{r['wmape']:>7.2f}% {r['delta_mae']:>+8.4f}")

    # Save
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "61_censoring_behavioral",
        "target": DEMAND_TARGET,
        "baseline_v3_mae": BASELINE_V3_MAE,
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
    save_results(EXP_DIR, metrics)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
