"""
Experiment 63: Combined exp 61 + exp 62 features.

Combines:
  - Exp 61: censoring (is_censored_lag1, lost_qty_lag1, pct_censored_7d),
            DOW averages (sales_dow_mean, demand_dow_mean),
            trend/volatility (demand_trend, cv_7d),
            stale ratio (stale_ratio_lag1)
  - Exp 62: bakery assortment (items_in_bakery_today, items_in_bakery_lag1, items_change)

Baseline: exp 60 V3 (MAE 2.8816)
Exp 61:   MAE 2.8601 (+A+B+C+D)
Exp 62:   MAE 2.8719 (+bakery assortment)

Input:  data/processed/daily_sales_8m_demand.csv
        data/raw/sales_hrs_all.csv (for stale ratio)
Output: src/experiments_v2/63_combined_61_62/metrics.json

Usage:
  .venv\\Scripts\\python.exe src/experiments_v2/63_combined_61_62/run.py
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
EXP61_MAE = 2.8601
EXP62_MAE = 2.8719


# -- Exp 61 features (copied from 61) ------------------------------------

def add_censoring_features(df):
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура"])
    df["is_censored_lag1"] = grp["is_censored"].shift(1).fillna(0).astype(int)
    df["lost_qty_lag1"] = grp["lost_qty"].shift(1).fillna(0)
    df["pct_censored_7d"] = (
        grp["is_censored"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean() * 100)
    ).fillna(0)
    return df

FEATURES_CENSORING = ["is_censored_lag1", "lost_qty_lag1", "pct_censored_7d"]


def add_dow_features(df):
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура", "ДеньНедели"])
    df["sales_dow_mean"] = (
        grp["Продано"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)
    df["demand_dow_mean"] = (
        grp[DEMAND_TARGET].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)
    return df

FEATURES_DOW = ["sales_dow_mean", "demand_dow_mean"]


def add_trend_features(df):
    df["demand_trend"] = df["demand_roll_mean7"] / (df["demand_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1)
    return df

FEATURES_TREND = ["demand_trend", "cv_7d"]


def build_stale_ratio(sales_hrs_path, df):
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
    stale.rename(columns={"Дата продажи": "Дата", "Касса.Торговая точка": "Пекарня"}, inplace=True)
    stale["Дата"] = pd.to_datetime(stale["Дата"], format="%d.%m.%Y", errors="coerce")
    stale = stale[["Дата", "Пекарня", "Номенклатура", "stale_ratio"]].dropna(subset=["Дата"])

    print(f"    Stale ratio: {len(stale):,} rows, mean={stale['stale_ratio'].mean():.3f}")

    df = df.merge(stale, on=["Дата", "Пекарня", "Номенклатура"], how="left")
    df["stale_ratio"] = df["stale_ratio"].fillna(0)
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"])
    df["stale_ratio_lag1"] = (
        df.groupby(["Пекарня", "Номенклатура"])["stale_ratio"].shift(1).fillna(0)
    )
    df.drop(columns=["stale_ratio"], inplace=True)
    return df

FEATURES_STALE = ["stale_ratio_lag1"]


# -- Exp 62 features (bakery assortment) ---------------------------------

def add_assortment_features(df):
    df["items_in_bakery_today"] = (
        df.groupby(["Пекарня", "Дата"])["Номенклатура"].transform("nunique")
    )
    bakery_day = (
        df.groupby(["Пекарня", "Дата"])["Номенклатура"]
        .nunique().reset_index(name="__bakery_items")
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
    for col in FEATURES_ASSORTMENT:
        df[col] = df[col].fillna(0)
    return df

FEATURES_ASSORTMENT = ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"]


# -- All new features combined -------------------------------------------

ALL_NEW_FEATURES = (
    FEATURES_CENSORING + FEATURES_DOW + FEATURES_TREND +
    FEATURES_STALE + FEATURES_ASSORTMENT
)


# -- Main ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EXPERIMENT 63: Combined (exp 61 + exp 62)")
    print("  Censoring + DOW + Trend + Stale + Assortment")
    print("  Baseline: exp 60 V3 (MAE 2.8816)")
    print("=" * 60)
    t_start = time.time()

    # [1/6] Load
    print(f"\n[1/6] Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")

    # [2/6] Build exp 61 features
    print(f"\n[2/6] Building exp 61 features...")
    print("  Group A: censoring...")
    df = add_censoring_features(df)
    print("  Group B: DOW averages...")
    df = add_dow_features(df)
    print("  Group C: trend & volatility...")
    df = add_trend_features(df)
    print("  Group D: stale ratio...")
    df = build_stale_ratio(str(SALES_HRS_PATH), df)

    # [3/6] Build exp 62 features
    print(f"\n[3/6] Building exp 62 features (assortment)...")
    df = add_assortment_features(df)

    # [4/6] Categoricals
    print(f"\n[4/6] Converting categoricals...")
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # [5/6] Split
    print(f"\n[5/6] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    y_train = train[DEMAND_TARGET]
    y_test = test[DEMAND_TARGET].values

    # [6/6] Variants
    print(f"\n[6/6] Training & evaluating...")

    base = [f for f in FEATURES_V3 if f in df.columns]
    exp61_feats = FEATURES_CENSORING + FEATURES_DOW + FEATURES_TREND
    if "stale_ratio_lag1" in df.columns:
        exp61_feats = exp61_feats + FEATURES_STALE

    variants = [
        ("baseline (V3)",          base),
        ("exp 61 (cens+beh)",      base + exp61_feats),
        ("exp 62 (assortment)",    base + FEATURES_ASSORTMENT),
        ("exp 61+62 combined",     base + exp61_feats + FEATURES_ASSORTMENT),
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

        all_results[name] = {
            "mae": round(mae, 4),
            "wmape": round(wm, 2),
            "bias": round(bias, 4),
            "r2": round(r2, 4),
            "n_features": len(available),
            "delta_vs_v3": round(mae - BASELINE_V3_MAE, 4),
            "delta_vs_exp61": round(mae - EXP61_MAE, 4),
            "train_time_s": round(train_time, 1),
        }

        print(f"    MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}, R2={r2:.4f}")
        print(f"    Delta vs V3: {mae - BASELINE_V3_MAE:+.4f}, "
              f"vs exp61: {mae - EXP61_MAE:+.4f}")
        print(f"    Train time: {train_time:.0f}s")

        for threshold in [15, 50, 100]:
            mask = y_test >= threshold
            if mask.sum() > 0:
                mae_h = mean_absolute_error(y_test[mask], y_pred[mask])
                bias_h = float(np.mean(y_test[mask] - y_pred[mask]))
                print(f"    demand>={threshold}: MAE={mae_h:.2f}, Bias={bias_h:+.2f}, N={mask.sum()}")

    # Feature importance for combined
    combined_name = "exp 61+62 combined"
    combined_feats = dict(variants)[combined_name]
    combined_available = [f for f in combined_feats if f in df.columns]

    print(f"\n  === Feature importance ({combined_name}) ===")
    combined_model = train_quantile(train[combined_available], y_train, alpha=0.5)
    importance = pd.DataFrame({
        "feature": combined_available,
        "importance": combined_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    new_feats = set(ALL_NEW_FEATURES)
    for _, row in importance.head(30).iterrows():
        marker = " <<<" if row["feature"] in new_feats else ""
        print(f"    {row['feature']:<30} {row['importance']:>6.0f}{marker}")

    print(f"\n  New feature ranks:")
    for rank, (_, row) in enumerate(importance.iterrows(), 1):
        if row["feature"] in new_feats:
            print(f"    #{rank}: {row['feature']} ({row['importance']:.0f})")

    # Per-category for combined
    combined_pred = predict_clipped(combined_model, test[combined_available])
    print(f"\n  === Per-category ({combined_name}) ===")
    print_category_metrics(y_test, combined_pred, test["Категория"].values)

    # Summary
    print(f"\n  {'=' * 80}")
    print(f"  SUMMARY")
    print(f"  {'=' * 80}")
    print(f"  {'Variant':<25} {'Feats':>5} {'MAE':>8} {'WMAPE':>8} {'vs V3':>8} {'vs 61':>8}")
    print(f"  {'-' * 65}")
    for name, r in all_results.items():
        print(f"  {name:<25} {r['n_features']:>5} {r['mae']:>8.4f} "
              f"{r['wmape']:>7.2f}% {r['delta_vs_v3']:>+8.4f} {r['delta_vs_exp61']:>+8.4f}")

    # Save
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "63_combined_61_62",
        "target": DEMAND_TARGET,
        "baseline_v3_mae": BASELINE_V3_MAE,
        "exp61_mae": EXP61_MAE,
        "exp62_mae": EXP62_MAE,
        "results": all_results,
        "best_variant": min(all_results, key=lambda k: all_results[k]["mae"]),
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_importance_new": [
            {"feature": row["feature"], "importance": int(row["importance"]), "rank": i + 1}
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
