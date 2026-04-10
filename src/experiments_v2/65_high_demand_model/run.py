"""
Experiment 65: Separate LightGBM model for high-demand items.

Hypothesis: A model trained ONLY on high-demand items will better capture
their patterns (weekly seasonality, larger amplitude) than a global model
that's diluted by millions of low-demand rows.

Approach:
  1. Compute mean demand per (bakery, product) over training period
  2. Filter training data to items with mean demand >= threshold
  3. Train Q50 model on this subset
  4. Evaluate on ALL test data, comparing:
     - Global model (trained on everything) evaluated on high-demand test
     - High-demand model evaluated on high-demand test
     - Ensemble: high-demand model for high items, global for the rest

Thresholds tested: 10, 20, 50

Features: FEATURES_V3 + exp 61 + exp 62 (68 features, same as exp 63 best)

Baseline: exp 63 combined (MAE 2.8540 overall)

Input:  data/processed/daily_sales_8m_demand.csv
        data/raw/sales_hrs_all.csv (for stale ratio)
Output: src/experiments_v2/65_high_demand_model/metrics.json

Usage:
  .venv\\Scripts\\python.exe src/experiments_v2/65_high_demand_model/run.py
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

BASELINE_MAE = 2.8540  # exp 63 combined

THRESHOLDS = [10, 20, 50]


# -- Feature engineering (same as exp 63) -----------------------------------

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

ALL_NEW_FEATURES = (
    FEATURES_CENSORING + FEATURES_DOW + FEATURES_TREND +
    FEATURES_STALE + FEATURES_ASSORTMENT
)


# -- Main ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  EXPERIMENT 65: Separate High-Demand Model")
    print("  Thresholds: mean demand >= {10, 20, 50}")
    print("  Features: V3 + exp61 + exp62 (68 feats)")
    print(f"  Baseline: exp 63 combined (MAE {BASELINE_MAE})")
    print("=" * 70)
    t_start = time.time()

    # [1/7] Load
    print(f"\n[1/7] Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}")

    # [2/7] Build all features (same as exp 63)
    print(f"\n[2/7] Building features (exp 61 + exp 62)...")
    df = add_censoring_features(df)
    df = add_dow_features(df)
    df = add_trend_features(df)
    df = build_stale_ratio(str(SALES_HRS_PATH), df)
    df = add_assortment_features(df)

    # [3/7] Categoricals
    print(f"\n[3/7] Converting categoricals...")
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # [4/7] Split
    print(f"\n[4/7] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    # Feature list
    base = [f for f in FEATURES_V3 if f in df.columns]
    extra = [f for f in ALL_NEW_FEATURES if f in df.columns]
    features = base + extra
    available = [f for f in features if f in df.columns]
    print(f"  Features: {len(available)}")

    y_train = train[DEMAND_TARGET]
    y_test = test[DEMAND_TARGET].values

    # [5/7] Compute mean demand per (bakery, product) on TRAIN only
    print(f"\n[5/7] Computing item-level mean demand (train only)...")
    item_mean = (
        train.groupby(["Пекарня", "Номенклатура"])[DEMAND_TARGET]
        .mean()
        .reset_index(name="item_mean_demand")
    )
    print(f"  Unique (bakery, product) pairs: {len(item_mean):,}")
    print(f"  Mean demand distribution:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"    P{int(q*100):02d}: {item_mean['item_mean_demand'].quantile(q):.1f}")

    # Tag train and test rows with item_mean_demand
    train = train.merge(item_mean, on=["Пекарня", "Номенклатура"], how="left")
    test = test.merge(item_mean, on=["Пекарня", "Номенклатура"], how="left")
    test["item_mean_demand"] = test["item_mean_demand"].fillna(0)

    # [6/7] Train global model (baseline)
    print(f"\n[6/7] Training global model (all data)...")
    t0 = time.time()
    global_model = train_quantile(train[available], y_train, alpha=0.5)
    global_time = time.time() - t0
    print(f"  Time: {global_time:.0f}s")

    global_pred = predict_clipped(global_model, test[available])
    global_mae_all = mean_absolute_error(y_test, global_pred)
    print(f"  Global MAE (all test): {global_mae_all:.4f}")

    # [7/7] Train high-demand models at each threshold
    print(f"\n[7/7] Training high-demand models...")

    all_results = {
        "global (all data)": {
            "mae_all": round(global_mae_all, 4),
            "wmape_all": round(wmape(y_test, global_pred), 2),
            "train_rows": len(train),
            "train_time_s": round(global_time, 1),
        }
    }

    # Pre-compute global metrics by threshold for comparison
    for thr in THRESHOLDS:
        mask_high = test["item_mean_demand"] >= thr
        mask_low = ~mask_high
        n_high = mask_high.sum()
        n_low = mask_low.sum()

        if n_high == 0:
            print(f"\n  --- threshold >= {thr}: no high-demand items in test, skip ---")
            continue

        mae_high_global = mean_absolute_error(y_test[mask_high], global_pred[mask_high])
        mae_low_global = mean_absolute_error(y_test[mask_low], global_pred[mask_low]) if n_low > 0 else 0

        all_results[f"global (all data)"][f"mae_high_{thr}"] = round(mae_high_global, 4)
        all_results[f"global (all data)"][f"n_high_{thr}"] = int(n_high)

        print(f"\n  === Threshold: mean demand >= {thr} ===")

        # Filter train
        train_high = train[train["item_mean_demand"] >= thr].copy()
        n_items_high = train_high.groupby(["Пекарня", "Номенклатура"]).ngroups
        print(f"  High-demand train: {len(train_high):,} rows "
              f"({len(train_high)/len(train)*100:.1f}%), "
              f"{n_items_high} item pairs")
        print(f"  High-demand test:  {n_high:,} rows ({n_high/len(test)*100:.1f}%)")
        print(f"  Low-demand test:   {n_low:,} rows")

        # Train high-demand model
        t0 = time.time()
        hd_model = train_quantile(train_high[available], train_high[DEMAND_TARGET], alpha=0.5)
        hd_time = time.time() - t0
        print(f"  Train time: {hd_time:.0f}s")

        # Predict with HD model on high-demand test
        hd_pred_high = predict_clipped(hd_model, test.loc[mask_high, available])
        mae_hd_on_high = mean_absolute_error(y_test[mask_high], hd_pred_high)

        # Ensemble: HD model for high-demand, global for low-demand
        ensemble_pred = global_pred.copy()
        ensemble_pred[mask_high] = hd_pred_high
        mae_ensemble = mean_absolute_error(y_test, ensemble_pred)
        wmape_ensemble = wmape(y_test, ensemble_pred)

        # Bias
        bias_global_high = float(np.mean(y_test[mask_high] - global_pred[mask_high]))
        bias_hd_high = float(np.mean(y_test[mask_high] - hd_pred_high))

        # R2
        r2_global_high = r2_score(y_test[mask_high], global_pred[mask_high])
        r2_hd_high = r2_score(y_test[mask_high], hd_pred_high)

        print(f"\n  Comparison on HIGH-demand test (>= {thr}):")
        print(f"    {'Model':<25} {'MAE':>8} {'Bias':>8} {'R2':>8}")
        print(f"    {'-' * 52}")
        print(f"    {'Global model':<25} {mae_high_global:>8.4f} {bias_global_high:>+8.4f} {r2_global_high:>8.4f}")
        print(f"    {'High-demand model':<25} {mae_hd_on_high:>8.4f} {bias_hd_high:>+8.4f} {r2_hd_high:>8.4f}")
        delta_hd = mae_hd_on_high - mae_high_global
        print(f"    Delta (HD - Global): {delta_hd:+.4f} MAE")

        print(f"\n  Ensemble (HD for high, global for low):")
        print(f"    MAE overall: {mae_ensemble:.4f} (vs global {global_mae_all:.4f}, "
              f"delta {mae_ensemble - global_mae_all:+.4f})")
        print(f"    WMAPE: {wmape_ensemble:.2f}%")

        # Breakdown by demand level within high-demand
        print(f"\n  High-demand breakdown:")
        for sub_thr in [thr, max(thr, 50), 100, 200]:
            if sub_thr < thr:
                continue
            sub_mask = (test["item_mean_demand"] >= sub_thr)
            n_sub = sub_mask.sum()
            if n_sub == 0:
                continue
            mae_g = mean_absolute_error(y_test[sub_mask], global_pred[sub_mask])
            mae_h = mean_absolute_error(y_test[sub_mask], ensemble_pred[sub_mask])
            bias_g = float(np.mean(y_test[sub_mask] - global_pred[sub_mask]))
            bias_h = float(np.mean(y_test[sub_mask] - ensemble_pred[sub_mask]))
            print(f"    demand>={sub_thr:>3}: N={n_sub:>6}, "
                  f"Global MAE={mae_g:.2f} (bias {bias_g:+.2f}), "
                  f"HD MAE={mae_h:.2f} (bias {bias_h:+.2f}), "
                  f"delta={mae_h - mae_g:+.4f}")

        # Feature importance comparison
        print(f"\n  Feature importance (HD model, top 15):")
        importance = pd.DataFrame({
            "feature": available,
            "importance": hd_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        new_feats = set(ALL_NEW_FEATURES)
        for _, row in importance.head(15).iterrows():
            marker = " <<<" if row["feature"] in new_feats else ""
            print(f"    {row['feature']:<30} {row['importance']:>6.0f}{marker}")

        # Per-category for ensemble
        print(f"\n  Per-category (ensemble vs global):")
        print(f"    {'Kategoriya':<25} {'N':>5} {'Global':>8} {'Ensemble':>8} {'Delta':>8}")
        print(f"    {'-' * 58}")
        cats = test["Категория"].values
        for cat in sorted(set(cats)):
            cat_mask = np.array(cats) == cat
            n_cat = cat_mask.sum()
            mae_g = mean_absolute_error(y_test[cat_mask], global_pred[cat_mask])
            mae_e = mean_absolute_error(y_test[cat_mask], ensemble_pred[cat_mask])
            print(f"    {str(cat):<25} {n_cat:>5} {mae_g:>8.4f} {mae_e:>8.4f} {mae_e - mae_g:>+8.4f}")

        # Save results for this threshold
        all_results[f"hd_model_thr{thr}"] = {
            "threshold": thr,
            "train_rows_hd": len(train_high),
            "train_pct": round(len(train_high) / len(train) * 100, 1),
            "n_item_pairs": n_items_high,
            "test_rows_high": int(n_high),
            "test_rows_low": int(n_low),
            "mae_hd_on_high": round(mae_hd_on_high, 4),
            "mae_global_on_high": round(mae_high_global, 4),
            "delta_hd_vs_global_high": round(delta_hd, 4),
            "bias_global_high": round(bias_global_high, 4),
            "bias_hd_high": round(bias_hd_high, 4),
            "r2_global_high": round(r2_global_high, 4),
            "r2_hd_high": round(r2_hd_high, 4),
            "mae_ensemble_all": round(mae_ensemble, 4),
            "delta_ensemble_vs_global": round(mae_ensemble - global_mae_all, 4),
            "wmape_ensemble": round(wmape_ensemble, 2),
            "train_time_s": round(hd_time, 1),
        }

    # Summary
    print(f"\n  {'=' * 80}")
    print(f"  SUMMARY")
    print(f"  {'=' * 80}")
    print(f"\n  Global model MAE (all test): {global_mae_all:.4f}")
    print(f"\n  {'Threshold':<12} {'HD train':>10} {'HD MAE':>10} {'Global MAE':>10} "
          f"{'Delta HD':>10} {'Ensemble':>10} {'Ens delta':>10}")
    print(f"  {'-' * 72}")
    for thr in THRESHOLDS:
        key = f"hd_model_thr{thr}"
        if key in all_results:
            r = all_results[key]
            print(f"  >= {thr:<8} {r['train_rows_hd']:>10,} {r['mae_hd_on_high']:>10.4f} "
                  f"{r['mae_global_on_high']:>10.4f} {r['delta_hd_vs_global_high']:>+10.4f} "
                  f"{r['mae_ensemble_all']:>10.4f} {r['delta_ensemble_vs_global']:>+10.4f}")

    # Best ensemble
    best_thr = min(
        [t for t in THRESHOLDS if f"hd_model_thr{t}" in all_results],
        key=lambda t: all_results[f"hd_model_thr{t}"]["mae_ensemble_all"]
    )
    best = all_results[f"hd_model_thr{best_thr}"]
    print(f"\n  Best ensemble: threshold >= {best_thr}")
    print(f"  Ensemble MAE: {best['mae_ensemble_all']:.4f} "
          f"(vs global {global_mae_all:.4f}, delta {best['delta_ensemble_vs_global']:+.4f})")
    print(f"  On high-demand: HD model {best['mae_hd_on_high']:.4f} "
          f"vs global {best['mae_global_on_high']:.4f} "
          f"(delta {best['delta_hd_vs_global_high']:+.4f})")

    # Save
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "65_high_demand_model",
        "target": DEMAND_TARGET,
        "baseline_mae": BASELINE_MAE,
        "global_mae_all": round(global_mae_all, 4),
        "thresholds": THRESHOLDS,
        "results": all_results,
        "best_threshold": best_thr,
        "best_ensemble_mae": best["mae_ensemble_all"],
        "best_delta_vs_global": best["delta_ensemble_vs_global"],
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": len(available),
    }
    save_results(EXP_DIR, metrics)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
