"""
Experiment 09: Price Features (demand target).

Compare two models:
  Model A: baseline + demand features (FEATURES_V2 + demand lags)
  Model B: baseline + demand features + price features

Price features extracted from sales_hrs_all.csv (30M check-level rows):
  - avg_price: weighted average price per day x bakery x product
  - price_vs_median: ratio of avg_price to product's overall median price
  - price_lag7: price 7 days ago
  - price_change_7d: % change in price vs 7 days ago
  - price_roll_mean7: 7-day rolling mean price
  - price_roll_std7: 7-day rolling std of price (volatility)

Both train on Spros (demand), evaluate vs Spros.

Input:  data/processed/daily_sales_8m_demand.csv
        data/raw/sales_hrs_all.csv
Output: src/experiments_v2/09_price_features/metrics.json
        src/experiments_v2/09_price_features/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/09_price_features/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
DEMAND_CSV = Path(ROOT) / "data" / "processed" / "daily_sales_8m_demand.csv"
CHECKS_CSV = Path(ROOT) / "data" / "raw" / "sales_hrs_all.csv"
PRICE_CACHE = Path(ROOT) / "data" / "processed" / "daily_prices_8m.csv"

DEMAND_TARGET = "Спрос"

DEMAND_FEATURES = [
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
    "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30",
    "demand_roll_std7",
]

PRICE_FEATURES = [
    "avg_price",
    "price_vs_median",
    "price_lag7",
    "price_change_7d",
    "price_roll_mean7",
    "price_roll_std7",
]


def aggregate_prices():
    """Aggregate check-level data to daily prices per bakery x product.
    Uses chunked reading for 4.8 GB file. Caches result.
    """
    if PRICE_CACHE.exists():
        print(f"  Loading cached prices from {PRICE_CACHE.name}...")
        return pd.read_csv(str(PRICE_CACHE), encoding="utf-8-sig")

    print(f"  Aggregating prices from {CHECKS_CSV.name} (chunked)...")
    CHUNK_SIZE = 2_000_000
    agg_list = []

    for i, chunk in enumerate(pd.read_csv(
        str(CHECKS_CSV), encoding="utf-8-sig", chunksize=CHUNK_SIZE,
        usecols=["Дата продажи", "Касса.Торговая точка", "Номенклатура", "Цена", "Кол-во"],
    )):
        # Weighted price: sum(price * qty) and sum(qty) per group
        chunk["price_x_qty"] = chunk["Цена"] * chunk["Кол-во"]
        daily = chunk.groupby(
            ["Дата продажи", "Касса.Торговая точка", "Номенклатура"]
        ).agg(
            total_revenue=("price_x_qty", "sum"),
            total_qty=("Кол-во", "sum"),
        ).reset_index()
        agg_list.append(daily)
        print(f"    Chunk {i + 1}: {len(chunk):,} rows -> {len(daily):,} groups", flush=True)

    print(f"  Merging {len(agg_list)} chunks...")
    all_daily = pd.concat(agg_list, ignore_index=True)

    # Re-aggregate (chunks may split same group)
    prices = all_daily.groupby(
        ["Дата продажи", "Касса.Торговая точка", "Номенклатура"]
    ).agg(
        total_revenue=("total_revenue", "sum"),
        total_qty=("total_qty", "sum"),
    ).reset_index()

    prices["avg_price"] = prices["total_revenue"] / prices["total_qty"]
    prices = prices.rename(columns={
        "Дата продажи": "Дата",
        "Касса.Торговая точка": "Пекарня",
    })
    prices = prices[["Дата", "Пекарня", "Номенклатура", "avg_price"]].copy()
    prices["Дата"] = pd.to_datetime(prices["Дата"], dayfirst=True)

    print(f"  Daily prices: {len(prices):,} rows")

    # Cache
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(str(PRICE_CACHE), index=False, encoding="utf-8-sig")
    print(f"  Cached to {PRICE_CACHE}")

    return prices


def build_price_features(prices):
    """Build price features from daily avg_price."""
    print(f"  Building price features...")

    # Sort for lag/rolling computation
    prices = prices.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()

    group = prices.groupby(["Пекарня", "Номенклатура"])

    # Product median price (static reference)
    product_median = prices.groupby("Номенклатура")["avg_price"].median()
    prices["price_vs_median"] = prices["Номенклатура"].map(product_median)
    prices["price_vs_median"] = prices["avg_price"] / prices["price_vs_median"]

    # Lag 7
    prices["price_lag7"] = group["avg_price"].shift(7)

    # Change vs 7 days ago (%)
    prices["price_change_7d"] = (
        (prices["avg_price"] - prices["price_lag7"]) / (prices["price_lag7"] + 1e-8) * 100
    )

    # Rolling 7-day stats
    prices["price_roll_mean7"] = group["avg_price"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    prices["price_roll_std7"] = group["avg_price"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )

    print(f"  Price features built. Shape: {prices.shape}")
    for col in PRICE_FEATURES:
        if col in prices.columns:
            fill = prices[col].notna().mean() * 100
            print(f"    {col:<25} fill: {fill:.0f}%, mean: {prices[col].mean():.2f}")

    return prices


def main():
    print("=" * 60)
    print("  EXPERIMENT 09: Price Features (demand target)")
    print("  Model A: baseline + demand lags")
    print("  Model B: baseline + demand lags + price features")
    print("  Target: Spros")
    print("=" * 60)
    t_start = time.time()

    # --- Step 1: Aggregate prices ---
    print(f"\n[1/6] Price aggregation...")
    prices_raw = aggregate_prices()

    # --- Step 2: Build price features ---
    print(f"\n[2/6] Building price features...")
    prices = build_price_features(prices_raw)

    # --- Step 3: Load demand data ---
    print(f"\n[3/6] Loading demand data from {DEMAND_CSV.name}...")
    if not DEMAND_CSV.exists():
        print(f"  ERROR: {DEMAND_CSV} not found!")
        return
    df = pd.read_csv(str(DEMAND_CSV), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {len(df):,} rows, {df['Пекарня'].nunique()} bakeries")

    # --- Merge price features ---
    print(f"\n[4/6] Merging price features...")
    price_cols = ["Дата", "Пекарня", "Номенклатура"] + PRICE_FEATURES
    price_merge = prices[[c for c in price_cols if c in prices.columns]].copy()

    before = len(df)
    df = df.merge(price_merge, on=["Дата", "Пекарня", "Номенклатура"], how="left")
    print(f"  Merged: {len(df):,} rows (was {before:,})")

    matched_pct = df["avg_price"].notna().mean() * 100
    print(f"  Price match rate: {matched_pct:.1f}%")

    # Demand stats
    mean_demand = df[DEMAND_TARGET].mean()
    censored_pct = 100 * df["is_censored"].mean()
    print(f"  mean({DEMAND_TARGET}): {mean_demand:.4f}, censored: {censored_pct:.1f}%")

    # --- Features ---
    baseline_features = [f for f in FEATURES_V2 + DEMAND_FEATURES if f in df.columns]
    extended_features = baseline_features + [f for f in PRICE_FEATURES if f in df.columns]

    print(f"\n  Model A (baseline + demand): {len(baseline_features)} features")
    print(f"  Model B (+ price):           {len(extended_features)} features "
          f"(+{len(extended_features) - len(baseline_features)} price)")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Train/test split ---
    print(f"\n[5/6] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    y_train = train[DEMAND_TARGET]
    y_test = test[DEMAND_TARGET]

    # --- Train ---
    print(f"\n[6/6] Training models...")

    print(f"\n  --- Model A: baseline + demand ({len(baseline_features)} features) ---")
    t_a = time.time()
    model_a = train_lgbm(train[baseline_features], y_train)
    time_a = time.time() - t_a
    pred_a = predict_clipped(model_a, test[baseline_features])
    mae_a, wm_a, bias_a = print_metrics("Model A (baseline)", y_test, pred_a)
    print(f"    Train time: {time_a:.0f}s")

    print(f"\n  --- Model B: + price ({len(extended_features)} features) ---")
    t_b = time.time()
    model_b = train_lgbm(train[extended_features], y_train)
    time_b = time.time() - t_b
    pred_b = predict_clipped(model_b, test[extended_features])
    mae_b, wm_b, bias_b = print_metrics("Model B (+ price)", y_test, pred_b, baseline_mae=mae_a)
    print(f"    Train time: {time_b:.0f}s")

    # --- Comparison ---
    print(f"\n  === Comparison (vs {DEMAND_TARGET}) ===")
    print(f"\n  {'Metric':<12} {'Model A':>10} {'Model B':>10} {'Delta':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'MAE':<12} {mae_a:>10.4f} {mae_b:>10.4f} {mae_b - mae_a:>+10.4f}")
    print(f"  {'WMAPE':<12} {wm_a:>10.2f}% {wm_b:>10.2f}% {wm_b - wm_a:>+10.2f}%")
    print(f"  {'Bias':<12} {bias_a:>+10.4f} {bias_b:>+10.4f} {bias_b - bias_a:>+10.4f}")

    winner = "Model B (+ price)" if mae_b < mae_a else "Model A (baseline)"
    print(f"\n  Winner: {winner}")

    # High demand analysis
    print(f"\n  High demand analysis (vs {DEMAND_TARGET}):")
    for threshold in [15, 50, 100]:
        mask = y_test.values >= threshold
        if mask.sum() > 0:
            ma_a = mean_absolute_error(y_test.values[mask], pred_a[mask])
            ma_b = mean_absolute_error(y_test.values[mask], pred_b[mask])
            print(f"    >= {threshold}: N={mask.sum()}, "
                  f"A MAE={ma_a:.2f} | B MAE={ma_b:.2f} (delta {ma_b - ma_a:+.2f})")

    # Per-category
    print(f"\n  === Model A per-category ===")
    print_category_metrics(y_test, pred_a, test["Категория"].values)

    print(f"\n  === Model B per-category ===")
    print_category_metrics(y_test, pred_b, test["Категория"].values)

    # --- Feature importance ---
    print(f"\n  === Price feature importance (Model B) ===")
    importance_b = pd.DataFrame({
        "feature": extended_features,
        "importance": model_b.feature_importances_,
    }).sort_values("importance", ascending=False)

    price_imp = importance_b[importance_b["feature"].str.startswith("price") | (importance_b["feature"] == "avg_price")]
    print(f"    {'Feature':<25} {'Importance':>10} {'Rank':>6}")
    print(f"    {'-' * 43}")
    for rank, (_, row) in enumerate(importance_b.iterrows(), 1):
        if row["feature"] in PRICE_FEATURES:
            print(f"    {row['feature']:<25} {row['importance']:>10.0f} {rank:>6}")

    # Top 15 overall
    print(f"\n  === Top 15 features (Model B) ===")
    for _, row in importance_b.head(15).iterrows():
        marker = " <-- PRICE" if row["feature"] in PRICE_FEATURES else ""
        print(f"    {row['feature']:<25} {row['importance']:>8.0f}{marker}")

    # --- Save ---
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "09_price_features",
        "target": DEMAND_TARGET,
        "model_a_baseline": {
            "mae": round(mae_a, 4),
            "wmape": round(wm_a, 2),
            "bias": round(bias_a, 4),
            "n_features": len(baseline_features),
            "train_time_s": round(time_a, 1),
        },
        "model_b_price": {
            "mae": round(mae_b, 4),
            "wmape": round(wm_b, 2),
            "bias": round(bias_b, 4),
            "n_features": len(extended_features),
            "train_time_s": round(time_b, 1),
        },
        "delta_mae": round(mae_b - mae_a, 4),
        "delta_wmape": round(wm_b - wm_a, 2),
        "winner": winner,
        "price_match_rate": round(matched_pct, 1),
        "train_rows": len(train),
        "test_rows": len(test),
        "censored_pct": round(censored_pct, 1),
        "price_feature_importance": price_imp[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_demand": y_test.values,
        "pred_baseline": np.round(pred_a, 2),
        "pred_price": np.round(pred_b, 2),
        "avg_price": test["avg_price"].values,
        "is_censored": test["is_censored"].values,
        "abs_err_baseline": np.round(np.abs(y_test.values - pred_a), 2),
        "abs_err_price": np.round(np.abs(y_test.values - pred_b), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
