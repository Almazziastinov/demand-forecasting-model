"""
One-time script: merge price features into daily_sales_8m_demand.csv.

Reads cached daily_prices_8m.csv (built by exp 09),
computes price features, merges into demand CSV, saves updated file.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/merge_prices.py
"""

import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

DEMAND_CSV = os.path.join(ROOT, "data", "processed", "daily_sales_8m_demand.csv")
PRICE_CACHE = os.path.join(ROOT, "data", "processed", "daily_prices_8m.csv")

PRICE_FEATURES = [
    "avg_price", "price_vs_median", "price_lag7",
    "price_change_7d", "price_roll_mean7", "price_roll_std7",
]


def main():
    t_start = time.time()
    print("=" * 60)
    print("  Merging price features into daily_sales_8m_demand.csv")
    print("=" * 60)

    # --- Load prices ---
    print(f"\n[1/4] Loading cached prices from {os.path.basename(PRICE_CACHE)}...")
    if not os.path.exists(PRICE_CACHE):
        print("  ERROR: daily_prices_8m.csv not found! Run exp 09 first.")
        return
    prices = pd.read_csv(PRICE_CACHE, encoding="utf-8-sig")
    prices["Дата"] = pd.to_datetime(prices["Дата"])
    print(f"  {len(prices):,} rows")

    # --- Build price features ---
    print(f"\n[2/4] Building price features...")
    prices = prices.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    group = prices.groupby(["Пекарня", "Номенклатура"])

    product_median = prices.groupby("Номенклатура")["avg_price"].median()
    prices["price_vs_median"] = prices["Номенклатура"].map(product_median)
    prices["price_vs_median"] = prices["avg_price"] / prices["price_vs_median"]

    prices["price_lag7"] = group["avg_price"].shift(7)
    prices["price_change_7d"] = (
        (prices["avg_price"] - prices["price_lag7"]) / (prices["price_lag7"] + 1e-8) * 100
    )
    prices["price_roll_mean7"] = group["avg_price"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    prices["price_roll_std7"] = group["avg_price"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )

    for col in PRICE_FEATURES:
        fill = prices[col].notna().mean() * 100
        print(f"    {col:<25} fill: {fill:.0f}%")

    # --- Load demand CSV ---
    print(f"\n[3/4] Loading demand CSV...")
    df = pd.read_csv(DEMAND_CSV, encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape before: {df.shape}")

    # Drop existing price columns if any (re-run safe)
    existing_price_cols = [c for c in PRICE_FEATURES if c in df.columns]
    if existing_price_cols:
        print(f"  Dropping existing price columns: {existing_price_cols}")
        df = df.drop(columns=existing_price_cols)

    # --- Merge ---
    print(f"\n[4/4] Merging...")
    price_merge = prices[["Дата", "Пекарня", "Номенклатура"] + PRICE_FEATURES].copy()
    df = df.merge(price_merge, on=["Дата", "Пекарня", "Номенклатура"], how="left")

    match_pct = df["avg_price"].notna().mean() * 100
    print(f"  Shape after: {df.shape}")
    print(f"  Price match rate: {match_pct:.1f}%")

    # --- Save ---
    print(f"\n  Saving updated CSV...")
    df.to_csv(DEMAND_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved to {DEMAND_CSV}")

    elapsed = time.time() - t_start
    print(f"  Done! ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
