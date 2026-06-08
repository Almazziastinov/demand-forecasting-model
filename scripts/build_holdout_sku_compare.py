"""Allocate the bakery-day holdout forecast to SKU level and join with fact.

Inputs:
  reports/holdout_30d_bakery_compare.csv
      bakery-day forecast+fact, 05-02..05-31
  data/processed/bakery_hour_profile.csv
      bakery_id, dow, hour, mean_hour_share_norm
  data/processed/sku_hour_share_profile.csv            (raw SKU-in-hour share)
  data/processed/sku_hour_share_profile_smoothed.csv   (smoothed SKU-in-hour share)
  data/processed/actual_sku_daily_clickhouse_eval30d.csv
      SKU fact, 04-13..05-12; we keep the overlap with holdout window

Math
----
For each (bakery_id, product_id, dow):

    sku_day_share = sum_hour( bakery_hour_share_norm(bakery,dow,hour)
                              * sku_in_hour_share_norm(bakery,product,dow,hour) )

Then for each (date, bakery, product):

    sku_day_forecast =
        bakery_day_forecast(date,bakery)
        * sku_day_share(bakery,product,dow(date))

We do this for both raw and smoothed SKU profiles.

Output:
  reports/holdout_sku_compare.csv
    columns: date, bakery_id, bakery_name, city, product_id, fact,
             forecast_raw, forecast_smoothed, err_raw, err_smoothed,
             abs_err_raw, abs_err_smoothed

Only the date range that has BOTH bakery-forecast (05-02..05-31) AND fact
(04-13..05-12) is kept -> effective window = 2026-05-02..2026-05-12 (11 days).
That is enough for a first look at SKU allocation structure; we can extend the
fact later if needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

BAKERY_HOLDOUT = REPO_ROOT / "reports" / "holdout_30d_bakery_compare.csv"
BAKERY_HOUR_PROFILE = REPO_ROOT / "data" / "processed" / "bakery_hour_profile.csv"
SKU_PROFILE_RAW = REPO_ROOT / "data" / "processed" / "sku_hour_share_profile.csv"
SKU_PROFILE_SMOOTHED = (
    REPO_ROOT / "data" / "processed" / "sku_hour_share_profile_smoothed.csv"
)
SKU_FACT = REPO_ROOT / "data" / "processed" / "actual_sku_daily_clickhouse_eval30d.csv"

OUT_CSV = REPO_ROOT / "reports" / "holdout_sku_compare.csv"

# Only the overlap of bakery-holdout window and SKU-fact window.
WINDOW_START = pd.Timestamp("2026-05-02")
WINDOW_END = pd.Timestamp("2026-05-12")


def _clean_csv_header(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def _build_sku_day_share(bakery_hour: pd.DataFrame, sku_hour: pd.DataFrame,
                        share_col: str) -> pd.DataFrame:
    """Collapse hourly profiles into one share per (bakery, sku, dow)."""
    cols_b = ["bakery_id", "dow", "hour", "mean_hour_share_norm"]
    cols_s = ["bakery_id", "product_id", "dow", "hour", share_col]
    bh = bakery_hour[cols_b].copy()
    sh = sku_hour[cols_s].copy()
    merged = sh.merge(bh, on=["bakery_id", "dow", "hour"], how="inner")
    merged["contrib"] = merged[share_col] * merged["mean_hour_share_norm"]
    out = (
        merged
        .groupby(["bakery_id", "product_id", "dow"], as_index=False)["contrib"]
        .sum()
        .rename(columns={"contrib": "sku_day_share"})
    )
    return out


def main() -> int:
    print("Loading bakery-holdout forecast...")
    holdout = pd.read_csv(BAKERY_HOLDOUT, parse_dates=["date"])
    holdout = holdout[
        (holdout["date"] >= WINDOW_START) & (holdout["date"] <= WINDOW_END)
    ].copy()
    holdout["dow"] = holdout["date"].dt.dayofweek
    print(f"  rows in window: {len(holdout)}  bakeries: {holdout.bakery_id.nunique()}")

    print("Loading bakery-hour profile...")
    bakery_hour = _clean_csv_header(BAKERY_HOUR_PROFILE)
    print(f"  rows: {len(bakery_hour)}")

    print("Loading raw SKU profile...")
    sku_raw = _clean_csv_header(SKU_PROFILE_RAW)
    print(f"  rows: {len(sku_raw)}")

    print("Loading smoothed SKU profile...")
    sku_sm = _clean_csv_header(SKU_PROFILE_SMOOTHED)
    print(f"  rows: {len(sku_sm)}")

    print("Building per-day SKU shares (raw)...")
    share_raw = _build_sku_day_share(
        bakery_hour, sku_raw, "mean_sku_share_in_hour_norm")
    share_raw = share_raw.rename(columns={"sku_day_share": "share_raw"})
    print(f"  rows: {len(share_raw)}")

    print("Building per-day SKU shares (smoothed)...")
    share_sm = _build_sku_day_share(
        bakery_hour, sku_sm, "mean_sku_share_in_hour_norm")
    share_sm = share_sm.rename(columns={"sku_day_share": "share_smoothed"})
    print(f"  rows: {len(share_sm)}")

    shares = share_raw.merge(
        share_sm, on=["bakery_id", "product_id", "dow"], how="outer")
    shares[["share_raw", "share_smoothed"]] = shares[
        ["share_raw", "share_smoothed"]].fillna(0.0)
    print(f"shares merged: {len(shares)} (bakery, product, dow) triples")

    # Sanity: shares should sum to ~1 per (bakery, dow) across products.
    ssum = shares.groupby(["bakery_id", "dow"])[
        ["share_raw", "share_smoothed"]].sum()
    print(f"  share_raw sum per (bakery,dow):       "
          f"min={ssum.share_raw.min():.4f} mean={ssum.share_raw.mean():.4f} "
          f"max={ssum.share_raw.max():.4f}")
    print(f"  share_smoothed sum per (bakery,dow):  "
          f"min={ssum.share_smoothed.min():.4f} mean={ssum.share_smoothed.mean():.4f} "
          f"max={ssum.share_smoothed.max():.4f}")

    print("Multiplying bakery-day forecast by SKU shares...")
    # Use the base (non-uplifted) bakery forecast as the volume to split.
    fc = holdout[
        ["date", "dow", "bakery_id", "bakery_name", "city", "forecast_base"]
    ].rename(columns={"forecast_base": "bakery_day_forecast"})
    sku_fc = fc.merge(shares, on=["bakery_id", "dow"], how="inner")
    sku_fc["forecast_raw"] = sku_fc["bakery_day_forecast"] * sku_fc["share_raw"]
    sku_fc["forecast_smoothed"] = (
        sku_fc["bakery_day_forecast"] * sku_fc["share_smoothed"]
    )
    print(f"  rows: {len(sku_fc)}")

    print("Loading SKU fact...")
    fact = pd.read_csv(SKU_FACT, parse_dates=["check_date"])
    fact.columns = [c.lstrip("﻿") for c in fact.columns]
    fact = fact[
        (fact["check_date"] >= WINDOW_START) & (fact["check_date"] <= WINDOW_END)
    ].rename(columns={"check_date": "date", "quantity": "fact"})
    print(f"  rows in window: {len(fact)}")

    # IMPORTANT: SKU fact (from checks, "quantity") and bakery fact_base
    # ("Продано") are NOT in the same units. Bakery fact_base is the cleaned
    # daily sold quantity used to train the bakery model; SKU fact from checks
    # is ~12% larger on aggregate.
    #
    # To make SKU forecast (which sums to bakery_day_forecast, i.e. the model's
    # estimate of fact_base) comparable to SKU fact, we rescale SKU fact per
    # (date, bakery) so that its sum matches bakery fact_base for that bakery
    # on that date. This preserves the SHARE structure of the real receipts
    # (which is what we actually want to evaluate — does the profile allocate
    # to the right SKUs?) while putting fact and forecast in the same level.
    fact_totals = (
        fact.groupby(["date", "bakery_id"])["fact"].sum()
        .rename("sku_fact_total").reset_index()
    )
    fact = fact.merge(fact_totals, on=["date", "bakery_id"], how="left")
    bakery_fact = holdout[["date", "bakery_id", "fact_base"]].rename(
        columns={"fact_base": "bakery_fact_total"})
    fact = fact.merge(bakery_fact, on=["date", "bakery_id"], how="left")
    # Scale: each SKU fact * (bakery_fact_base / sku_fact_total).
    # If sku_fact_total is 0 or NaN, scale is 1 (no fact to scale anyway).
    fact["scale"] = np.where(
        (fact["sku_fact_total"] > 0) & fact["bakery_fact_total"].notna(),
        fact["bakery_fact_total"] / fact["sku_fact_total"],
        1.0,
    )
    fact["fact"] = fact["fact"] * fact["scale"]

    merged = sku_fc.merge(
        fact[["date", "bakery_id", "product_id", "fact"]],
        on=["date", "bakery_id", "product_id"], how="outer",
    )

    # For rows that exist only on the forecast side, fact is implicitly 0
    # (the product was not sold that day). For rows that exist only on the
    # fact side, the forecast is 0 (no profile entry for that SKU in that
    # bakery/dow). Both are real allocation errors, not bugs.
    merged["fact"] = merged["fact"].fillna(0.0)
    merged["forecast_raw"] = merged["forecast_raw"].fillna(0.0)
    merged["forecast_smoothed"] = merged["forecast_smoothed"].fillna(0.0)

    # Re-fill bakery metadata for fact-only rows.
    name_lookup = (
        holdout[["bakery_id", "bakery_name", "city"]]
        .drop_duplicates("bakery_id")
        .set_index("bakery_id")
    )
    for col in ("bakery_name", "city"):
        missing = merged[col].isna()
        if missing.any():
            merged.loc[missing, col] = merged.loc[missing, "bakery_id"].map(
                name_lookup[col])

    # Errors
    merged["err_raw"] = merged["forecast_raw"] - merged["fact"]
    merged["err_smoothed"] = merged["forecast_smoothed"] - merged["fact"]
    merged["abs_err_raw"] = merged["err_raw"].abs()
    merged["abs_err_smoothed"] = merged["err_smoothed"].abs()

    keep = [
        "date", "bakery_id", "bakery_name", "city", "product_id",
        "fact", "forecast_raw", "forecast_smoothed",
        "err_raw", "err_smoothed", "abs_err_raw", "abs_err_smoothed",
    ]
    merged = (
        merged[keep]
        .sort_values(["bakery_id", "product_id", "date"])
        .reset_index(drop=True)
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Console summary (ASCII).
    n_rows = len(merged)
    n_bakeries = merged["bakery_id"].nunique()
    n_products = merged["product_id"].nunique()
    days = merged["date"].nunique()
    sum_fact = merged["fact"].sum()
    sum_raw = merged["forecast_raw"].sum()
    sum_sm = merged["forecast_smoothed"].sum()
    mae_raw = merged["abs_err_raw"].mean()
    mae_sm = merged["abs_err_smoothed"].mean()
    wmape_raw = merged["abs_err_raw"].sum() / max(sum_fact, 1.0) * 100
    wmape_sm = merged["abs_err_smoothed"].sum() / max(sum_fact, 1.0) * 100

    print()
    print(f"output: {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"rows:               {n_rows}")
    print(f"bakeries:           {n_bakeries}")
    print(f"products:           {n_products}")
    print(f"days:               {days}")
    print(f"sum fact:           {sum_fact:>16.2f}")
    print(f"sum forecast raw:   {sum_raw:>16.2f}")
    print(f"sum forecast smooth:{sum_sm:>16.2f}")
    print(f"MAE   raw:          {mae_raw:.4f}")
    print(f"MAE   smoothed:     {mae_sm:.4f}")
    print(f"WMAPE raw:          {wmape_raw:.4f} %")
    print(f"WMAPE smoothed:     {wmape_sm:.4f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
