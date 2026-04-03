"""
Experiment 03: Build demand profiles and corrected demand target.

Scales the lost-demand method from checks_analysis.ipynb (Section 16)
to ALL bakeries and products. For each (bakery, product) pair, builds
a cumulative hourly sales profile from "full days" (last_sale >= 17:00),
then estimates true demand for days when the product sold out early.

Input:
  data/raw/sales_hrs_all.csv          (~4.6 GB, 30M rows)
  data/processed/daily_sales_8m.csv   (daily aggregated features)

Output:
  data/processed/demand_profiles.json   {bakery|product: {hour: cum_share}}
  data/processed/daily_sales_8m_demand.csv  (daily_sales_8m + demand columns)

Usage:
  .venv/Scripts/python.exe src/experiments_v2/03_demand_target/build_demand_profiles.py
"""

import sys
import os
import time
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import pandas as pd

from src.experiments_v2.common import (
    SALES_HRS_PATH, DAILY_8M_PATH, TARGET,
)

# --- Constants ---
CHUNK_SIZE = 2_000_000
USE_COLS = ["Дата продажи", "Дата время чека", "Вид события по кассе",
            "Касса.Торговая точка", "Категория", "Номенклатура", "Кол-во"]

FULL_DAY_THRESHOLD = 17   # "full day" = last sale at hour >= 17
MIN_FULL_DAYS = 5          # minimum full days for reliable profile
CUM_FLOOR = 0.05           # don't extrapolate if cum < 5%
MAX_MULTIPLIER = 5.0       # cap: estimated <= actual * 5

OUTPUT_PROFILES = os.path.join(ROOT, "data", "processed", "demand_profiles.json")
OUTPUT_DEMAND_CSV = os.path.join(ROOT, "data", "processed", "daily_sales_8m_demand.csv")


def step1_hourly_aggregation():
    """Chunked read of sales_hrs_all.csv -> hourly aggregation per (bakery, product, date, hour)."""
    print("=" * 60)
    print("  STEP 1: Chunked hourly aggregation")
    print("=" * 60)

    agg_chunks = []
    total_rows = 0
    total_sales = 0
    t0 = time.time()

    reader = pd.read_csv(
        str(SALES_HRS_PATH), encoding="utf-8-sig",
        usecols=USE_COLS, chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader, 1):
        n = len(chunk)
        total_rows += n

        # Filter: only sales
        sales = chunk[chunk["Вид события по кассе"] == "Продажа"].copy()
        total_sales += len(sales)

        # Extract hour from datetime
        sales["hour"] = pd.to_datetime(sales["Дата время чека"], format="%d.%m.%Y %H:%M:%S").dt.hour

        # Aggregate: (date, bakery, product, hour) -> sum qty
        agg = (sales.groupby(["Дата продажи", "Касса.Торговая точка",
                               "Номенклатура", "hour"])["Кол-во"]
               .sum()
               .reset_index())
        agg_chunks.append(agg)

        elapsed = time.time() - t0
        print(f"  Chunk {i}: {n:>10,} rows, {len(sales):>10,} sales, "
              f"{len(agg):>8,} aggregated  ({elapsed:.0f}s)", flush=True)

    print(f"\n  Total: {total_rows:,} rows, {total_sales:,} sales")

    # Merge all chunks
    print("  Merging chunks...", flush=True)
    hourly = pd.concat(agg_chunks, ignore_index=True)
    hourly = (hourly.groupby(["Дата продажи", "Касса.Торговая точка",
                               "Номенклатура", "hour"])["Кол-во"]
              .sum()
              .reset_index())

    # Rename
    hourly.rename(columns={
        "Дата продажи": "Дата",
        "Касса.Торговая точка": "Пекарня",
    }, inplace=True)
    hourly["Дата"] = pd.to_datetime(hourly["Дата"], format="%d.%m.%Y")

    print(f"  Hourly rows: {len(hourly):,}")
    print(f"  Bakeries: {hourly['Пекарня'].nunique()}, "
          f"Products: {hourly['Номенклатура'].nunique()}, "
          f"Days: {hourly['Дата'].nunique()}")
    print(f"  Time: {time.time() - t0:.0f}s")

    return hourly


def step2_last_hour(hourly):
    """For each (bakery, product, date): find last hour of sale."""
    print("\n" + "=" * 60)
    print("  STEP 2: Last sale hour per (bakery, product, date)")
    print("=" * 60)

    last_hour = (hourly.groupby(["Пекарня", "Номенклатура", "Дата"])["hour"]
                 .max()
                 .reset_index()
                 .rename(columns={"hour": "last_hour"}))

    n_full = (last_hour["last_hour"] >= FULL_DAY_THRESHOLD).sum()
    n_total = len(last_hour)
    print(f"  Total (bakery, product, date) combos: {n_total:,}")
    print(f"  Full days (last_hour >= {FULL_DAY_THRESHOLD}): {n_full:,} ({100*n_full/n_total:.1f}%)")
    print(f"  Early stop: {n_total - n_full:,} ({100*(n_total-n_full)/n_total:.1f}%)")

    return last_hour


def step3_build_profiles(hourly, last_hour):
    """Build cumulative demand profiles per (bakery, product) from full days.

    Vectorized: computes cumulative shares via groupby cumsum,
    then pivots to (pair x hour) and averages across days.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: Build cumulative profiles (vectorized)")
    print("=" * 60)
    t0 = time.time()

    # Merge hourly with last_hour to know which days are full
    hourly_ext = hourly.merge(
        last_hour, on=["Пекарня", "Номенклатура", "Дата"], how="left"
    )
    full_days = hourly_ext[
        hourly_ext["last_hour"] >= FULL_DAY_THRESHOLD
    ].copy()
    print(f"  Full-day hourly rows: {len(full_days):,}")

    # Count full days per (bakery, product) — filter eligible
    full_day_counts = (
        last_hour[last_hour["last_hour"] >= FULL_DAY_THRESHOLD]
        .groupby(["Пекарня", "Номенклатура"])
        .size().rename("n_full_days").reset_index()
    )
    eligible = full_day_counts[
        full_day_counts["n_full_days"] >= MIN_FULL_DAYS
    ]
    print(f"  Eligible pairs (>= {MIN_FULL_DAYS} full days): "
          f"{len(eligible):,}")

    # Keep only eligible pairs
    full_days = full_days.merge(
        eligible[["Пекарня", "Номенклатура"]],
        on=["Пекарня", "Номенклатура"], how="inner"
    )
    print(f"  Eligible full-day hourly rows: {len(full_days):,}")

    # --- Vectorized cumulative profile computation ---
    # 1) Day totals
    day_key = ["Пекарня", "Номенклатура", "Дата"]
    daily_totals = (full_days.groupby(day_key)["Кол-во"]
                    .transform("sum"))
    # Skip zero-total days
    full_days = full_days[daily_totals > 0].copy()
    daily_totals = daily_totals[daily_totals > 0]

    # 2) Hourly share = qty / day_total
    full_days["share"] = full_days["Кол-во"].values / daily_totals.values

    # 3) Cumulative share within each (bakery, product, date)
    #    Sort by hour first, then cumsum within group
    full_days = full_days.sort_values(
        day_key + ["hour"]
    ).reset_index(drop=True)
    full_days["cum_share"] = full_days.groupby(day_key)["share"].cumsum()

    # 4) Average cum_share across days -> per (bakery, product, hour)
    avg_cum = (full_days
               .groupby(["Пекарня", "Номенклатура", "hour"])["cum_share"]
               .mean()
               .reset_index())

    # 5) Forward-fill missing hours within each pair
    #    Pivot to (pair) x (hour), ffill along hours, melt back
    avg_cum["pair_key"] = (avg_cum["Пекарня"].astype(str) + "|"
                           + avg_cum["Номенклатура"].astype(str))

    # Build profiles dict from grouped data
    profiles = {}
    n_built = 0
    for pair_key, grp in avg_cum.groupby("pair_key"):
        cum_series = grp.set_index("hour")["cum_share"].sort_index()
        # Reindex to fill gaps, ffill
        all_hours = range(int(cum_series.index.min()),
                          int(cum_series.index.max()) + 1)
        cum_series = cum_series.reindex(all_hours).ffill()
        profile_dict = {int(h): round(float(v), 6)
                        for h, v in cum_series.items()
                        if pd.notna(v)}
        profiles[pair_key] = profile_dict
        n_built += 1

    elapsed = time.time() - t0
    print(f"  Profiles built: {n_built:,}")
    print(f"  Time: {elapsed:.0f}s")

    return profiles


def step4_estimate_demand(hourly, last_hour, profiles):
    """Estimate demand for each (bakery, product, date) using profiles.

    Vectorized approach:
    1) Expand each profile to all hours 0-23 (with ffill) -> flat lookup
    2) Merge daily with lookup on (bakery, product, last_hour) -> O(N)
    3) Compute demand with numpy vectorized ops
    """
    print("\n" + "=" * 60)
    print("  STEP 4: Estimate demand (vectorized)")
    print("=" * 60)
    t0 = time.time()

    import numpy as np

    # Daily actual sales
    daily = (hourly.groupby(["Пекарня", "Номенклатура", "Дата"])["Кол-во"]
             .sum().rename("actual_sold").reset_index())
    daily = daily.merge(
        last_hour, on=["Пекарня", "Номенклатура", "Дата"], how="left"
    )
    daily["last_hour"] = daily["last_hour"].astype(int)
    print(f"  Daily rows: {len(daily):,}")

    # --- Build flat lookup: (bakery, product, hour) -> cum_share ---
    # For each profile, expand to hours 0..23 with ffill so that
    # any last_hour can be looked up directly via merge.
    print("  Building profile lookup (0-23h per pair)...", flush=True)
    all_hours = list(range(24))
    prof_rows = []
    for pair_key, hour_dict in profiles.items():
        parts = pair_key.split("|", 1)
        bakery, product = parts[0], parts[1]
        # Build full 0-23 series with ffill
        cum_val = float("nan")
        for h in all_hours:
            if h in hour_dict:
                cum_val = hour_dict[h]
            # cum_val is ffilled from last known hour
            if not (cum_val != cum_val):  # not NaN
                prof_rows.append((bakery, product, h, cum_val))

    prof_df = pd.DataFrame(
        prof_rows,
        columns=["Пекарня", "Номенклатура", "hour", "cum_at_stop"]
    )
    print(f"  Profile lookup rows: {len(prof_df):,} "
          f"({len(profiles)} pairs x up to 24h)")

    # --- Single merge on (bakery, product, last_hour=hour) ---
    daily = daily.merge(
        prof_df,
        left_on=["Пекарня", "Номенклатура", "last_hour"],
        right_on=["Пекарня", "Номенклатура", "hour"],
        how="left",
    )
    # Drop extra 'hour' column from profile
    daily.drop(columns=["hour"], inplace=True, errors="ignore")

    print(f"  Matched with profile: "
          f"{daily['cum_at_stop'].notna().sum():,} / {len(daily):,}")

    # --- Vectorized demand calculation ---
    actual = daily["actual_sold"].values.astype(float)
    cum = daily["cum_at_stop"].values.astype(float)
    is_full = daily["last_hour"].values >= FULL_DAY_THRESHOLD

    # Mask: has profile, not full day, cum >= floor
    can_adjust = (~np.isnan(cum) & ~is_full & (cum >= CUM_FLOOR))

    # estimated = actual / cum_at_stop, with cap x5, floor = actual
    est_raw = np.where(can_adjust, actual / cum, actual)
    est_capped = np.minimum(est_raw, actual * MAX_MULTIPLIER)
    estimated = np.where(can_adjust,
                         np.maximum(est_capped, actual), actual)
    lost_qty = np.where(can_adjust, estimated - actual, 0.0)
    is_censored = np.where(can_adjust, 1, 0)

    daily["demand_estimated"] = np.round(estimated, 2)
    daily["lost_qty"] = np.round(lost_qty, 2)
    daily["is_censored"] = is_censored

    n_censored = int(is_censored.sum())
    print(f"  Total rows: {len(daily):,}")
    print(f"  Censored (adjusted): {n_censored:,} "
          f"({100*n_censored/len(daily):.1f}%)")
    print(f"  Mean demand: {daily['demand_estimated'].mean():.2f}")
    cens_mask = daily["is_censored"] == 1
    if cens_mask.any():
        print(f"  Mean lost_qty (censored): "
              f"{daily.loc[cens_mask, 'lost_qty'].mean():.2f}")
    print(f"  Time: {time.time() - t0:.0f}s")

    return daily[["Пекарня", "Номенклатура", "Дата",
                   "demand_estimated", "lost_qty", "is_censored"]]


def step5_merge_with_daily(demand_df):
    """Merge demand estimates with daily_sales_8m.csv and add demand lags/rolling."""
    print("\n" + "=" * 60)
    print("  STEP 5: Merge with daily_sales_8m and add demand features")
    print("=" * 60)
    t0 = time.time()

    # Load daily_sales_8m
    print(f"  Loading {DAILY_8M_PATH}...")
    daily = pd.read_csv(str(DAILY_8M_PATH), encoding="utf-8-sig")
    daily["Дата"] = pd.to_datetime(daily["Дата"])
    print(f"  daily_sales_8m: {len(daily):,} rows")

    # Ensure demand_df dates match
    demand_df["Дата"] = pd.to_datetime(demand_df["Дата"])

    # Merge
    before = len(daily)
    daily = daily.merge(
        demand_df[["Пекарня", "Номенклатура", "Дата", "demand_estimated", "lost_qty", "is_censored"]],
        on=["Пекарня", "Номенклатура", "Дата"],
        how="left",
    )
    after = len(daily)
    print(f"  Merge: {before} -> {after} rows")

    # Fallback: rows without demand estimate -> demand = sold
    n_null = daily["demand_estimated"].isna().sum()
    print(f"  Rows without demand estimate: {n_null:,} ({100*n_null/len(daily):.1f}%) -> fallback to Prodano")
    daily["demand_estimated"] = daily["demand_estimated"].fillna(daily[TARGET])
    daily["lost_qty"] = daily["lost_qty"].fillna(0.0)
    daily["is_censored"] = daily["is_censored"].fillna(0).astype(int)

    # Create demand target column
    daily["Спрос"] = daily["demand_estimated"]

    # --- Demand lags and rolling features ---
    print("  Computing demand lags and rolling features...")
    daily = daily.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grouped = daily.groupby(["Пекарня", "Номенклатура"])["Спрос"]

    # Lags
    for lag in [1, 2, 3, 7, 14, 30]:
        col = f"demand_lag{lag}"
        daily[col] = grouped.shift(lag)
        print(f"    {col}: done", flush=True)

    # Rolling means (shift(1) to prevent leakage)
    for window in [3, 7, 14, 30]:
        col = f"demand_roll_mean{window}"
        min_p = max(1, window // 2)
        daily[col] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_p).mean()
        )
        print(f"    {col}: done", flush=True)

    # Rolling std
    daily["demand_roll_std7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )
    print("    demand_roll_std7: done", flush=True)

    # Drop rows with NaN in demand lag features (warmup)
    demand_lag_cols = [c for c in daily.columns if c.startswith("demand_lag") or c.startswith("demand_roll")]
    n_before = len(daily)
    daily = daily.dropna(subset=demand_lag_cols).reset_index(drop=True)
    n_dropped = n_before - len(daily)
    print(f"  Dropped {n_dropped:,} warmup rows for demand lags")

    elapsed = time.time() - t0
    print(f"  Final shape: {daily.shape}")
    print(f"  Time: {elapsed:.0f}s")

    return daily


def main():
    print("=" * 60)
    print("  BUILD DEMAND PROFILES (Experiment 03)")
    print("=" * 60)
    t_start = time.time()

    # Step 1: Hourly aggregation from raw checks
    hourly = step1_hourly_aggregation()

    # Step 2: Last hour per (bakery, product, date)
    last_hour = step2_last_hour(hourly)

    # Step 3: Build cumulative profiles
    profiles = step3_build_profiles(hourly, last_hour)

    # Step 4: Estimate demand
    demand_df = step4_estimate_demand(hourly, last_hour, profiles)

    # Step 5: Merge with daily_sales_8m and add demand features
    daily = step5_merge_with_daily(demand_df)

    # --- Save profiles ---
    print("\n" + "=" * 60)
    print("  SAVING")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_PROFILES), exist_ok=True)

    with open(OUTPUT_PROFILES, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False)
    print(f"  Profiles: {OUTPUT_PROFILES} ({len(profiles)} pairs)")

    daily.to_csv(OUTPUT_DEMAND_CSV, index=False, encoding="utf-8-sig")
    print(f"  Demand CSV: {OUTPUT_DEMAND_CSV}")
    print(f"  Shape: {daily.shape}")

    # --- Verification ---
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)
    mean_sold = daily[TARGET].mean()
    mean_demand = daily["Спрос"].mean()
    uplift = (mean_demand - mean_sold) / mean_sold * 100

    print(f"  mean({TARGET}):  {mean_sold:.4f}")
    print(f"  mean(Spros):   {mean_demand:.4f}")
    print(f"  Demand uplift: {uplift:+.2f}%")
    print(f"  mean(lost_qty): {daily['lost_qty'].mean():.4f}")
    print(f"  % censored:    {100*daily['is_censored'].mean():.1f}%")
    print(f"  Demand > Sold:  {(daily['Спрос'] > daily[TARGET]).sum():,} rows")

    # Per-category stats
    print("\n  Per-category demand uplift:")
    for cat in sorted(daily["Категория"].unique()):
        mask = daily["Категория"] == cat
        ms = daily.loc[mask, TARGET].mean()
        md = daily.loc[mask, "Спрос"].mean()
        up = (md - ms) / ms * 100 if ms > 0 else 0
        cens = 100 * daily.loc[mask, "is_censored"].mean()
        print(f"    {str(cat):<25} sold={ms:.2f}  demand={md:.2f}  uplift={up:+.1f}%  censored={cens:.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
