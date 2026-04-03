"""
preprocess_v2.py -- Aggregate 30M check-level rows to daily sales CSV.

Input:  data/raw/sales_hrs_all.csv  (~4.6 GB, 30M rows)
Output: data/processed/daily_sales_8m.csv (~50-200 MB)

Steps:
  1. Chunked reading (2M rows) with usecols (6 of 9 columns)
  2. Filter: only 'Продажа' events (skip returns/cancels)
  3. Aggregate per chunk: groupby(Дата, Пекарня, Категория, Номенклатура).sum(Кол-во)
  4. Rename columns, extract city from bakery name
  5. Clip Продано >= 0
  6. Calendar features (ДеньНедели, День, IsWeekend, Месяц, etc.)
  7. Lag + rolling features (shift(1) to prevent leakage)
  8. Weather from Open-Meteo (15 features for 9 cities)
  9. Drop NaN warmup rows
  10. Save to CSV

Usage:
  .venv/Scripts/python.exe src/experiments_v2/preprocess_v2.py
"""

import sys
import os
import time

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.experiments_v2.common import (
    SALES_HRS_PATH, DAILY_8M_PATH, TARGET,
    CITY_COORDS, extract_city,
)
from src.features.fetch_weather import fetch_weather, enrich_weather

CHUNK_SIZE = 2_000_000
USE_COLS = ["Дата продажи", "Вид события по кассе", "Касса.Торговая точка",
            "Категория", "Номенклатура", "Кол-во"]

# Russian holidays (month, day)
RU_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}


def step1_aggregate_chunks(input_path):
    """Read CSV in chunks, filter sales, aggregate to daily level."""
    print("=" * 60)
    print("  STEP 1: Chunked aggregation")
    print("=" * 60)

    agg_chunks = []
    total_rows = 0
    total_sales = 0
    total_skipped = 0

    t0 = time.time()
    reader = pd.read_csv(
        input_path, encoding="utf-8-sig",
        usecols=USE_COLS, chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader, 1):
        n = len(chunk)
        total_rows += n

        # Filter: keep only 'Продажа'
        sales = chunk[chunk["Вид события по кассе"] == "Продажа"].copy()
        skipped = n - len(sales)
        total_skipped += skipped
        total_sales += len(sales)

        # Drop event type column, no longer needed
        sales.drop(columns=["Вид события по кассе"], inplace=True)

        # Aggregate within chunk
        agg = (sales.groupby(["Дата продажи", "Касса.Торговая точка",
                              "Категория", "Номенклатура"])["Кол-во"]
               .sum()
               .reset_index())
        agg_chunks.append(agg)

        elapsed = time.time() - t0
        print(f"  Chunk {i}: {n:>10,} rows, {len(sales):>10,} sales, "
              f"{len(agg):>8,} aggregated  ({elapsed:.0f}s)", flush=True)

    print(f"\n  Total: {total_rows:,} rows read, {total_sales:,} sales, "
          f"{total_skipped:,} skipped (returns/cancels)")

    # Final aggregation across chunks
    print("  Merging chunks...", flush=True)
    daily = pd.concat(agg_chunks, ignore_index=True)
    daily = (daily.groupby(["Дата продажи", "Касса.Торговая точка",
                            "Категория", "Номенклатура"])["Кол-во"]
             .sum()
             .reset_index())

    # Rename columns
    daily.rename(columns={
        "Дата продажи": "Дата",
        "Касса.Торговая точка": "Пекарня",
        "Кол-во": TARGET,
    }, inplace=True)

    # Parse dates
    daily["Дата"] = pd.to_datetime(daily["Дата"], format="%d.%m.%Y")

    print(f"  Result: {len(daily):,} rows, "
          f"{daily['Дата'].nunique()} days, "
          f"{daily['Пекарня'].nunique()} bakeries, "
          f"{daily['Номенклатура'].nunique()} products, "
          f"{daily['Категория'].nunique()} categories")
    print(f"  Date range: {daily['Дата'].min().date()} -- {daily['Дата'].max().date()}")

    return daily


def step2_extract_city(df):
    """Extract city from bakery name."""
    print("\n" + "=" * 60)
    print("  STEP 2: Extract city from bakery names")
    print("=" * 60)

    df["Город"] = df["Пекарня"].apply(extract_city)

    city_counts = df.groupby("Город")["Пекарня"].nunique()
    for city, count in city_counts.items():
        print(f"  {city}: {count} bakeries")

    unknown = df[df["Город"] == "Казань"]["Пекарня"].nunique()
    print(f"  (default to Kazan: {unknown} bakeries)")

    return df


def step3_clip_and_filter(df):
    """Clip negative sales, drop rows with NaN category."""
    print("\n" + "=" * 60)
    print("  STEP 3: Clip & filter")
    print("=" * 60)

    neg = (df[TARGET] < 0).sum()
    df[TARGET] = df[TARGET].clip(lower=0)
    print(f"  Negative {TARGET} clipped: {neg}")

    nan_cat = df["Категория"].isna().sum()
    if nan_cat > 0:
        df = df.dropna(subset=["Категория"])
        print(f"  Dropped {nan_cat} rows with NaN Kategoriya")

    print(f"  Rows after filter: {len(df):,}")
    return df


def step4_calendar_features(df):
    """Add calendar features."""
    print("\n" + "=" * 60)
    print("  STEP 4: Calendar features")
    print("=" * 60)

    df["ДеньНедели"] = df["Дата"].dt.dayofweek
    df["День"] = df["Дата"].dt.day
    df["IsWeekend"] = (df["ДеньНедели"] >= 5).astype(int)
    df["Месяц"] = df["Дата"].dt.month
    df["НомерНедели"] = df["Дата"].dt.isocalendar().week.astype(int)

    # Holidays
    date_md = list(zip(df["Дата"].dt.month, df["Дата"].dt.day))
    df["is_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in date_md]

    # Pre/post holiday
    next_day = df["Дата"] + pd.Timedelta(days=1)
    prev_day = df["Дата"] - pd.Timedelta(days=1)
    next_md = list(zip(next_day.dt.month, next_day.dt.day))
    prev_md = list(zip(prev_day.dt.month, prev_day.dt.day))
    df["is_pre_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in next_md]
    df["is_post_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in prev_md]

    # Payday week
    df["is_payday_week"] = df["День"].isin([4, 5, 6, 19, 20, 21]).astype(int)

    # Month start/end
    df["is_month_start"] = (df["День"] <= 5).astype(int)
    df["is_month_end"] = (df["День"] >= 25).astype(int)

    n_holidays = df["is_holiday"].sum()
    print(f"  Holiday rows: {n_holidays:,}")
    print(f"  Calendar features added: DenNedeli, Den, IsWeekend, Mesyac, "
          "NomerNedeli, is_holiday, is_pre/post_holiday, is_payday_week, "
          "is_month_start/end")

    return df


def step5_lag_and_rolling(df):
    """Add lag and rolling features per bakery x product series."""
    print("\n" + "=" * 60)
    print("  STEP 5: Lag & rolling features")
    print("=" * 60)

    t0 = time.time()

    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grouped = df.groupby(["Пекарня", "Номенклатура"])[TARGET]

    # Lags
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"sales_lag{lag}"] = grouped.shift(lag)
        print(f"  sales_lag{lag}: done", flush=True)

    # Rolling means (shift(1) to prevent leakage)
    for window in [3, 7, 14, 30]:
        col_name = f"sales_roll_mean{window}"
        min_p = max(1, window // 2)
        df[col_name] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_p).mean()
        )
        print(f"  {col_name}: done", flush=True)

    # Rolling std
    df["sales_roll_std7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )
    print(f"  sales_roll_std7: done", flush=True)

    elapsed = time.time() - t0
    print(f"  Lag/rolling features completed in {elapsed:.0f}s")

    return df


def step6_weather(df):
    """Fetch and merge weather data for 9 cities."""
    print("\n" + "=" * 60)
    print("  STEP 6: Weather features (Open-Meteo)")
    print("=" * 60)

    start_date = df["Дата"].min().strftime("%Y-%m-%d")
    end_date = df["Дата"].max().strftime("%Y-%m-%d")
    print(f"  Date range: {start_date} -- {end_date}")
    print(f"  Cities: {list(CITY_COORDS.keys())}")

    weather_df = fetch_weather(CITY_COORDS, start_date, end_date)

    if weather_df.empty:
        print("  WARNING: Weather not loaded! Skipping weather features.")
        # Fill weather columns with 0
        weather_cols = ["temp_max", "temp_min", "temp_mean", "temp_range",
                        "precipitation", "rain", "snowfall", "windspeed_max",
                        "is_rainy", "is_snowy", "is_cold", "is_warm",
                        "is_windy", "is_bad_weather", "weather_cat_code"]
        for col in weather_cols:
            df[col] = 0
        return df

    weather_df = enrich_weather(weather_df)

    # Encode weather_category -> weather_cat_code
    weather_cat_map = {
        "clear": 0, "cloudy": 1, "fog": 2, "rain": 3,
        "showers": 4, "snow": 5, "storm": 6,
    }
    weather_df["weather_cat_code"] = (weather_df["weather_category"]
                                       .map(weather_cat_map).fillna(0).astype(int))
    weather_df.drop(columns=["weather_category"], inplace=True)

    # Drop weathercode (raw WMO code, replaced by weather_cat_code)
    if "weathercode" in weather_df.columns:
        weather_df.drop(columns=["weathercode"], inplace=True)

    print(f"  Weather loaded: {len(weather_df):,} rows for "
          f"{weather_df['Город'].nunique()} cities")

    # Merge
    df["Город"] = df["Город"].astype(str)
    weather_df["Город"] = weather_df["Город"].astype(str)
    weather_df["Дата"] = pd.to_datetime(weather_df["Дата"]).dt.normalize()

    before = len(df)
    df = df.merge(weather_df, on=["Дата", "Город"], how="left")
    after = len(df)
    print(f"  Merged: {before} -> {after} rows")

    # Check nulls in weather columns
    weather_cols = [c for c in weather_df.columns if c not in ["Дата", "Город"]]
    for col in weather_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"    WARNING: {col} has {n_null} nulls (filling with 0)")
            df[col] = df[col].fillna(0)

    return df


def step7_drop_warmup(df):
    """Drop NaN rows from warmup period (first ~30 days per series)."""
    print("\n" + "=" * 60)
    print("  STEP 7: Drop warmup NaN rows")
    print("=" * 60)

    before = len(df)

    # Only check lag/rolling columns for NaN (not all columns)
    lag_cols = [c for c in df.columns if c.startswith("sales_lag") or c.startswith("sales_roll")]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    dropped = before - len(df)
    print(f"  Dropped {dropped:,} warmup rows ({dropped/before*100:.1f}%)")
    print(f"  Remaining: {len(df):,} rows, {df['Дата'].nunique()} days")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")

    return df


def main():
    print("=" * 60)
    print("  PREPROCESS V2: 30M checks -> daily sales CSV")
    print("=" * 60)
    t_start = time.time()

    input_path = str(SALES_HRS_PATH)
    output_path = str(DAILY_8M_PATH)

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print()

    # Step 1: Chunked aggregation
    df = step1_aggregate_chunks(input_path)

    # Step 2: Extract city
    df = step2_extract_city(df)

    # Step 3: Clip & filter
    df = step3_clip_and_filter(df)

    # Step 4: Calendar features
    df = step4_calendar_features(df)

    # Step 5: Lag & rolling features
    df = step5_lag_and_rolling(df)

    # Step 6: Weather
    df = step6_weather(df)

    # Step 7: Drop warmup NaN
    df = step7_drop_warmup(df)

    # Save
    print("\n" + "=" * 60)
    print("  SAVING")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t_start
    print(f"  Saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Total time: {elapsed:.0f}s")

    # Verification
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)
    print(f"  Dates: {df['Дата'].nunique()}")
    print(f"  Bakeries: {df['Пекарня'].nunique()}")
    print(f"  Products: {df['Номенклатура'].nunique()}")
    print(f"  Categories: {df['Категория'].nunique()}")
    print(f"  Cities: {df['Город'].nunique()} -> {sorted(df['Город'].unique())}")
    print(f"  NaN in features: {df.isna().sum().sum()}")
    print(f"  Negative {TARGET}: {(df[TARGET] < 0).sum()}")
    print(f"  {TARGET} stats: min={df[TARGET].min():.1f}, "
          f"mean={df[TARGET].mean():.2f}, max={df[TARGET].max():.1f}")


if __name__ == "__main__":
    main()
