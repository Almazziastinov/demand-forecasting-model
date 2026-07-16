"""
Build SKU-level daily training dataset from stg_check_lines export CSV.

Replaces the old Excel-based preprocessing pipeline for experiments_v2.

Input:
  data/raw/sales_stg_2025_2026.csv        (stg_check_lines export)
  data/processed/bakery_weather_features.csv

Output:
  data/processed/daily_sales_stg.csv      (equivalent to daily_sales_8m_demand.csv)

Usage:
  .venv/Scripts/python.exe -m src.experiments_v2.build_sku_daily_from_stg
  .venv/Scripts/python.exe -m src.experiments_v2.build_sku_daily_from_stg \
      --sales-path data/raw/sales_stg_2025_2026.csv \
      --output data/processed/daily_sales_stg.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.raw_sales_dedup import deduplicate_sales_chunk

CHUNK_SIZE = 1_000_000

DEFAULT_SALES = ROOT / "data" / "raw" / "sales_stg_2025_2026.csv"
DEFAULT_WEATHER = ROOT / "data" / "processed" / "bakery_weather_features.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "daily_sales_stg.csv"

RU_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}

LAG_COLS = [1, 2, 3, 7, 14, 30]
ROLL_COLS = [3, 7, 14, 30]


def add_calendar(df: pd.DataFrame, date_col: str = "Дата") -> pd.DataFrame:
    d = df[date_col]
    df["ДеньНедели"] = d.dt.dayofweek
    df["День"] = d.dt.day
    df["IsWeekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["Месяц"] = d.dt.month
    df["НомерНедели"] = d.dt.isocalendar().week.astype(int)
    df["is_holiday"] = d.apply(lambda x: int((x.month, x.day) in RU_HOLIDAYS))
    df["is_pre_holiday"] = d.apply(
        lambda x: int(((x + pd.Timedelta(days=1)).month,
                       (x + pd.Timedelta(days=1)).day) in RU_HOLIDAYS)
    )
    df["is_post_holiday"] = d.apply(
        lambda x: int(((x - pd.Timedelta(days=1)).month,
                       (x - pd.Timedelta(days=1)).day) in RU_HOLIDAYS)
    )
    df["is_payday_week"] = d.dt.day.isin([4, 5, 6, 19, 20, 21]).astype(int)
    df["is_month_start"] = (d.dt.day <= 3).astype(int)
    df["is_month_end"] = (d.dt.day >= 28).astype(int)
    return df


def add_lags_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grp = df.groupby(["Пекарня", "Номенклатура"])["Продано"]
    for lag in LAG_COLS:
        df[f"sales_lag{lag}"] = grp.shift(lag)
    for w in ROLL_COLS:
        df[f"sales_roll_mean{w}"] = grp.shift(1).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
    df["sales_roll_std7"] = grp.shift(1).transform(
        lambda s: s.rolling(7, min_periods=2).std()
    )
    return df


def build(sales_path: Path, weather_path: Path, output_path: Path) -> None:
    print(f"Reading {sales_path} ...", flush=True)

    chunks = []
    reader = pd.read_csv(
        sales_path,
        encoding="utf-8-sig",
        chunksize=CHUNK_SIZE,
        usecols=["check_date", "cash_event_type", "quantity",
                 "bakery_name", "city", "product_name", "category_name"],
        low_memory=False,
    )
    for i, chunk in enumerate(reader, 1):
        chunk = chunk[chunk["cash_event_type"] == "Продажа"].copy()
        chunk["check_date"] = pd.to_datetime(chunk["check_date"], errors="coerce")
        chunk = chunk.dropna(subset=["check_date"])
        # Simple dedup: drop exact duplicates within chunk
        key_cols = ["check_date", "bakery_name", "product_name", "category_name", "quantity"]
        chunk = chunk.drop_duplicates(subset=key_cols)
        chunks.append(chunk)
        if i % 10 == 0:
            print(f"  chunk {i}", flush=True)

    raw = pd.concat(chunks, ignore_index=True)
    print(f"  raw rows after dedup: {len(raw):,}", flush=True)

    # Aggregate to SKU-day level
    agg = (
        raw.groupby(["check_date", "bakery_name", "city", "category_name", "product_name"],
                    as_index=False)["quantity"]
        .sum()
        .rename(columns={
            "check_date": "Дата",
            "bakery_name": "Пекарня",
            "city": "Город",
            "category_name": "Категория",
            "product_name": "Номенклатура",
            "quantity": "Продано",
        })
    )
    print(f"  SKU-day rows: {len(agg):,}  dates: {agg['Дата'].nunique()}  "
          f"bakeries: {agg['Пекарня'].nunique()}  products: {agg['Номенклатура'].nunique()}",
          flush=True)

    # Calendar features
    agg = add_calendar(agg)

    # Weather features
    if weather_path.exists():
        wx = pd.read_csv(weather_path, encoding="utf-8-sig")
        wx["date"] = pd.to_datetime(wx["date"])
        wx = wx.rename(columns={"date": "Дата", "city": "Город"})
        agg = agg.merge(wx, on=["Дата", "Город"], how="left")
        print(f"  weather merged: {agg['is_bad_weather'].notna().sum():,} rows with weather",
              flush=True)
    else:
        print(f"  WARNING: weather file not found: {weather_path}", flush=True)

    # Lag and rolling features
    print("  building lag/rolling features ...", flush=True)
    agg = add_lags_rolling(agg)

    # Drop rows with missing lags (first days of each series)
    before = len(agg)
    agg = agg.dropna(subset=[f"sales_lag{l}" for l in LAG_COLS]).reset_index(drop=True)
    print(f"  dropped {before - len(agg):,} rows with NaN lags", flush=True)
    print(f"  final rows: {len(agg):,}  dates: {agg['Дата'].min().date()} .. {agg['Дата'].max().date()}",
          flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SKU daily dataset from stg export")
    p.add_argument("--sales-path", default=str(DEFAULT_SALES))
    p.add_argument("--weather-path", default=str(DEFAULT_WEATHER))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(Path(args.sales_path), Path(args.weather_path), Path(args.output))
