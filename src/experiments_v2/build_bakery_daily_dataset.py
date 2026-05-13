"""
Build bakery-level daily dataset from raw check lines using chunked processing.

Input:
  data/raw/sales_hrs_all.csv (or another CSV with the normalized English schema)

Required raw columns:
  check_date
  cash_event_type
  quantity
  bakery_id
  bakery_name
  city

Optional raw columns:
  price
  line_amount

Output:
  data/processed/bakery_daily_sales.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments_v2.raw_snapshot_schema import normalize_snapshot_chunk
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_date


ROOT = Path(__file__).resolve().parents[2]

RAW_DATE_COL = "check_date"
RAW_EVENT_COL = "cash_event_type"
RAW_QTY_COL = "quantity"
RAW_PRICE_COL = "price"
RAW_AMOUNT_COL = "line_amount"
RAW_BAKERY_ID_COL = "bakery_id"
RAW_BAKERY_NAME_COL = "bakery_name"
RAW_CITY_COL = "city"

DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
TARGET_COL = "bakery_sales"

SALES_EVENT = "Продажа"
CHUNK_SIZE = 1_000_000
SALES_EVENTS = {"Продажа", SALES_EVENT}

OUTPUT_NAME = "bakery_daily_sales.csv"
SUMMARY_OUTPUT_NAME = "bakery_daily_sales_summary.json"


def aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = normalize_snapshot_chunk(chunk)
    sales = chunk[chunk[RAW_EVENT_COL].isin(SALES_EVENTS)].copy()
    if sales.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                CITY_COL,
                TARGET_COL,
                "line_amount_sum",
                "priced_quantity",
                "price_x_qty_sum",
            ]
        )

    sales[DATE_COL] = parse_snapshot_date(sales[RAW_DATE_COL]).dt.normalize()
    sales = sales.dropna(subset=[DATE_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL])
    if sales.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                CITY_COL,
                TARGET_COL,
                "line_amount_sum",
                "priced_quantity",
                "price_x_qty_sum",
            ]
        )

    sales[RAW_QTY_COL] = pd.to_numeric(sales[RAW_QTY_COL], errors="coerce").fillna(0.0)
    sales[RAW_QTY_COL] = sales[RAW_QTY_COL].clip(lower=0.0)

    if RAW_AMOUNT_COL in sales.columns:
        sales[RAW_AMOUNT_COL] = pd.to_numeric(sales[RAW_AMOUNT_COL], errors="coerce").fillna(0.0)
    else:
        sales[RAW_AMOUNT_COL] = 0.0

    if RAW_PRICE_COL in sales.columns:
        sales[RAW_PRICE_COL] = pd.to_numeric(sales[RAW_PRICE_COL], errors="coerce")
        sales["price_x_qty_sum"] = sales[RAW_PRICE_COL].fillna(0.0) * sales[RAW_QTY_COL]
        sales["priced_quantity"] = np.where(sales[RAW_PRICE_COL].notna(), sales[RAW_QTY_COL], 0.0)
    else:
        sales["price_x_qty_sum"] = 0.0
        sales["priced_quantity"] = 0.0

    grouped = (
        sales.groupby([DATE_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL, RAW_CITY_COL], as_index=False)
        .agg(
            bakery_sales=(RAW_QTY_COL, "sum"),
            line_amount_sum=(RAW_AMOUNT_COL, "sum"),
            priced_quantity=("priced_quantity", "sum"),
            price_x_qty_sum=("price_x_qty_sum", "sum"),
        )
        .rename(
            columns={
                RAW_BAKERY_ID_COL: BAKERY_ID_COL,
                RAW_BAKERY_NAME_COL: BAKERY_NAME_COL,
                RAW_CITY_COL: CITY_COL,
            }
        )
    )
    return grouped


def merge_partial_results(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    daily = pd.concat(parts, ignore_index=True)
    daily = (
        daily.groupby([DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL], as_index=False)
        .agg(
            bakery_sales=(TARGET_COL, "sum"),
            line_amount_sum=("line_amount_sum", "sum"),
            priced_quantity=("priced_quantity", "sum"),
            price_x_qty_sum=("price_x_qty_sum", "sum"),
        )
        .sort_values([BAKERY_ID_COL, DATE_COL])
        .reset_index(drop=True)
    )
    return daily


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["dow"] = work[DATE_COL].dt.dayofweek
    work["day"] = work[DATE_COL].dt.day
    work["month"] = work[DATE_COL].dt.month
    work["iso_week"] = work[DATE_COL].dt.isocalendar().week.astype(int)
    work["is_weekend"] = (work["dow"] >= 5).astype(int)
    work["is_month_start"] = (work["day"] <= 5).astype(int)
    work["is_month_end"] = (work["day"] >= 25).astype(int)
    work["is_payday_week"] = work["day"].isin([4, 5, 6, 19, 20, 21]).astype(int)
    return work


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["avg_price"] = np.where(
        work["priced_quantity"] > 0,
        work["price_x_qty_sum"] / work["priced_quantity"],
        np.nan,
    )
    return work


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    grouped = work.groupby(BAKERY_ID_COL)[TARGET_COL]
    for lag in [1, 2, 3, 7, 14, 30]:
        work[f"bakery_sales_lag{lag}"] = grouped.shift(lag)

    for window, min_periods in [(3, 1), (7, 1), (14, 7), (30, 14)]:
        work[f"bakery_sales_roll_mean{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )
    work["bakery_sales_roll_std7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )
    return work


def build_bakery_daily_dataset(source_path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    usecols = [
        RAW_DATE_COL,
        RAW_EVENT_COL,
        RAW_QTY_COL,
        RAW_BAKERY_ID_COL,
        RAW_BAKERY_NAME_COL,
        RAW_CITY_COL,
        RAW_PRICE_COL,
        RAW_AMOUNT_COL,
        "Дата продажи",
        "Вид события по кассе",
        "Касса.Торговая точка",
        "Цена",
        "Кол-во",
    ]
    parts: list[pd.DataFrame] = []

    reader = pd.read_csv(source_path, encoding="utf-8-sig", usecols=lambda c: c in usecols, chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        part = aggregate_chunk(chunk)
        parts.append(part)
        if i % 5 == 0:
            print(f"processed chunks: {i}", flush=True)

    daily = merge_partial_results(parts)
    daily = add_price_features(daily)
    daily = add_calendar_features(daily)
    daily = add_lag_features(daily)
    return daily


def build_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "date_min": None if df.empty else str(df[DATE_COL].min().date()),
        "date_max": None if df.empty else str(df[DATE_COL].max().date()),
        "dates": int(df[DATE_COL].nunique()) if len(df) else 0,
        "bakeries": int(df[BAKERY_ID_COL].nunique()) if len(df) else 0,
        "cities": int(df[CITY_COL].nunique()) if len(df) else 0,
        "mean_bakery_sales": round(float(df[TARGET_COL].mean()), 6) if len(df) else 0.0,
    }


def save_outputs(output_dir: str | Path, df: pd.DataFrame, summary: dict) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": csv_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bakery-level daily dataset from raw checks")
    parser.add_argument("--source-path", default=str(ROOT / "data" / "raw" / "sales_hrs_all.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    df = build_bakery_daily_dataset(args.source_path, chunk_size=args.chunk_size)
    summary = build_summary(df)
    paths = save_outputs(args.output_dir, df, summary)

    print("=" * 72)
    print("BAKERY DAILY DATASET")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
