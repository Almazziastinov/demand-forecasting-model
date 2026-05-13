"""
Build bakery-level hourly share profiles from raw check lines using chunked processing.

Input:
  data/raw/sales_hrs_all.csv (or another CSV with the normalized English schema)

Required raw columns:
  check_datetime
  check_date
  cash_event_type
  quantity
  bakery_id
  bakery_name

Output:
  data/processed/bakery_hour_profile.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments_v2.raw_snapshot_schema import normalize_snapshot_chunk
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_date
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_datetime


ROOT = Path(__file__).resolve().parents[2]

RAW_DATETIME_COL = "check_datetime"
RAW_DATE_COL = "check_date"
RAW_EVENT_COL = "cash_event_type"
RAW_QTY_COL = "quantity"
RAW_BAKERY_ID_COL = "bakery_id"
RAW_BAKERY_NAME_COL = "bakery_name"

DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
HOUR_COL = "hour"
DOW_COL = "dow"
QTY_COL = "bakery_hour_sales"

SALES_EVENT = "Продажа"
CHUNK_SIZE = 1_000_000
SALES_EVENTS = {"Продажа", SALES_EVENT}

OUTPUT_NAME = "bakery_hour_profile.csv"
APPLIED_DAILY_OUTPUT_NAME = "bakery_hour_profile_daily.csv"
SUMMARY_OUTPUT_NAME = "bakery_hour_profile_summary.json"


def aggregate_hourly_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = normalize_snapshot_chunk(chunk)
    sales = chunk[chunk[RAW_EVENT_COL].isin(SALES_EVENTS)].copy()
    if sales.empty:
        return pd.DataFrame(columns=[DATE_COL, DOW_COL, BAKERY_ID_COL, BAKERY_NAME_COL, HOUR_COL, QTY_COL])

    sales[DATE_COL] = parse_snapshot_date(sales[RAW_DATE_COL]).dt.normalize()
    sales["_dt"] = parse_snapshot_datetime(sales[RAW_DATETIME_COL])
    sales = sales.dropna(subset=[DATE_COL, "_dt", RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL])
    if sales.empty:
        return pd.DataFrame(columns=[DATE_COL, DOW_COL, BAKERY_ID_COL, BAKERY_NAME_COL, HOUR_COL, QTY_COL])

    sales[RAW_QTY_COL] = pd.to_numeric(sales[RAW_QTY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    sales[HOUR_COL] = sales["_dt"].dt.hour
    sales[DOW_COL] = sales[DATE_COL].dt.dayofweek

    grouped = (
        sales.groupby([DATE_COL, DOW_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL, HOUR_COL], as_index=False)
        .agg(bakery_hour_sales=(RAW_QTY_COL, "sum"))
        .rename(columns={RAW_BAKERY_ID_COL: BAKERY_ID_COL, RAW_BAKERY_NAME_COL: BAKERY_NAME_COL})
    )
    return grouped


def merge_hourly_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    hourly = pd.concat(parts, ignore_index=True)
    hourly = (
        hourly.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, BAKERY_NAME_COL, HOUR_COL], as_index=False)
        .agg(bakery_hour_sales=(QTY_COL, "sum"))
        .sort_values([BAKERY_ID_COL, DATE_COL, HOUR_COL])
        .reset_index(drop=True)
    )
    return hourly


def build_hour_profile(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_totals = (
        hourly.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, BAKERY_NAME_COL], as_index=False)
        .agg(bakery_day_sales=(QTY_COL, "sum"))
    )
    applied = hourly.merge(
        daily_totals,
        on=[DATE_COL, DOW_COL, BAKERY_ID_COL, BAKERY_NAME_COL],
        how="left",
    )
    applied["hour_share"] = (
        applied[QTY_COL] / applied["bakery_day_sales"].replace(0, np.nan)
    ).fillna(0.0)

    profile = (
        applied.groupby([BAKERY_ID_COL, BAKERY_NAME_COL, DOW_COL, HOUR_COL], as_index=False)
        .agg(
            n_days=("hour_share", "size"),
            mean_hour_share=("hour_share", "mean"),
            median_hour_share=("hour_share", "median"),
            std_hour_share=("hour_share", "std"),
            mean_hour_sales=(QTY_COL, "mean"),
        )
    )
    profile["std_hour_share"] = profile["std_hour_share"].fillna(0.0)

    totals = (
        profile.groupby([BAKERY_ID_COL, DOW_COL])["mean_hour_share"]
        .sum()
        .rename("profile_sum")
        .reset_index()
    )
    profile = profile.merge(totals, on=[BAKERY_ID_COL, DOW_COL], how="left")
    profile["mean_hour_share_norm"] = np.where(
        profile["profile_sum"] > 0,
        profile["mean_hour_share"] / profile["profile_sum"],
        0.0,
    )
    profile.drop(columns=["profile_sum"], inplace=True)
    return profile.sort_values([BAKERY_ID_COL, DOW_COL, HOUR_COL]).reset_index(drop=True), applied


def build_bakery_hour_profile(source_path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        RAW_DATETIME_COL,
        RAW_DATE_COL,
        RAW_EVENT_COL,
        RAW_QTY_COL,
        RAW_BAKERY_ID_COL,
        RAW_BAKERY_NAME_COL,
        "Дата время чека",
        "Дата продажи",
        "Вид события по кассе",
        "Кол-во",
        "Касса.Торговая точка",
    ]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(source_path, encoding="utf-8-sig", usecols=lambda c: c in usecols, chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        parts.append(aggregate_hourly_chunk(chunk))
        if i % 5 == 0:
            print(f"processed chunks: {i}", flush=True)

    hourly = merge_hourly_parts(parts)
    return build_hour_profile(hourly)


def build_summary(profile: pd.DataFrame, applied: pd.DataFrame) -> dict:
    return {
        "profile_rows": int(len(profile)),
        "applied_rows": int(len(applied)),
        "dates": int(applied[DATE_COL].nunique()) if len(applied) else 0,
        "bakeries": int(profile[BAKERY_ID_COL].nunique()) if len(profile) else 0,
        "mean_norm_share_sum": round(
            float(
                profile.groupby([BAKERY_ID_COL, DOW_COL])["mean_hour_share_norm"].sum().mean()
            ),
            6,
        )
        if len(profile)
        else 0.0,
    }


def save_outputs(output_dir: str | Path, profile: pd.DataFrame, applied: pd.DataFrame, summary: dict) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / OUTPUT_NAME
    applied_path = out_dir / APPLIED_DAILY_OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")
    applied.to_csv(applied_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"profile": profile_path, "applied": applied_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bakery-level hourly share profile from raw checks")
    parser.add_argument("--source-path", default=str(ROOT / "data" / "raw" / "sales_hrs_all.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    profile, applied = build_bakery_hour_profile(args.source_path, chunk_size=args.chunk_size)
    summary = build_summary(profile, applied)
    paths = save_outputs(args.output_dir, profile, applied, summary)

    print("=" * 72)
    print("BAKERY HOUR PROFILE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
