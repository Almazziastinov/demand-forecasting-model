"""
Build SKU hour-share profiles from raw check lines using chunked processing.

Input:
  data/raw/sales_hrs_all.csv (English or legacy Russian raw schema)

Output:
  data/processed/sku_hour_share_profile.csv
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
RAW_PRODUCT_ID_COL = "product_id"
RAW_PRODUCT_NAME_COL = "product_name"
RAW_CATEGORY_COL = "category_name"

DATE_COL = "date"
DOW_COL = "dow"
HOUR_COL = "hour"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "category_name"
SKU_HOUR_SALES_COL = "sku_hour_sales"
BAKERY_HOUR_SALES_COL = "bakery_hour_sales"
SKU_SHARE_COL = "sku_share_in_hour"

SALES_EVENT = "РџСЂРѕРґР°Р¶Р°"
SALES_EVENTS = {"Продажа", SALES_EVENT}
CHUNK_SIZE = 1_000_000

OUTPUT_NAME = "sku_hour_share_profile.csv"
APPLIED_DAILY_OUTPUT_NAME = "sku_hour_share_profile_daily.csv"
SUMMARY_OUTPUT_NAME = "sku_hour_share_profile_summary.json"


def aggregate_sku_hourly_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = normalize_snapshot_chunk(chunk)
    sales = chunk[chunk[RAW_EVENT_COL].isin(SALES_EVENTS)].copy()
    if sales.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
                SKU_HOUR_SALES_COL,
            ]
        )

    sales[DATE_COL] = parse_snapshot_date(sales[RAW_DATE_COL]).dt.normalize()
    sales["_dt"] = parse_snapshot_datetime(sales[RAW_DATETIME_COL])
    sales = sales.dropna(
        subset=[
            DATE_COL,
            "_dt",
            RAW_BAKERY_ID_COL,
            RAW_BAKERY_NAME_COL,
            RAW_PRODUCT_ID_COL,
            RAW_PRODUCT_NAME_COL,
        ]
    )
    if sales.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
                SKU_HOUR_SALES_COL,
            ]
        )

    sales[RAW_QTY_COL] = pd.to_numeric(sales[RAW_QTY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    sales[HOUR_COL] = sales["_dt"].dt.hour
    sales[DOW_COL] = sales[DATE_COL].dt.dayofweek
    sales[RAW_CATEGORY_COL] = sales[RAW_CATEGORY_COL].fillna("unknown")

    grouped = (
        sales.groupby(
            [
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                RAW_BAKERY_ID_COL,
                RAW_BAKERY_NAME_COL,
                RAW_PRODUCT_ID_COL,
                RAW_PRODUCT_NAME_COL,
                RAW_CATEGORY_COL,
            ],
            as_index=False,
        )
        .agg(sku_hour_sales=(RAW_QTY_COL, "sum"))
        .rename(
            columns={
                RAW_BAKERY_ID_COL: BAKERY_ID_COL,
                RAW_BAKERY_NAME_COL: BAKERY_NAME_COL,
                RAW_PRODUCT_ID_COL: PRODUCT_ID_COL,
                RAW_PRODUCT_NAME_COL: PRODUCT_NAME_COL,
                RAW_CATEGORY_COL: CATEGORY_COL,
            }
        )
    )
    return grouped


def merge_hourly_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    hourly = pd.concat(parts, ignore_index=True)
    hourly = (
        hourly.groupby(
            [
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
            ],
            as_index=False,
        )
        .agg(sku_hour_sales=(SKU_HOUR_SALES_COL, "sum"))
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL, HOUR_COL])
        .reset_index(drop=True)
    )
    return hourly


def build_sku_hour_share_profile(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bakery_hour = (
        hourly.groupby([DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_NAME_COL], as_index=False)
        .agg(bakery_hour_sales=(SKU_HOUR_SALES_COL, "sum"))
    )
    applied = hourly.merge(
        bakery_hour,
        on=[DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_NAME_COL],
        how="left",
    )
    applied[SKU_SHARE_COL] = (
        applied[SKU_HOUR_SALES_COL] / applied[BAKERY_HOUR_SALES_COL].replace(0, np.nan)
    ).fillna(0.0)

    profile = (
        applied.groupby(
            [
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
                DOW_COL,
                HOUR_COL,
            ],
            as_index=False,
        )
        .agg(
            n_days=(SKU_SHARE_COL, "size"),
            mean_sku_share_in_hour=(SKU_SHARE_COL, "mean"),
            median_sku_share_in_hour=(SKU_SHARE_COL, "median"),
            std_sku_share_in_hour=(SKU_SHARE_COL, "std"),
            mean_sku_hour_sales=(SKU_HOUR_SALES_COL, "mean"),
        )
    )
    profile["std_sku_share_in_hour"] = profile["std_sku_share_in_hour"].fillna(0.0)

    totals = (
        profile.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])["mean_sku_share_in_hour"]
        .sum()
        .rename("profile_sum")
        .reset_index()
    )
    profile = profile.merge(totals, on=[BAKERY_ID_COL, DOW_COL, HOUR_COL], how="left")
    profile["mean_sku_share_in_hour_norm"] = np.where(
        profile["profile_sum"] > 0,
        profile["mean_sku_share_in_hour"] / profile["profile_sum"],
        0.0,
    )
    profile.drop(columns=["profile_sum"], inplace=True)
    profile = profile.sort_values([BAKERY_ID_COL, DOW_COL, HOUR_COL, PRODUCT_ID_COL]).reset_index(drop=True)
    return profile, applied


def build_from_raw(source_path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        RAW_DATETIME_COL,
        RAW_DATE_COL,
        RAW_EVENT_COL,
        RAW_QTY_COL,
        RAW_BAKERY_ID_COL,
        RAW_BAKERY_NAME_COL,
        RAW_PRODUCT_ID_COL,
        RAW_PRODUCT_NAME_COL,
        RAW_CATEGORY_COL,
        "Дата время чека",
        "Дата продажи",
        "Вид события по кассе",
        "Кол-во",
        "Касса.Торговая точка",
        "Номенклатура",
        "Категория",
    ]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(source_path, encoding="utf-8-sig", usecols=lambda c: c in usecols, chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        parts.append(aggregate_sku_hourly_chunk(chunk))
        if i % 5 == 0:
            print(f"processed chunks: {i}", flush=True)

    hourly = merge_hourly_parts(parts)
    return build_sku_hour_share_profile(hourly)


def build_summary(profile: pd.DataFrame, applied: pd.DataFrame) -> dict:
    return {
        "profile_rows": int(len(profile)),
        "applied_rows": int(len(applied)),
        "dates": int(applied[DATE_COL].nunique()) if len(applied) else 0,
        "bakeries": int(profile[BAKERY_ID_COL].nunique()) if len(profile) else 0,
        "products": int(profile[PRODUCT_ID_COL].nunique()) if len(profile) else 0,
        "mean_norm_share_sum": round(
            float(
                profile.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])["mean_sku_share_in_hour_norm"].sum().mean()
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
    parser = argparse.ArgumentParser(description="Build SKU hour-share profile from raw checks")
    parser.add_argument("--source-path", default=str(ROOT / "data" / "raw" / "sales_hrs_all.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    profile, applied = build_from_raw(args.source_path, chunk_size=args.chunk_size)
    summary = build_summary(profile, applied)
    paths = save_outputs(args.output_dir, profile, applied, summary)

    print("=" * 72)
    print("SKU HOUR SHARE PROFILE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
