"""
Experimental positive-only hourly profiles.

Idea:
- build bakery x SKU x dow x hour profiles
- use only hourly slots where the SKU had at least one sale
- do not use stockout or good-execution filters

This is intentionally a simple exploratory baseline for visual comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
CATEGORY_COL = "Категория"
PRODUCT_COL = "Номенклатура"
QTY_COL = "qty"
HOUR_COL = "hour"
DOW_COL = "ДеньНедели"

RAW_DATE_COL = "Дата продажи"
RAW_DATETIME_COL = "Дата время чека"
RAW_EVENT_COL = "Вид события по кассе"
RAW_BAKERY_COL = "Касса.Торговая точка"
RAW_CATEGORY_COL = "Категория"
RAW_PRODUCT_COL = "Номенклатура"
RAW_QTY_COL = "Кол-во"
SALES_EVENT = "Продажа"

USE_COLS = [
    RAW_DATE_COL,
    RAW_DATETIME_COL,
    RAW_EVENT_COL,
    RAW_BAKERY_COL,
    RAW_CATEGORY_COL,
    RAW_PRODUCT_COL,
    RAW_QTY_COL,
]

PROFILE_OUTPUT_NAME = "hourly_positive_profiles.csv"
APPLIED_OUTPUT_NAME = "hourly_positive_profile_applied.csv"
DAILY_OUTPUT_NAME = "hourly_positive_profile_daily.csv"
SUMMARY_OUTPUT_NAME = "hourly_positive_profile_summary.json"


def load_sales(
    path: str | Path,
    *,
    bakery: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    print(f"[1/5] Loading raw hourly sales: {path}", flush=True)
    df = pd.read_csv(path, usecols=USE_COLS, encoding="utf-8-sig")
    print(f"      Raw rows loaded: {len(df):,}", flush=True)

    sales = df[df[RAW_EVENT_COL] == SALES_EVENT].copy()
    print(f"      Sales-event rows: {len(sales):,}", flush=True)
    sales[DATE_COL] = pd.to_datetime(sales[RAW_DATE_COL], dayfirst=True, errors="coerce")
    sales["_dt"] = pd.to_datetime(sales[RAW_DATETIME_COL], dayfirst=True, errors="coerce")
    sales[HOUR_COL] = sales["_dt"].dt.hour
    sales = sales.rename(
        columns={
            RAW_BAKERY_COL: BAKERY_COL,
            RAW_CATEGORY_COL: CATEGORY_COL,
            RAW_PRODUCT_COL: PRODUCT_COL,
            RAW_QTY_COL: QTY_COL,
        }
    )
    sales = sales.dropna(subset=[DATE_COL, BAKERY_COL, PRODUCT_COL, HOUR_COL]).copy()
    sales[QTY_COL] = pd.to_numeric(sales[QTY_COL], errors="coerce").fillna(0.0)
    sales[DOW_COL] = sales[DATE_COL].dt.dayofweek

    if bakery:
        sales = sales[sales[BAKERY_COL] == bakery].copy()
        print(f"      After bakery filter: {len(sales):,}", flush=True)
    if date_from:
        sales = sales[sales[DATE_COL] >= pd.Timestamp(date_from)].copy()
        print(f"      After date_from filter: {len(sales):,}", flush=True)
    if date_to:
        sales = sales[sales[DATE_COL] <= pd.Timestamp(date_to)].copy()
        print(f"      After date_to filter: {len(sales):,}", flush=True)

    print(
        f"      Final sales rows: {len(sales):,} | "
        f"dates: {sales[DATE_COL].nunique():,} | "
        f"products: {sales[PRODUCT_COL].nunique():,}",
        flush=True,
    )
    return sales[[DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL, QTY_COL]]


def aggregate_hourly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Aggregating hourly sales", flush=True)
    hourly_sku = (
        sales.groupby([DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL], as_index=False)[QTY_COL]
        .sum()
        .rename(columns={QTY_COL: "sku_qty"})
    )
    bakery_hourly = (
        sales.groupby([DATE_COL, DOW_COL, BAKERY_COL, HOUR_COL], as_index=False)[QTY_COL]
        .sum()
        .rename(columns={QTY_COL: "bakery_qty"})
    )
    print(f"      Hourly SKU rows: {len(hourly_sku):,}", flush=True)
    print(f"      Hourly bakery rows: {len(bakery_hourly):,}", flush=True)
    return hourly_sku.merge(bakery_hourly, on=[DATE_COL, DOW_COL, BAKERY_COL, HOUR_COL], how="left")


def build_positive_profiles(hourly: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Building positive-only hourly profiles", flush=True)
    positive = hourly[hourly["sku_qty"] > 0].copy()
    positive["share_of_bakery"] = (
        positive["sku_qty"] / positive["bakery_qty"].replace(0, np.nan)
    ).fillna(0.0)

    profile = (
        positive.groupby([BAKERY_COL, CATEGORY_COL, PRODUCT_COL, DOW_COL, HOUR_COL], as_index=False)
        .agg(
            n_positive_slots=("sku_qty", "size"),
            mean_sku_qty_positive=("sku_qty", "mean"),
            median_sku_qty_positive=("sku_qty", "median"),
            mean_share_positive=("share_of_bakery", "mean"),
            median_share_positive=("share_of_bakery", "median"),
            share_std_positive=("share_of_bakery", "std"),
        )
    )
    profile["share_std_positive"] = profile["share_std_positive"].fillna(0.0)
    profile["cv_share_positive"] = (
        profile["share_std_positive"] / profile["mean_share_positive"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"      Profile rows: {len(profile):,}", flush=True)
    return profile


def build_hourly_application_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    bakery_hours = hourly[[DATE_COL, DOW_COL, BAKERY_COL, HOUR_COL, "bakery_qty"]].drop_duplicates().copy()
    sku_keys = hourly[[BAKERY_COL, CATEGORY_COL, PRODUCT_COL]].drop_duplicates().copy()
    return bakery_hours.merge(sku_keys, on=BAKERY_COL, how="inner")


def apply_profiles(hourly: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Applying hourly profiles to all bakery hours", flush=True)
    observed = hourly[[DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL, "sku_qty"]].copy()
    frame = build_hourly_application_frame(hourly)
    applied = frame.merge(
        profiles,
        on=[BAKERY_COL, CATEGORY_COL, PRODUCT_COL, DOW_COL, HOUR_COL],
        how="left",
    )
    applied = applied.merge(
        observed,
        on=[DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL],
        how="left",
    )
    applied["sku_qty"] = applied["sku_qty"].fillna(0.0)
    applied["expected_qty_from_share"] = (
        applied["bakery_qty"] * applied["mean_share_positive"].fillna(0.0)
    )
    applied["expected_qty_from_share"] = np.maximum(applied["expected_qty_from_share"], applied["sku_qty"])
    applied["hourly_gap"] = applied["expected_qty_from_share"] - applied["sku_qty"]
    print(f"      Applied rows: {len(applied):,}", flush=True)
    return applied


def build_daily_from_applied(applied: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] Rolling up to daily SKU view", flush=True)
    daily = (
        applied.groupby([DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL], as_index=False)
        .agg(
            observed_sales=("sku_qty", "sum"),
            bakery_sales_total=("bakery_qty", "sum"),
            expected_sales_from_hourly_profile=("expected_qty_from_share", "sum"),
            total_hourly_gap=("hourly_gap", "sum"),
            profiled_hours=("mean_share_positive", lambda s: int(s.notna().sum())),
            positive_hours_observed=("sku_qty", lambda s: int((s > 0).sum())),
        )
    )
    print(f"      Daily rows: {len(daily):,}", flush=True)
    return daily


def build_summary(sales: pd.DataFrame, profiles: pd.DataFrame, applied: pd.DataFrame, daily: pd.DataFrame) -> dict:
    return {
        "raw_sales_rows": int(len(sales)),
        "dates": int(sales[DATE_COL].nunique()) if len(sales) else 0,
        "products": int(sales[PRODUCT_COL].nunique()) if len(sales) else 0,
        "profile_rows": int(len(profiles)),
        "applied_rows": int(len(applied)),
        "daily_rows": int(len(daily)),
        "rows_with_hourly_profile": int(applied["mean_share_positive"].notna().sum()),
        "daily_rows_with_profiled_hours": int((daily["profiled_hours"] > 0).sum()),
        "mean_profiled_hours_per_day": round(float(daily["profiled_hours"].mean()), 6),
        "mean_total_hourly_gap": round(float(daily["total_hourly_gap"].mean()), 6),
        "median_total_hourly_gap": round(float(daily["total_hourly_gap"].median()), 6),
        "date_min": None if sales.empty else str(sales[DATE_COL].min().date()),
        "date_max": None if sales.empty else str(sales[DATE_COL].max().date()),
    }


def save_outputs(
    output_dir: str | Path,
    profiles: pd.DataFrame,
    applied: pd.DataFrame,
    daily: pd.DataFrame,
    summary: dict,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""

    profile_path = out_dir / PROFILE_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    applied_path = out_dir / APPLIED_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    daily_path = out_dir / DAILY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")

    profiles.to_csv(profile_path, index=False, encoding="utf-8-sig")
    applied.to_csv(applied_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "profiles": profile_path,
        "applied": applied_path,
        "daily": daily_path,
        "summary": summary_path,
    }


def build_and_save_hourly_positive_profile(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    bakery: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    output_suffix: str = "",
) -> dict[str, Path]:
    sales = load_sales(source_path, bakery=bakery, date_from=date_from, date_to=date_to)
    hourly = aggregate_hourly_sales(sales)
    profiles = build_positive_profiles(hourly)
    applied = apply_profiles(hourly, profiles)
    daily = build_daily_from_applied(applied)
    summary = build_summary(sales, profiles, applied, daily)
    return save_outputs(output_dir, profiles, applied, daily, summary, output_suffix=output_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build positive-only hourly profiles")
    parser.add_argument("--bakery", default=None, help="Exact bakery name filter")
    parser.add_argument("--date-from", default=None, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--output-suffix", default="", help="Suffix for output files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    paths = build_and_save_hourly_positive_profile(
        root / "data" / "raw" / "sales_hrs_all.csv",
        root / "data" / "processed",
        bakery=args.bakery,
        date_from=args.date_from,
        date_to=args.date_to,
        output_suffix=args.output_suffix,
    )

    print("=" * 72)
    print("HOURLY POSITIVE PROFILE")
    print("=" * 72)
    if args.bakery:
        print(f"bakery: {args.bakery}")
    if args.date_from or args.date_to:
        print(f"date range: {args.date_from or 'min'} .. {args.date_to or 'max'}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
