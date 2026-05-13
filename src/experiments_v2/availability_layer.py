"""
Build a sales-first availability layer from hourly sales checks.

The goal of this layer is not to estimate "true demand" directly. Instead it
produces operational signals that help separate normal weak demand from likely
availability / execution issues before SKU profiles are built.
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
TARGET_COL = "Продано"
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

HOURLY_OUTPUT_NAME = "availability_hourly_signals.csv"
DAILY_OUTPUT_NAME = "availability_daily_signals.csv"
SUMMARY_OUTPUT_NAME = "availability_layer_summary.json"

TRAFFIC_FACTOR = 0.6
HIST_POSITIVE_RATE_THRESHOLD = 0.25
EARLY_STOP_HOUR_GAP = 2.0
MIN_BAKERY_SALES_AFTER_LAST = 10.0
MIN_BAKERY_AFTER_SHARE = 0.10
MIN_SKU_SALES_FOR_EARLY_STOP = 3.0
MIN_SKU_SELLING_HOURS_FOR_EARLY_STOP = 2
MAX_STOCKOUT_LIKE_RATIO_FOR_GOOD_DAY = 0.15
MIN_AVAILABILITY_SCORE_FOR_GOOD_DAY = 0.70


def _safe_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def load_hourly_sales(
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
    sales["_check_dt"] = pd.to_datetime(sales[RAW_DATETIME_COL], dayfirst=True, errors="coerce")
    sales[HOUR_COL] = sales["_check_dt"].dt.hour
    sales = sales.rename(
        columns={
            RAW_BAKERY_COL: BAKERY_COL,
            RAW_CATEGORY_COL: CATEGORY_COL,
            RAW_PRODUCT_COL: PRODUCT_COL,
            RAW_QTY_COL: TARGET_COL,
        }
    )
    sales = sales.dropna(subset=[DATE_COL, BAKERY_COL, PRODUCT_COL, HOUR_COL]).copy()
    sales[TARGET_COL] = pd.to_numeric(sales[TARGET_COL], errors="coerce").fillna(0.0)
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
        f"bakeries: {sales[BAKERY_COL].nunique():,} | "
        f"products: {sales[PRODUCT_COL].nunique():,} | "
        f"dates: {sales[DATE_COL].nunique():,}",
        flush=True,
    )
    return sales[[DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL, TARGET_COL]]


def aggregate_hourly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Aggregating to hourly sales", flush=True)
    hourly = (
        sales.groupby([DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL], as_index=False)[TARGET_COL]
        .sum()
        .sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL, HOUR_COL])
        .reset_index(drop=True)
    )
    print(f"      Hourly aggregated rows: {len(hourly):,}", flush=True)
    return hourly


def build_hourly_frame(hourly_sales: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Building full hourly frame", flush=True)
    bakery_hourly = (
        hourly_sales.groupby([DATE_COL, DOW_COL, BAKERY_COL, HOUR_COL], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "bakery_qty"})
    )

    sku_day = hourly_sales[[DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL]].drop_duplicates()
    bakery_hours = bakery_hourly[[DATE_COL, DOW_COL, BAKERY_COL, HOUR_COL]].drop_duplicates()
    frame = sku_day.merge(bakery_hours, on=[DATE_COL, DOW_COL, BAKERY_COL], how="left")

    work = frame.merge(
        hourly_sales.rename(columns={TARGET_COL: "sku_qty"}),
        on=[DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, HOUR_COL],
        how="left",
    ).merge(
        bakery_hourly,
        on=[DATE_COL, DOW_COL, BAKERY_COL, HOUR_COL],
        how="left",
    )

    work["sku_qty"] = work["sku_qty"].fillna(0.0)
    work["bakery_qty"] = work["bakery_qty"].fillna(0.0)
    print(f"      Hourly frame rows: {len(work):,}", flush=True)
    return work.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL, HOUR_COL]).reset_index(drop=True)


def add_hourly_availability_signals(hourly_frame: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Computing hourly availability signals", flush=True)
    work = hourly_frame.copy()

    traffic_ref = (
        work.groupby([BAKERY_COL, DOW_COL, HOUR_COL])["bakery_qty"]
        .median()
        .reset_index(name="traffic_median")
    )
    work = work.merge(traffic_ref, on=[BAKERY_COL, DOW_COL, HOUR_COL], how="left")
    work["has_normal_traffic"] = work["bakery_qty"] >= (TRAFFIC_FACTOR * work["traffic_median"].fillna(0.0))

    hist_positive = (
        work.groupby([BAKERY_COL, PRODUCT_COL, DOW_COL, HOUR_COL])["sku_qty"]
        .apply(lambda s: float((s > 0).mean()))
        .reset_index(name="hist_positive_rate")
    )
    hist_positive = hist_positive.sort_values([BAKERY_COL, PRODUCT_COL, DOW_COL, HOUR_COL]).reset_index(drop=True)
    hist_positive["hist_positive_rate_prev"] = hist_positive.groupby([BAKERY_COL, PRODUCT_COL, DOW_COL])[
        "hist_positive_rate"
    ].shift(1)
    hist_positive["hist_positive_rate_next"] = hist_positive.groupby([BAKERY_COL, PRODUCT_COL, DOW_COL])[
        "hist_positive_rate"
    ].shift(-1)
    hist_positive["hist_positive_rate_adjacent"] = hist_positive[
        ["hist_positive_rate", "hist_positive_rate_prev", "hist_positive_rate_next"]
    ].max(axis=1, skipna=True)
    work = work.merge(hist_positive, on=[BAKERY_COL, PRODUCT_COL, DOW_COL, HOUR_COL], how="left")
    work["has_hist_demand"] = (
        work["hist_positive_rate_adjacent"].fillna(work["hist_positive_rate"]).fillna(0.0)
        >= HIST_POSITIVE_RATE_THRESHOLD
    )

    work = work.sort_values([DATE_COL, BAKERY_COL, PRODUCT_COL, HOUR_COL]).reset_index(drop=True)
    group_keys = [DATE_COL, BAKERY_COL, PRODUCT_COL]
    work["prev_sku_qty"] = work.groupby(group_keys)["sku_qty"].shift(1).fillna(0.0)
    work["next_sku_qty"] = work.groupby(group_keys)["sku_qty"].shift(-1).fillna(0.0)
    work["has_neighbor_sales"] = (work["prev_sku_qty"] > 0) | (work["next_sku_qty"] > 0)

    work["zero_under_traffic"] = (
        (work["sku_qty"] == 0)
        & _safe_bool(work["has_normal_traffic"])
        & _safe_bool(work["has_hist_demand"])
    )
    work["stockout_like_hour"] = work["zero_under_traffic"] & _safe_bool(work["has_neighbor_sales"])
    print(
        f"      zero_under_traffic rows: {int(work['zero_under_traffic'].sum()):,} | "
        f"stockout_like rows: {int(work['stockout_like_hour'].sum()):,}",
        flush=True,
    )
    return work


def build_daily_availability(hourly_signals: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] Rolling up daily availability signals", flush=True)
    work = hourly_signals.copy()

    sku_day = (
        work.groupby([DATE_COL, DOW_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL], as_index=False)
        .agg(
            sku_sales_total=("sku_qty", "sum"),
            bakery_sales_total=("bakery_qty", "sum"),
            active_hours_count=(HOUR_COL, "nunique"),
            normal_traffic_hours=("has_normal_traffic", "sum"),
            hist_demand_hours=("has_hist_demand", "sum"),
            zero_under_traffic_hours=("zero_under_traffic", "sum"),
            stockout_like_hours=("stockout_like_hour", "sum"),
        )
    )

    positive_hours = work[work["sku_qty"] > 0].copy()
    first_last = (
        positive_hours.groupby([DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL], as_index=False)
        .agg(
            first_sale_hour=(HOUR_COL, "min"),
            last_sale_hour=(HOUR_COL, "max"),
            selling_hours_count=(HOUR_COL, "nunique"),
        )
    )
    sku_day = sku_day.merge(
        first_last,
        on=[DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL],
        how="left",
    )

    typical_last = (
        first_last.groupby([BAKERY_COL, PRODUCT_COL, CATEGORY_COL])["last_sale_hour"]
        .median()
        .reset_index(name="typical_last_sale_hour")
    )
    sku_day = sku_day.merge(typical_last, on=[BAKERY_COL, PRODUCT_COL, CATEGORY_COL], how="left")

    bakery_after = (
        work.merge(
            sku_day[[DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL, "last_sale_hour"]],
            on=[DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL],
            how="left",
        )
    )
    bakery_after["bakery_after_last_sale_qty"] = np.where(
        bakery_after[HOUR_COL] > bakery_after["last_sale_hour"].fillna(99),
        bakery_after["bakery_qty"],
        0.0,
    )
    after_sum = (
        bakery_after.groupby([DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL], as_index=False)[
            "bakery_after_last_sale_qty"
        ]
        .sum()
    )
    sku_day = sku_day.merge(after_sum, on=[DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL], how="left")
    sku_day["bakery_after_last_sale_qty"] = sku_day["bakery_after_last_sale_qty"].fillna(0.0)

    sku_day["early_stop_gap_hours"] = sku_day["typical_last_sale_hour"] - sku_day["last_sale_hour"]
    sku_day["bakery_after_last_sale_share"] = (
        sku_day["bakery_after_last_sale_qty"] / sku_day["bakery_sales_total"].replace(0, np.nan)
    ).fillna(0.0)
    sku_day["early_stop_flag"] = (
        sku_day["early_stop_gap_hours"].fillna(0.0) >= EARLY_STOP_HOUR_GAP
    ) & (
        sku_day["bakery_after_last_sale_qty"] >= MIN_BAKERY_SALES_AFTER_LAST
    ) & (
        sku_day["bakery_after_last_sale_share"] >= MIN_BAKERY_AFTER_SHARE
    ) & (
        sku_day["sku_sales_total"] >= MIN_SKU_SALES_FOR_EARLY_STOP
    ) & (
        sku_day["selling_hours_count"].fillna(0) >= MIN_SKU_SELLING_HOURS_FOR_EARLY_STOP
    )

    sku_day["zero_under_traffic_ratio"] = (
        sku_day["zero_under_traffic_hours"] / sku_day["active_hours_count"].replace(0, np.nan)
    ).fillna(0.0)
    sku_day["stockout_like_ratio"] = (
        sku_day["stockout_like_hours"] / sku_day["active_hours_count"].replace(0, np.nan)
    ).fillna(0.0)

    availability_score = (
        1.0
        - 0.50 * sku_day["early_stop_flag"].astype(float)
        - 0.30 * np.clip(sku_day["stockout_like_ratio"], 0.0, 1.0)
        - 0.20 * np.clip(sku_day["zero_under_traffic_ratio"], 0.0, 1.0)
    )
    sku_day["availability_score"] = np.clip(availability_score, 0.0, 1.0)
    sku_day["good_execution_day"] = (
        (~sku_day["early_stop_flag"])
        & (sku_day["stockout_like_ratio"] <= MAX_STOCKOUT_LIKE_RATIO_FOR_GOOD_DAY)
        & (sku_day["availability_score"] >= MIN_AVAILABILITY_SCORE_FOR_GOOD_DAY)
    )

    print(
        f"      Daily rows: {len(sku_day):,} | "
        f"early_stop days: {int(sku_day['early_stop_flag'].sum()):,} | "
        f"good_execution days: {int(sku_day['good_execution_day'].sum()):,}",
        flush=True,
    )

    return sku_day.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)


def build_summary(hourly_signals: pd.DataFrame, daily_signals: pd.DataFrame) -> dict:
    return {
        "hourly_rows": int(len(hourly_signals)),
        "daily_rows": int(len(daily_signals)),
        "dates": int(daily_signals[DATE_COL].nunique()),
        "bakeries": int(daily_signals[BAKERY_COL].nunique()),
        "products": int(daily_signals[PRODUCT_COL].nunique()),
        "stockout_like_hours_total": int(daily_signals["stockout_like_hours"].sum()),
        "zero_under_traffic_hours_total": int(daily_signals["zero_under_traffic_hours"].sum()),
        "early_stop_days_total": int(daily_signals["early_stop_flag"].sum()),
        "good_execution_days_total": int(daily_signals["good_execution_day"].sum()),
        "good_execution_share": round(float(daily_signals["good_execution_day"].mean()), 4),
        "availability_score_mean": round(float(daily_signals["availability_score"].mean()), 4),
        "date_min": None if daily_signals.empty else str(daily_signals[DATE_COL].min().date()),
        "date_max": None if daily_signals.empty else str(daily_signals[DATE_COL].max().date()),
    }


def save_outputs(
    output_dir: str | Path,
    hourly_signals: pd.DataFrame,
    daily_signals: pd.DataFrame,
    summary: dict,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{output_suffix}" if output_suffix else ""
    hourly_path = out_dir / HOURLY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    daily_path = out_dir / DAILY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")

    hourly_signals.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    daily_signals.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "hourly": hourly_path,
        "daily": daily_path,
        "summary": summary_path,
    }


def build_and_save_availability_layer(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    bakery: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    output_suffix: str = "",
) -> dict[str, Path]:
    sales = load_hourly_sales(
        source_path,
        bakery=bakery,
        date_from=date_from,
        date_to=date_to,
    )
    hourly_sales = aggregate_hourly_sales(sales)
    hourly_frame = build_hourly_frame(hourly_sales)
    hourly_signals = add_hourly_availability_signals(hourly_frame)
    daily_signals = build_daily_availability(hourly_signals)
    summary = build_summary(hourly_signals, daily_signals)
    return save_outputs(
        output_dir,
        hourly_signals,
        daily_signals,
        summary,
        output_suffix=output_suffix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly availability signals from raw sales checks")
    parser.add_argument("--bakery", help="Exact bakery name filter", default=None)
    parser.add_argument("--date-from", help="Inclusive start date YYYY-MM-DD", default=None)
    parser.add_argument("--date-to", help="Inclusive end date YYYY-MM-DD", default=None)
    parser.add_argument("--output-suffix", help="Suffix for output filenames", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source_path = root / "data" / "raw" / "sales_hrs_all.csv"
    output_dir = root / "data" / "processed"
    paths = build_and_save_availability_layer(
        source_path,
        output_dir,
        bakery=args.bakery,
        date_from=args.date_from,
        date_to=args.date_to,
        output_suffix=args.output_suffix,
    )

    print("=" * 72)
    print("AVAILABILITY LAYER")
    print("=" * 72)
    if args.bakery:
        print(f"bakery: {args.bakery}")
    if args.date_from or args.date_to:
        print(f"date range: {args.date_from or 'min'} .. {args.date_to or 'max'}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
