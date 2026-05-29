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
  check_datetime
  product_id
  product_name
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

from src.experiments_v2.sales_cleaning import add_base_training_policy_flags
from src.experiments_v2.sales_cleaning import add_quantile_capped_base_target
from src.experiments_v2.sales_cleaning import add_robust_sales_outlier_flags
from src.experiments_v2.sales_cleaning import add_rolling_median_capped_base_target
from src.experiments_v2.sales_cleaning import add_rolling_quantile_capped_base_target
from src.experiments_v2.raw_sales_dedup import RAW_AMOUNT_COL
from src.experiments_v2.raw_sales_dedup import RAW_BAKERY_ID_COL
from src.experiments_v2.raw_sales_dedup import RAW_BAKERY_NAME_COL
from src.experiments_v2.raw_sales_dedup import RAW_CITY_COL
from src.experiments_v2.raw_sales_dedup import RAW_DATE_COL
from src.experiments_v2.raw_sales_dedup import RAW_PRICE_COL
from src.experiments_v2.raw_sales_dedup import RAW_QTY_COL
from src.experiments_v2.raw_sales_dedup import deduplicate_sales_chunk
from src.experiments_v2.raw_sales_dedup import prepare_sales_chunk


ROOT = Path(__file__).resolve().parents[2]

RAW_EVENT_COL = "cash_event_type"

DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
TARGET_COL = "bakery_sales"
OBSERVED_TARGET_COL = "bakery_sales_observed"
FILLED_TARGET_COL = "bakery_sales_filled"
SALES_MISSING_FLAG_COL = "sales_missing_flag"
SALES_OBSERVED_FLAG_COL = "sales_observed_flag"
SALES_IMPUTATION_SOURCE_COL = "sales_imputation_source"
SALES_IMPUTED_VALUE_COL = "sales_imputed_value"

SALES_EVENT = "\u041f\u0440\u043e\u0434\u0430\u0436\u0430"
CHUNK_SIZE = 1_000_000
SALES_EVENTS = {"\u041f\u0440\u043e\u0434\u0430\u0436\u0430", SALES_EVENT}

OUTPUT_NAME = "bakery_daily_sales.csv"
SUMMARY_OUTPUT_NAME = "bakery_daily_sales_summary.json"
RU_CHECK_DATE_COL = (
    "\u0414\u0430\u0442\u0430 \u043f\u0440\u043e\u0434\u0430\u0436\u0438"
)
RU_CHECK_DATETIME_COL = (
    "\u0414\u0430\u0442\u0430 \u0432\u0440\u0435\u043c\u044f \u0447\u0435\u043a\u0430"
)
RU_EVENT_COL = (
    "\u0412\u0438\u0434 \u0441\u043e\u0431\u044b\u0442\u0438\u044f "
    "\u043f\u043e \u043a\u0430\u0441\u0441\u0435"
)
RU_BAKERY_NAME_COL = (
    "\u041a\u0430\u0441\u0441\u0430.\u0422\u043e\u0440\u0433\u043e\u0432\u0430\u044f "
    "\u0442\u043e\u0447\u043a\u0430"
)
RU_PRICE_COL = "\u0426\u0435\u043d\u0430"
RU_QUANTITY_COL = "\u041a\u043e\u043b-\u0432\u043e"
RU_PRODUCT_NAME_COL = (
    "\u041d\u043e\u043c\u0435\u043d\u043a\u043b\u0430\u0442\u0443\u0440\u0430"
)


def _empty_chunk_aggregate() -> pd.DataFrame:
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


def _empty_dedup_stats() -> dict[str, float | int]:
    return {
        "raw_rows": 0,
        "deduped_rows": 0,
        "removed_rows": 0,
        "duplicate_groups": 0,
        "raw_quantity_sum": 0.0,
        "deduped_quantity_sum": 0.0,
        "removed_quantity_sum": 0.0,
        "raw_line_amount_sum": 0.0,
        "deduped_line_amount_sum": 0.0,
        "removed_line_amount_sum": 0.0,
    }


def aggregate_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    sales = prepare_sales_chunk(chunk, sales_events=SALES_EVENTS)
    if sales.empty:
        return _empty_chunk_aggregate(), _empty_dedup_stats()

    sales, dedup_stats = deduplicate_sales_chunk(sales)
    sales[DATE_COL] = sales[RAW_DATE_COL]
    sales[RAW_AMOUNT_COL] = sales[RAW_AMOUNT_COL].fillna(0.0)
    sales["price_x_qty_sum"] = sales[RAW_PRICE_COL].fillna(0.0) * sales[RAW_QTY_COL]
    sales["priced_quantity"] = np.where(
        sales[RAW_PRICE_COL].notna(),
        sales[RAW_QTY_COL],
        0.0,
    )

    grouped = (
        sales.groupby(
            [DATE_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL, RAW_CITY_COL],
            as_index=False,
        )
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
    return grouped, dedup_stats


def merge_partial_results(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    daily = pd.concat(parts, ignore_index=True)
    daily = (
        daily.groupby(
            [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL],
            as_index=False,
        )
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


def add_complete_bakery_calendar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    work = work.dropna(subset=[DATE_COL, BAKERY_ID_COL]).copy()

    calendar_parts = []
    for bakery_id, group in work.groupby(BAKERY_ID_COL, sort=False):
        start_date = group[DATE_COL].min()
        end_date = group[DATE_COL].max()
        calendar_parts.append(
            pd.DataFrame(
                {
                    BAKERY_ID_COL: bakery_id,
                    DATE_COL: pd.date_range(start_date, end_date, freq="D"),
                }
            )
        )
    calendar = pd.concat(calendar_parts, ignore_index=True)

    attrs = work.attrs.copy()
    panel = calendar.merge(work, on=[BAKERY_ID_COL, DATE_COL], how="left")
    panel = panel.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)
    group = panel.groupby(BAKERY_ID_COL, sort=False)
    for col in [BAKERY_NAME_COL, CITY_COL]:
        panel[col] = group[col].transform(lambda s: s.ffill().bfill())

    panel[OBSERVED_TARGET_COL] = pd.to_numeric(
        panel[TARGET_COL],
        errors="coerce",
    )
    panel[SALES_MISSING_FLAG_COL] = panel[OBSERVED_TARGET_COL].isna().astype(int)
    panel[SALES_OBSERVED_FLAG_COL] = (panel[SALES_MISSING_FLAG_COL] == 0).astype(int)

    for col in ["line_amount_sum", "priced_quantity", "price_x_qty_sum"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)

    panel.attrs.update(attrs)
    return panel


def add_missing_sales_imputation(
    df: pd.DataFrame,
    *,
    window_same_dow: int = 8,
    min_same_dow_periods: int = 4,
    window_recent: int = 30,
    min_recent_periods: int = 4,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    observed = pd.to_numeric(work[OBSERVED_TARGET_COL], errors="coerce")
    work["_observed_sales_for_imputation"] = observed

    same_dow_group = work.groupby([BAKERY_ID_COL, "dow"], sort=False)[
        "_observed_sales_for_imputation"
    ]
    work["_impute_same_dow"] = same_dow_group.transform(
        lambda s: s.shift(1)
        .rolling(window=window_same_dow, min_periods=min_same_dow_periods)
        .median()
    )
    same_dow_expanding = same_dow_group.transform(
        lambda s: s.shift(1).expanding(min_periods=min_same_dow_periods).median()
    )
    work["_impute_same_dow"] = work["_impute_same_dow"].fillna(same_dow_expanding)

    bakery_group = work.groupby(BAKERY_ID_COL, sort=False)[
        "_observed_sales_for_imputation"
    ]
    work["_impute_recent"] = bakery_group.transform(
        lambda s: s.shift(1)
        .rolling(window=window_recent, min_periods=min_recent_periods)
        .median()
    )
    recent_expanding = bakery_group.transform(
        lambda s: s.shift(1).expanding(min_periods=min_recent_periods).median()
    )
    work["_impute_recent"] = work["_impute_recent"].fillna(recent_expanding)

    missing_mask = pd.to_numeric(
        work[SALES_MISSING_FLAG_COL],
        errors="coerce",
    ).fillna(0).astype(int) == 1
    imputed = work["_impute_same_dow"].where(
        work["_impute_same_dow"].notna(),
        work["_impute_recent"],
    )
    imputed = imputed.fillna(0.0).clip(lower=0.0)

    work[SALES_IMPUTED_VALUE_COL] = np.where(missing_mask, imputed, np.nan)
    work[SALES_IMPUTATION_SOURCE_COL] = "observed"
    work.loc[
        missing_mask & work["_impute_same_dow"].notna(),
        SALES_IMPUTATION_SOURCE_COL,
    ] = "rolling_same_dow_median"
    work.loc[
        missing_mask & work["_impute_same_dow"].isna() & work["_impute_recent"].notna(),
        SALES_IMPUTATION_SOURCE_COL,
    ] = "rolling_recent_bakery_median"
    work.loc[
        missing_mask & work["_impute_same_dow"].isna() & work["_impute_recent"].isna(),
        SALES_IMPUTATION_SOURCE_COL,
    ] = "zero_insufficient_history"

    work[FILLED_TARGET_COL] = observed.where(~missing_mask, imputed).fillna(0.0)
    work[TARGET_COL] = work[FILLED_TARGET_COL]
    work = work.drop(
        columns=[
            "_observed_sales_for_imputation",
            "_impute_same_dow",
            "_impute_recent",
        ]
    )
    return work.reset_index(drop=True)


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


def add_cleaning_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = add_robust_sales_outlier_flags(
        work,
        value_col=TARGET_COL,
        entity_cols=[BAKERY_ID_COL],
        seasonal_cols=["dow"],
    )
    work = add_base_training_policy_flags(work)
    work = add_quantile_capped_base_target(
        work,
        value_col=TARGET_COL,
        entity_cols=[BAKERY_ID_COL],
        seasonal_cols=["dow"],
        lower_quantile=0.05,
        upper_quantile=0.95,
        min_seasonal_rows=8,
        capped_col="bakery_sales_base_quantile_capped",
    )
    work = add_rolling_median_capped_base_target(
        work,
        value_col=TARGET_COL,
        entity_cols=[BAKERY_ID_COL],
        seasonal_cols=["dow"],
        window=8,
        min_periods=4,
        upper_multiplier=1.6,
        lower_multiplier=0.5,
        capped_col="bakery_sales_base_rolling_capped",
    )
    work = add_rolling_quantile_capped_base_target(
        work,
        value_col=TARGET_COL,
        entity_cols=[BAKERY_ID_COL],
        seasonal_cols=["dow"],
        window=26,
        min_periods=8,
        lower_quantile=0.05,
        upper_quantile=0.95,
        capped_col="bakery_sales_base_rolling_quantile_capped",
    )
    work.attrs.update(df.attrs)
    return work


def build_bakery_daily_dataset(
    source_path: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    usecols = [
        RAW_DATE_COL,
        "check_datetime",
        RAW_EVENT_COL,
        RAW_QTY_COL,
        RAW_BAKERY_ID_COL,
        RAW_BAKERY_NAME_COL,
        RAW_CITY_COL,
        RAW_PRICE_COL,
        RAW_AMOUNT_COL,
        "product_id",
        "product_name",
        RU_CHECK_DATE_COL,
        RU_CHECK_DATETIME_COL,
        RU_EVENT_COL,
        RU_BAKERY_NAME_COL,
        RU_PRICE_COL,
        RU_QUANTITY_COL,
        RU_PRODUCT_NAME_COL,
    ]
    parts: list[pd.DataFrame] = []
    dedup_totals = _empty_dedup_stats()

    reader = pd.read_csv(
        source_path,
        encoding="utf-8-sig",
        usecols=lambda c: c in usecols,
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader, start=1):
        part, dedup_stats = aggregate_chunk(chunk)
        parts.append(part)
        for key, value in dedup_stats.items():
            dedup_totals[key] += value
        if i % 5 == 0:
            print(f"processed chunks: {i}", flush=True)

    daily = merge_partial_results(parts)
    daily = add_complete_bakery_calendar(daily)
    daily = add_price_features(daily)
    daily = add_calendar_features(daily)
    daily = add_missing_sales_imputation(daily)
    daily = add_cleaning_features(daily)
    daily = add_lag_features(daily)
    daily.attrs["dedup_summary"] = dedup_totals
    return daily


def build_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "date_min": None if df.empty else str(df[DATE_COL].min().date()),
        "date_max": None if df.empty else str(df[DATE_COL].max().date()),
        "dates": int(df[DATE_COL].nunique()) if len(df) else 0,
        "bakeries": int(df[BAKERY_ID_COL].nunique()) if len(df) else 0,
        "cities": int(df[CITY_COL].nunique()) if len(df) else 0,
        "mean_bakery_sales": round(float(df[TARGET_COL].mean()), 6) if len(df) else 0.0,
    }
    if len(df) and SALES_MISSING_FLAG_COL in df.columns:
        summary["missing_sales_rows"] = int(df[SALES_MISSING_FLAG_COL].sum())
        summary["missing_sales_share"] = round(
            float(df[SALES_MISSING_FLAG_COL].mean()),
            6,
        )
    if len(df) and "bakery_sales_base_quantile_capped" in df.columns:
        summary["mean_bakery_sales_base_quantile_capped"] = round(
            float(df["bakery_sales_base_quantile_capped"].mean()),
            6,
        )
        summary["mean_bakery_sales_base_rolling_capped"] = round(
            float(df["bakery_sales_base_rolling_capped"].mean()),
            6,
        )
        summary["quantile_capped_rows"] = int(
            df["quantile_base_target_capped_flag"].sum()
        )
        summary["rolling_capped_rows"] = int(
            df["rolling_base_target_capped_flag"].sum()
        )
        if "rolling_quantile_base_target_capped_flag" in df.columns:
            summary["mean_bakery_sales_base_rolling_quantile_capped"] = round(
                float(df["bakery_sales_base_rolling_quantile_capped"].mean()),
                6,
            )
            summary["rolling_quantile_capped_rows"] = int(
                df["rolling_quantile_base_target_capped_flag"].sum()
            )
        summary["unexplained_high_outlier_rows"] = int(
            df["unexplained_high_outlier_flag"].sum()
        )
        summary["contextual_high_outlier_rows"] = int(
            df["contextual_high_outlier_flag"].sum()
        )
    dedup_summary = df.attrs.get("dedup_summary")
    if dedup_summary:
        summary["raw_sales_dedup"] = {
            key: (round(float(value), 6) if isinstance(value, float) else int(value))
            for key, value in dedup_summary.items()
        }
    return summary


def save_outputs(
    output_dir: str | Path,
    df: pd.DataFrame,
    summary: dict,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"dataset": csv_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bakery-level daily dataset from raw checks"
    )
    parser.add_argument(
        "--source-path",
        default=str(ROOT / "data" / "raw" / "sales_hrs_all.csv"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    df = build_bakery_daily_dataset(
        args.source_path,
        chunk_size=args.chunk_size,
    )
    summary = build_summary(df)
    paths = save_outputs(args.output_dir, df, summary)

    print("=" * 72)
    print("BAKERY DAILY DATASET")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
