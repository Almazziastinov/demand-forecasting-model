"""
Build the sales-first factual backbone and stable aggregate layers.

This module treats observed sales as the only canonical target-like fact.
Legacy demand-derived columns are explicitly excluded from the canonical
backbone so later attainable-demand logic can be rebuilt on top of raw sales.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"
CATEGORY_COL = "Категория"
CITY_COL = "Город"
TARGET_COL = "Продано"

BACKBONE_REQUIRED_COLS = [
    DATE_COL,
    BAKERY_COL,
    CATEGORY_COL,
    PRODUCT_COL,
    TARGET_COL,
    CITY_COL,
]

CALENDAR_COLS = [
    "ДеньНедели",
    "День",
    "IsWeekend",
    "Месяц",
    "НомерНедели",
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_payday_week",
    "is_month_start",
    "is_month_end",
]

WEATHER_COLS = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "precipitation",
    "rain",
    "snowfall",
    "windspeed_max",
    "temp_range",
    "is_rainy",
    "is_snowy",
    "is_cold",
    "is_warm",
    "is_windy",
    "is_bad_weather",
    "weather_cat_code",
]

PRICE_COLS = [
    "avg_price",
    "price_vs_median",
    "price_lag7",
    "price_change_7d",
    "price_roll_mean7",
    "price_roll_std7",
]

SALES_HISTORY_COLS = [
    "sales_lag1",
    "sales_lag2",
    "sales_lag3",
    "sales_lag7",
    "sales_lag14",
    "sales_lag30",
    "sales_roll_mean3",
    "sales_roll_mean7",
    "sales_roll_mean14",
    "sales_roll_mean30",
    "sales_roll_std7",
]

LEGACY_DERIVED_COLS = [
    "Спрос",
    "demand_estimated",
    "lost_qty",
    "is_censored",
    "demand_lag1",
    "demand_lag2",
    "demand_lag3",
    "demand_lag7",
    "demand_lag14",
    "demand_lag30",
    "demand_roll_mean3",
    "demand_roll_mean7",
    "demand_roll_mean14",
    "demand_roll_mean30",
    "demand_roll_std7",
]

BACKBONE_OUTPUT_NAME = "daily_sales_backbone.csv"
BAKERY_OUTPUT_NAME = "bakery_daily_sales.csv"
CATEGORY_OUTPUT_NAME = "bakery_category_daily_sales.csv"
SUMMARY_OUTPUT_NAME = "sales_first_backbone_summary.json"


def _available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def load_daily_sales(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if DATE_COL not in df.columns:
        raise KeyError(f"Missing required date column: {DATE_COL}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def validate_daily_sales(df: pd.DataFrame) -> dict:
    missing = [col for col in BACKBONE_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required backbone columns: {missing}")

    work = df.copy()
    duplicates = int(work.duplicated([DATE_COL, BAKERY_COL, PRODUCT_COL]).sum())
    null_rows = int(
        work[[DATE_COL, BAKERY_COL, PRODUCT_COL, CATEGORY_COL, TARGET_COL]].isna().any(axis=1).sum()
    )
    negative_sales = int((pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0) < 0).sum())

    return {
        "duplicates_by_key": duplicates,
        "rows_with_null_required": null_rows,
        "negative_sales_rows": negative_sales,
        "rows_total": int(len(work)),
        "date_min": None if work[DATE_COL].isna().all() else str(work[DATE_COL].min().date()),
        "date_max": None if work[DATE_COL].isna().all() else str(work[DATE_COL].max().date()),
    }


def build_sales_backbone(df: pd.DataFrame) -> pd.DataFrame:
    validation = validate_daily_sales(df)
    if validation["duplicates_by_key"] > 0:
        raise ValueError(
            f"Found {validation['duplicates_by_key']} duplicate rows for key "
            f"{DATE_COL} x {BAKERY_COL} x {PRODUCT_COL}"
        )

    keep_cols = []
    keep_cols.extend(BACKBONE_REQUIRED_COLS)
    keep_cols.extend(_available(CALENDAR_COLS, df))
    keep_cols.extend(_available(WEATHER_COLS, df))
    keep_cols.extend(_available(PRICE_COLS, df))
    keep_cols.extend(_available(SALES_HISTORY_COLS, df))

    # Preserve optional factual columns that are not demand-derived.
    for col in df.columns:
        if col in keep_cols or col in LEGACY_DERIVED_COLS:
            continue
        keep_cols.append(col)

    backbone = df.loc[:, keep_cols].copy()
    backbone = backbone.drop(columns=_available(LEGACY_DERIVED_COLS, backbone), errors="ignore")
    backbone = backbone.dropna(subset=[DATE_COL, BAKERY_COL, PRODUCT_COL, CATEGORY_COL, TARGET_COL]).copy()
    backbone[TARGET_COL] = pd.to_numeric(backbone[TARGET_COL], errors="coerce").fillna(0).clip(lower=0)
    backbone = backbone.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)
    return backbone


def _add_group_history_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    prefix: str,
) -> pd.DataFrame:
    work = df.sort_values(group_cols + [DATE_COL]).copy()
    grouped = work.groupby(group_cols)[target_col]

    for lag in [1, 7, 14]:
        work[f"{prefix}_lag{lag}"] = grouped.shift(lag)

    for window, min_periods in [(7, 3), (30, 7)]:
        work[f"{prefix}_roll_mean{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )

    work[f"{prefix}_roll_std7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )
    return work


def build_bakery_daily_sales(backbone: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        TARGET_COL: "sum",
        CITY_COL: "first",
        CATEGORY_COL: pd.Series.nunique,
        PRODUCT_COL: pd.Series.nunique,
    }
    for col in _available(CALENDAR_COLS, backbone):
        agg_map[col] = "first"
    for col in _available(WEATHER_COLS, backbone):
        agg_map[col] = "first"

    bakery = (
        backbone.groupby([DATE_COL, BAKERY_COL], as_index=False)
        .agg(agg_map)
        .rename(
            columns={
                TARGET_COL: "bakery_sales_total",
                CATEGORY_COL: "categories_in_bakery_today",
                PRODUCT_COL: "items_in_bakery_today",
            }
        )
    )

    bakery = _add_group_history_features(
        bakery,
        group_cols=[BAKERY_COL],
        target_col="bakery_sales_total",
        prefix="bakery_sales",
    )
    return bakery.sort_values([BAKERY_COL, DATE_COL]).reset_index(drop=True)


def build_bakery_category_daily_sales(backbone: pd.DataFrame) -> pd.DataFrame:
    bakery_daily = build_bakery_daily_sales(backbone)[[DATE_COL, BAKERY_COL, "bakery_sales_total"]]

    agg_map = {
        TARGET_COL: "sum",
        CITY_COL: "first",
        PRODUCT_COL: pd.Series.nunique,
    }
    for col in _available(CALENDAR_COLS, backbone):
        agg_map[col] = "first"
    for col in _available(WEATHER_COLS, backbone):
        agg_map[col] = "first"

    category = (
        backbone.groupby([DATE_COL, BAKERY_COL, CATEGORY_COL], as_index=False)
        .agg(agg_map)
        .rename(
            columns={
                TARGET_COL: "category_sales_total",
                PRODUCT_COL: "items_in_category_today",
            }
        )
    )

    category = category.merge(bakery_daily, on=[DATE_COL, BAKERY_COL], how="left")
    category["category_share_in_bakery"] = (
        category["category_sales_total"] / category["bakery_sales_total"].replace(0, pd.NA)
    ).fillna(0.0)

    category = _add_group_history_features(
        category,
        group_cols=[BAKERY_COL, CATEGORY_COL],
        target_col="category_sales_total",
        prefix="category_sales",
    )
    return category.sort_values([BAKERY_COL, CATEGORY_COL, DATE_COL]).reset_index(drop=True)


def build_summary(
    source_df: pd.DataFrame,
    backbone: pd.DataFrame,
    bakery_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
) -> dict:
    source_validation = validate_daily_sales(source_df)
    return {
        "source_rows": int(len(source_df)),
        "source_validation": source_validation,
        "backbone_rows": int(len(backbone)),
        "backbone_columns": backbone.columns.tolist(),
        "backbone_bakeries": int(backbone[BAKERY_COL].nunique()),
        "backbone_products": int(backbone[PRODUCT_COL].nunique()),
        "backbone_categories": int(backbone[CATEGORY_COL].nunique()),
        "backbone_dates": int(backbone[DATE_COL].nunique()),
        "bakery_daily_rows": int(len(bakery_daily)),
        "bakery_daily_dates": int(bakery_daily[DATE_COL].nunique()),
        "bakery_daily_bakeries": int(bakery_daily[BAKERY_COL].nunique()),
        "bakery_category_daily_rows": int(len(bakery_category_daily)),
        "bakery_category_daily_dates": int(bakery_category_daily[DATE_COL].nunique()),
        "legacy_columns_excluded": _available(LEGACY_DERIVED_COLS, source_df),
    }


def save_outputs(
    output_dir: str | Path,
    backbone: pd.DataFrame,
    bakery_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
    summary: dict,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_path = out_dir / BACKBONE_OUTPUT_NAME
    bakery_path = out_dir / BAKERY_OUTPUT_NAME
    category_path = out_dir / CATEGORY_OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME

    backbone.to_csv(backbone_path, index=False, encoding="utf-8-sig")
    bakery_daily.to_csv(bakery_path, index=False, encoding="utf-8-sig")
    bakery_category_daily.to_csv(category_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "backbone": backbone_path,
        "bakery_daily": bakery_path,
        "bakery_category_daily": category_path,
        "summary": summary_path,
    }


def build_and_save_sales_first_layers(
    source_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    source_df = load_daily_sales(source_path)
    backbone = build_sales_backbone(source_df)
    bakery_daily = build_bakery_daily_sales(backbone)
    bakery_category_daily = build_bakery_category_daily_sales(backbone)
    summary = build_summary(source_df, backbone, bakery_daily, bakery_category_daily)
    return save_outputs(output_dir, backbone, bakery_daily, bakery_category_daily, summary)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source_path = root / "data" / "processed" / "daily_sales_8m.csv"
    output_dir = root / "data" / "processed"
    paths = build_and_save_sales_first_layers(source_path, output_dir)

    print("=" * 72)
    print("SALES-FIRST BACKBONE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
