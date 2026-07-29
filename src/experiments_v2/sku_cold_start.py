"""Cold-start forecast floor for SKUs with sales but immature forecast history."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColdStartConfig:
    product_ids: tuple[int, ...] = (11573, 11574)
    alpha: float = 0.90
    min_sales_days: int = 3
    max_forecast_days: int = 13


def build_cold_start_registry(
    history: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    config: ColdStartConfig = ColdStartConfig(),
) -> pd.DataFrame:
    """Build an own-sales EWMA floor using information before ``as_of_date``."""
    required = {"date", "bakery_id", "product_id", "sold_qty", "forecast_qty"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    as_of = pd.Timestamp(as_of_date).normalize()
    work = history.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["sold_qty"] = pd.to_numeric(work["sold_qty"], errors="coerce").fillna(0.0)
    work["forecast_qty"] = pd.to_numeric(
        work["forecast_qty"],
        errors="coerce",
    ).fillna(0.0)
    work = work[
        work["date"].lt(as_of)
        & work["product_id"].isin(config.product_ids)
    ].sort_values(["bakery_id", "product_id", "date"])
    if work.empty:
        return pd.DataFrame()

    keys = ["bakery_id", "product_id"]
    work["sales_ewma"] = work.groupby(keys)["sold_qty"].transform(
        lambda values: values.ewm(
            alpha=config.alpha,
            adjust=False,
            min_periods=config.min_sales_days,
        ).mean()
    )
    summary = work.groupby(keys, as_index=False).agg(
        sales_days=("date", "nunique"),
        forecast_days=("forecast_qty", lambda values: int(values.gt(0).sum())),
        cold_start_floor=("sales_ewma", "last"),
    )
    eligible = (
        summary["sales_days"].ge(config.min_sales_days)
        & summary["forecast_days"].le(config.max_forecast_days)
        & summary["cold_start_floor"].notna()
    )
    return summary[eligible].reset_index(drop=True)


def apply_category_neutral_cold_start(
    forecast: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    output_col: str = "cold_start_forecast_qty",
) -> pd.DataFrame:
    """Apply cold-start floors while preserving bakery/category totals."""
    required = {
        "bakery_id",
        "product_id",
        "category_name",
        "forecast_qty",
    }
    missing = sorted(required.difference(forecast.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = forecast.copy()
    if registry.empty:
        work[output_col] = work["forecast_qty"]
        return work

    work = work.merge(
        registry[["bakery_id", "product_id", "cold_start_floor"]],
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    work["_candidate"] = work[["forecast_qty", "cold_start_floor"]].max(axis=1)
    group_cols = ["bakery_id", "category_name"]
    if "date" in work.columns:
        group_cols.insert(0, "date")
    work["_original_total"] = work.groupby(group_cols)["forecast_qty"].transform(
        "sum"
    )
    work["_candidate_total"] = work.groupby(group_cols)["_candidate"].transform(
        "sum"
    )
    scale = (
        work["_original_total"]
        / work["_candidate_total"].replace(0.0, pd.NA)
    ).fillna(1.0)
    work[output_col] = work["_candidate"] * scale
    return work.drop(
        columns=[
            "cold_start_floor",
            "_candidate",
            "_original_total",
            "_candidate_total",
        ]
    )
