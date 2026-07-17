"""Experimental reconstruction of censored hourly demand.

This module is deliberately disconnected from the production pipeline.  It
turns likely post-stockout zeroes into conservative demand estimates before
hourly profiles or daily training targets are built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["bakery_id", "product_id"]
DAY_KEYS = ["date", *KEYS]
PROFILE_KEYS = [*KEYS, "dow", "hour"]


def mark_stockout_days(
    hourly: pd.DataFrame,
    production: pd.DataFrame,
    *,
    stockout_ratio: float = 0.90,
) -> pd.DataFrame:
    """Attach production totals and the current daily stockout signal."""
    work = hourly.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["dow"] = work["date"].dt.dayofweek
    daily_sales = (
        work.groupby(DAY_KEYS, as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "daily_sold"})
    )
    daily = daily_sales.merge(production, on=DAY_KEYS, how="left")
    daily["is_production_observed"] = daily["produced"].notna()
    daily["sell_through"] = daily["daily_sold"] / daily["produced"].replace(0.0, np.nan)
    daily["is_stockout_day"] = daily["sell_through"] >= stockout_ratio
    return work.merge(daily, on=DAY_KEYS, how="left")


def build_uncensored_hour_reference(
    marked_train: pd.DataFrame,
    *,
    min_days: int = 3,
) -> pd.DataFrame:
    """Learn expected hourly demand from days not classified as stockouts.

    The primary estimate is weekday-aware. Sparse cells borrow the product's
    hour-of-day mean, then its positive hourly mean.  No holdout rows should be
    passed here.
    """
    clean = marked_train[
        marked_train["is_production_observed"].fillna(False)
        & ~marked_train["is_stockout_day"].fillna(False)
    ].copy()
    positive = clean[clean["sold"] > 0].copy()

    primary = clean.groupby(PROFILE_KEYS, as_index=False).agg(
        expected_primary=("sold", "mean"), reference_days=("date", "nunique")
    )
    fallback = (
        clean.groupby([*KEYS, "hour"], as_index=False)["sold"]
        .mean()
        .rename(columns={"sold": "expected_hour"})
    )
    positive_fallback = (
        positive.groupby(KEYS, as_index=False)["sold"]
        .mean()
        .rename(columns={"sold": "expected_positive"})
    )
    reference = primary.merge(fallback, on=[*KEYS, "hour"], how="left")
    reference = reference.merge(positive_fallback, on=KEYS, how="left")
    reference["expected_demand"] = np.where(
        reference["reference_days"] >= min_days,
        reference["expected_primary"],
        reference["expected_hour"],
    )
    reference["expected_demand"] = reference["expected_demand"].fillna(
        reference["expected_positive"]
    )
    return reference[PROFILE_KEYS + ["expected_demand", "reference_days"]]


def build_bakery_share_reference(
    marked_train: pd.DataFrame,
    *,
    min_days: int = 3,
) -> pd.DataFrame:
    """Learn the SKU's mean share of its bakery's hourly traffic."""
    work = marked_train.copy()
    if "bakery_hour_sales" not in work.columns:
        work["bakery_hour_sales"] = work.groupby(["date", "bakery_id", "hour"])[
            "sold"
        ].transform("sum")
    work["sku_share"] = work["sold"] / work["bakery_hour_sales"].replace(0.0, np.nan)

    primary = work.groupby(PROFILE_KEYS, as_index=False).agg(
        mean_share_primary=("sku_share", "mean"),
        reference_days=("date", "nunique"),
    )
    hour_fallback = (
        work.groupby([*KEYS, "hour"], as_index=False)["sku_share"]
        .mean()
        .rename(columns={"sku_share": "mean_share_hour"})
    )
    product_fallback = (
        work.groupby(KEYS, as_index=False)["sku_share"]
        .mean()
        .rename(columns={"sku_share": "mean_share_product"})
    )
    reference = primary.merge(hour_fallback, on=[*KEYS, "hour"], how="left")
    reference = reference.merge(product_fallback, on=KEYS, how="left")
    reference["mean_sku_share"] = np.where(
        reference["reference_days"] >= min_days,
        reference["mean_share_primary"],
        reference["mean_share_hour"],
    )
    reference["mean_sku_share"] = reference["mean_sku_share"].fillna(
        reference["mean_share_product"]
    )
    return reference[PROFILE_KEYS + ["mean_sku_share", "reference_days"]]


def reconstruct_stockout_demand(
    marked: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    min_bakery_active: float = 3.0,
    max_fill_ratio: float = 2.0,
) -> pd.DataFrame:
    """Fill zeroes after the last sale on likely stockout days.

    Only post-last-sale zeroes are changed. The bakery must still be active,
    and every fill is capped relative to that SKU-day's observed positive-hour
    rate. Original sales remain available in ``sold_observed``.
    """
    work = marked.copy()
    work["dow"] = pd.to_datetime(work["date"]).dt.dayofweek
    work = work.merge(reference, on=PROFILE_KEYS, how="left")
    if "bakery_hour_sales" not in work.columns:
        work["bakery_hour_sales"] = work.groupby(["date", "bakery_id", "hour"])[
            "sold"
        ].transform("sum")

    positive_hour = work["hour"].where(work["sold"] > 0)
    work["last_sale_hour"] = positive_hour.groupby(
        [work["date"], work["bakery_id"], work["product_id"]]
    ).transform("max")
    positive_rate = work["sold"].where(work["sold"] > 0)
    work["observed_positive_rate"] = positive_rate.groupby(
        [work["date"], work["bakery_id"], work["product_id"]]
    ).transform("mean")

    work["is_censored_hour"] = (
        work["is_stockout_day"].fillna(False)
        & (work["sold"] <= 0)
        & (work["hour"] > work["last_sale_hour"])
        & (work["bakery_hour_sales"] >= min_bakery_active)
        & work["expected_demand"].notna()
        & (work["expected_demand"] > 0)
    )
    cap = work["observed_positive_rate"] * max_fill_ratio
    fill = np.minimum(
        work["expected_demand"],
        cap.where(cap > 0, work["expected_demand"]),
    )
    work["sold_observed"] = work["sold"]
    work["sold_demand"] = np.where(work["is_censored_hour"], fill, work["sold"])
    work["imputed_demand"] = work["sold_demand"] - work["sold_observed"]
    return work


def reconstruct_stockout_demand_from_bakery_share(
    marked: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    min_bakery_active: float = 3.0,
    max_fill_ratio: float = 2.0,
) -> pd.DataFrame:
    """Fill censored zeroes as historical SKU share times current traffic."""
    work = marked.copy()
    work["dow"] = pd.to_datetime(work["date"]).dt.dayofweek
    work = work.merge(reference, on=PROFILE_KEYS, how="left")
    if "bakery_hour_sales" not in work.columns:
        work["bakery_hour_sales"] = work.groupby(["date", "bakery_id", "hour"])[
            "sold"
        ].transform("sum")
    work["expected_demand"] = work["mean_sku_share"] * work["bakery_hour_sales"]

    positive_hour = work["hour"].where(work["sold"] > 0)
    work["last_sale_hour"] = positive_hour.groupby(
        [work["date"], work["bakery_id"], work["product_id"]]
    ).transform("max")
    positive_rate = work["sold"].where(work["sold"] > 0)
    work["observed_positive_rate"] = positive_rate.groupby(
        [work["date"], work["bakery_id"], work["product_id"]]
    ).transform("mean")

    work["is_censored_hour"] = (
        work["is_stockout_day"].fillna(False)
        & (work["sold"] <= 0)
        & (work["hour"] > work["last_sale_hour"])
        & (work["bakery_hour_sales"] >= min_bakery_active)
        & work["expected_demand"].notna()
        & (work["expected_demand"] > 0)
    )
    cap = work["observed_positive_rate"] * max_fill_ratio
    fill = np.minimum(
        work["expected_demand"],
        cap.where(cap > 0, work["expected_demand"]),
    )
    work["sold_observed"] = work["sold"]
    work["sold_demand"] = np.where(work["is_censored_hour"], fill, work["sold"])
    work["imputed_demand"] = work["sold_demand"] - work["sold_observed"]
    return work


def aggregate_daily_training_target(reconstructed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconstructed SKU demand to a bakery-day training target."""
    return reconstructed.groupby(["date", "bakery_id"], as_index=False).agg(
        sales_observed=("sold_observed", "sum"),
        demand_target=("sold_demand", "sum"),
        imputed_demand=("imputed_demand", "sum"),
        censored_hours=("is_censored_hour", "sum"),
    )


def build_inventory_balance(
    daily: pd.DataFrame,
    moves: pd.DataFrame,
    *,
    produced_includes_opening_stock: bool = False,
    stock_tolerance: float = 0.5,
    balance_error_tolerance: float = 1.0,
) -> pd.DataFrame:
    """Combine opening stock, production, moves, sales, and closing stock."""
    keys = ["date", "bakery_id", "product_id"]
    work = daily.merge(moves, on=keys, how="left")
    for column in ["incoming_move_qty", "outgoing_move_qty"]:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    work["net_move_qty"] = work["incoming_move_qty"] - work["outgoing_move_qty"]
    opening_component = (
        0.0 if produced_includes_opening_stock else work["opening_stock"]
    )
    work["available_qty"] = opening_component + work["produced"] + work["net_move_qty"]
    work["expected_closing_stock"] = work["available_qty"] - work["sold"]
    work["balance_error"] = work["closing_stock"] - work["expected_closing_stock"]
    work["balance_is_consistent"] = (
        work["balance_error"].abs() <= balance_error_tolerance
    )
    work["is_inventory_stockout"] = (
        work["balance_is_consistent"]
        & (work["closing_stock"] <= stock_tolerance)
        & (work["available_qty"] > 0)
    )
    return work
