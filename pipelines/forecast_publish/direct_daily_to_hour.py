"""Expand authoritative Direct SKU-day quantities to serving-compatible hours."""

from __future__ import annotations

import numpy as np
import pandas as pd


DAY_KEYS = ["date", "bakery_id"]
SKU_DAY_KEYS = [*DAY_KEYS, "product_id"]


def expand_direct_sku_day_to_hour(
    sku_day: pd.DataFrame,
    bakery_hour_profile: pd.DataFrame,
    *,
    forecast_col: str = "sku_day_forecast",
    share_col: str = "mean_hour_share_norm",
) -> pd.DataFrame:
    """Apply bakery-only hourly timing without changing any SKU-day quantity."""
    required_day = {*SKU_DAY_KEYS, forecast_col}
    required_profile = {"bakery_id", "dow", "hour", share_col}
    missing_day = sorted(required_day.difference(sku_day.columns))
    missing_profile = sorted(required_profile.difference(bakery_hour_profile.columns))
    if missing_day:
        raise ValueError(f"sku_day is missing columns: {missing_day}")
    if missing_profile:
        raise ValueError(f"bakery_hour_profile is missing columns: {missing_profile}")

    daily = sku_day.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily[forecast_col] = pd.to_numeric(daily[forecast_col], errors="raise").clip(
        lower=0.0
    )
    daily["dow"] = daily["date"].dt.dayofweek

    profile = bakery_hour_profile[["bakery_id", "dow", "hour", share_col]].copy()
    profile[share_col] = (
        pd.to_numeric(profile[share_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    profile = profile.groupby(["bakery_id", "dow", "hour"], as_index=False)[
        share_col
    ].mean()
    totals = profile.groupby(["bakery_id", "dow"])[share_col].transform("sum")
    profile["hour_share"] = profile[share_col] / totals.replace(0.0, np.nan)
    profile = profile[profile["hour_share"].gt(0.0)]

    result = daily.merge(
        profile[["bakery_id", "dow", "hour", "hour_share"]],
        on=["bakery_id", "dow"],
        how="left",
        validate="many_to_many",
    )
    missing = result["hour_share"].isna()
    if missing.any():
        bakeries = sorted(result.loc[missing, "bakery_id"].astype(int).unique())
        raise ValueError(f"Missing bakery hourly profile for bakeries: {bakeries[:20]}")
    result["sku_hour_forecast"] = result[forecast_col] * result["hour_share"]
    check = result.groupby(SKU_DAY_KEYS)["sku_hour_forecast"].sum()
    expected = daily.set_index(SKU_DAY_KEYS)[forecast_col]
    max_error = float((check - expected).abs().max()) if not check.empty else 0.0
    if max_error > 1e-8:
        raise RuntimeError(f"SKU-day conservation failed: max_error={max_error}")
    return result.drop(columns=["dow", "hour_share"])
