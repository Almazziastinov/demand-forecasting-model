"""Causal bakery-day/category-day allocation to SKU-day quantities.

The allocator deliberately has no hourly inputs.  Intraday profiles may be
applied after this module, but they cannot change the resulting SKU-day total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP_KEYS = ["bakery_id", "category"]
SKU_KEYS = [*GROUP_KEYS, "product_id"]


@dataclass(frozen=True)
class DailySkuAllocationConfig:
    recent_days: int = 14
    history_days: int = 365
    local_prior_days: float = 28.0
    recent_prior_days: float = 14.0
    weekday_max_weight: float = 0.25
    min_share: float = 1e-10


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _shares(
    frame: pd.DataFrame,
    *,
    numerator_keys: list[str],
    denominator_keys: list[str],
    target_col: str,
    output_col: str,
) -> pd.DataFrame:
    numerator = frame.groupby(numerator_keys, as_index=False)[target_col].sum()
    denominator = frame.groupby(denominator_keys, as_index=False)[target_col].sum()
    denominator = denominator.rename(columns={target_col: "_denominator"})
    result = numerator.merge(denominator, on=denominator_keys, how="left")
    result[output_col] = result[target_col] / result["_denominator"].replace(0.0, np.nan)
    return result[numerator_keys + [output_col]]


def build_daily_sku_shares(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    *,
    target_col: str = "demand_mid",
    config: DailySkuAllocationConfig | None = None,
) -> pd.DataFrame:
    """Build causal daily SKU shares for a known forecast-date universe.

    ``history`` must contain observations strictly before ``forecast_date``.
    Rows on or after the forecast date are rejected instead of silently used.
    ``universe`` is the assortment known at the forecast origin and therefore
    controls which SKU receive allocation, including cold-start SKU.
    """
    cfg = config or DailySkuAllocationConfig()
    required_history = {"date", "bakery_id", "city", "category", "product_id", target_col}
    required_universe = {"bakery_id", "city", "category", "product_id"}
    _validate_columns(history, required_history, "history")
    _validate_columns(universe, required_universe, "universe")

    cutoff = pd.Timestamp(forecast_date).normalize()
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    if hist["date"].ge(cutoff).any():
        raise ValueError("history contains rows on or after forecast_date")
    hist[target_col] = pd.to_numeric(hist[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist[hist["date"].ge(cutoff - pd.Timedelta(days=cfg.history_days))]

    out = universe[["bakery_id", "city", "category", "product_id"]].drop_duplicates().copy()
    if out.empty:
        return out.assign(sku_share=pd.Series(dtype=float), allocation_source=pd.Series(dtype=str))

    pair = hist.groupby(SKU_KEYS, as_index=False).agg(
        pair_demand=(target_col, "sum"), pair_days=("date", "nunique")
    )
    pair_total = hist.groupby(GROUP_KEYS, as_index=False)[target_col].sum().rename(
        columns={target_col: "pair_total"}
    )
    pair = pair.merge(pair_total, on=GROUP_KEYS, how="left")
    pair["pair_share"] = pair["pair_demand"] / pair["pair_total"].replace(0.0, np.nan)

    recent = hist[hist["date"].ge(cutoff - pd.Timedelta(days=cfg.recent_days))]
    weekday = hist[hist["date"].dt.dayofweek.eq(cutoff.dayofweek)]
    recent_share = _shares(
        recent,
        numerator_keys=SKU_KEYS,
        denominator_keys=GROUP_KEYS,
        target_col=target_col,
        output_col="recent_share",
    )
    weekday_share = _shares(
        weekday,
        numerator_keys=SKU_KEYS,
        denominator_keys=GROUP_KEYS,
        target_col=target_col,
        output_col="weekday_share",
    )
    city_share = _shares(
        hist,
        numerator_keys=["city", "category", "product_id"],
        denominator_keys=["city", "category"],
        target_col=target_col,
        output_col="city_share",
    )
    network_share = _shares(
        hist,
        numerator_keys=["category", "product_id"],
        denominator_keys=["category"],
        target_col=target_col,
        output_col="network_share",
    )

    out = out.merge(pair, on=SKU_KEYS, how="left")
    out = out.merge(recent_share, on=SKU_KEYS, how="left")
    out = out.merge(weekday_share, on=SKU_KEYS, how="left")
    out = out.merge(city_share, on=["city", "category", "product_id"], how="left")
    out = out.merge(network_share, on=["category", "product_id"], how="left")

    pooled = out["city_share"].combine_first(out["network_share"])
    local = out["pair_share"].combine_first(pooled)
    local_weight = (out["pair_days"].fillna(0.0) / cfg.local_prior_days).clip(0.0, 1.0)
    base = local_weight * local.fillna(0.0) + (1.0 - local_weight) * pooled.fillna(local).fillna(0.0)
    recent_weight = (out["pair_days"].fillna(0.0) / cfg.recent_prior_days).clip(0.0, 0.65)
    raw = recent_weight * out["recent_share"].fillna(base) + (1.0 - recent_weight) * base
    weekday_weight = (out["pair_days"].fillna(0.0) / cfg.local_prior_days).clip(
        0.0, cfg.weekday_max_weight
    )
    raw = weekday_weight * out["weekday_share"].fillna(raw) + (1.0 - weekday_weight) * raw

    group_size = out.groupby(GROUP_KEYS)["product_id"].transform("size")
    no_evidence = raw.groupby([out[key] for key in GROUP_KEYS]).transform("sum").le(0.0)
    raw = raw.mask(no_evidence, 1.0 / group_size).clip(lower=cfg.min_share)
    out["sku_share"] = raw / raw.groupby([out[key] for key in GROUP_KEYS]).transform("sum")
    out["allocation_source"] = np.select(
        [out["pair_share"].notna(), out["city_share"].notna(), out["network_share"].notna()],
        ["local", "city", "network"],
        default="uniform",
    )
    return out[["bakery_id", "city", "category", "product_id", "sku_share", "allocation_source"]]


def allocate_category_totals(
    category_forecast: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    forecast_col: str = "category_forecast",
) -> pd.DataFrame:
    """Allocate category-day totals while preserving every positive total."""
    _validate_columns(category_forecast, {"bakery_id", "category", forecast_col}, "category_forecast")
    _validate_columns(shares, {*SKU_KEYS, "sku_share"}, "shares")
    result = shares.merge(
        category_forecast[["bakery_id", "category", forecast_col]],
        on=GROUP_KEYS,
        how="inner",
        validate="many_to_one",
    )
    result["sku_day_forecast"] = result["sku_share"] * result[forecast_col]
    return result
