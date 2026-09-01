"""Selected daily Direct SKU allocation post-processing contract.

This module contains no ClickHouse writes and no hourly/category allocation.
It converts Direct P50 quantities plus causal expected-loss evidence into the
frozen alpha=.25 integration candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DAY_KEYS = ["date", "bakery_id"]


@dataclass(frozen=True)
class DirectAlphaAllocationConfig:
    alpha: float = 0.25
    core_volume_share: float = 0.70
    floor_min_history: int = 8
    floor_min_stockout_rate: float = 0.75
    floor_min_lost_mean: float = 4.0
    floor_scale: float = 0.80
    floor_unit_cap: float = 5.0
    floor_relative_cap: float = 0.10
    tail_min_history: int = 4
    tail_share_threshold: float = 0.18
    tail_p67_scale: float = 1.0


REQUIRED_COLUMNS = {
    "date",
    "bakery_id",
    "product_id",
    "direct_p50",
    "predictive_uplift",
    "loss_scale",
    "broad_56_mean",
    "floor_history_n",
    "floor_demand_p67",
    "historical_stockout_rate",
    "historical_lost_mean",
}


def _validate(rows: pd.DataFrame, config: DirectAlphaAllocationConfig) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(rows.columns))
    if missing:
        raise ValueError(f"Direct allocation input is missing columns: {missing}")
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if not 0.0 < config.core_volume_share <= 1.0:
        raise ValueError("core_volume_share must be in (0, 1]")


def _add_core_flag(
    rows: pd.DataFrame, config: DirectAlphaAllocationConfig
) -> pd.DataFrame:
    result = rows.copy()
    result["historical_volume"] = result["broad_56_mean"].clip(lower=0.0)
    result = result.sort_values(
        [*DAY_KEYS, "historical_volume", "product_id"],
        ascending=[True, True, False, True],
    )
    groups = [result[key] for key in DAY_KEYS]
    total = result["historical_volume"].groupby(groups).transform("sum")
    share = result["historical_volume"] / total.replace(0.0, np.nan)
    cumulative_before = share.groupby(groups).cumsum() - share
    result["is_core_sku"] = cumulative_before.lt(config.core_volume_share) & share.gt(
        0.0
    )
    return result.sort_index()


def _protect_core(
    candidate: pd.Series,
    base: pd.Series,
    target: pd.Series,
    core: pd.Series,
    rows: pd.DataFrame,
) -> pd.Series:
    protected = candidate.where(~core, np.maximum(candidate, base))
    groups = [rows[key] for key in DAY_KEYS]
    core_value = protected.where(core, 0.0).groupby(groups).transform("sum")
    noncore_value = protected.where(~core, 0.0).groupby(groups).transform("sum")
    noncore_factor = ((target - core_value) / noncore_value.replace(0.0, np.nan)).clip(
        lower=0.0
    )
    return protected.where(core, protected * noncore_factor.fillna(0.0))


def build_selected_direct_plan(
    rows: pd.DataFrame,
    config: DirectAlphaAllocationConfig | None = None,
) -> pd.DataFrame:
    """Build the frozen alpha=.25/floor/tail-cap daily SKU candidate."""
    cfg = config or DirectAlphaAllocationConfig()
    _validate(rows, cfg)
    result = _add_core_flag(rows.reset_index(drop=True), cfg)
    groups = [result[key] for key in DAY_KEYS]

    base = result["direct_p50"].clip(lower=0.0)
    uplift = result["loss_scale"].clip(lower=0.0) * result["predictive_uplift"].clip(
        lower=0.0
    )
    base_total = base.groupby(groups).transform("sum")
    uplift_total = uplift.groupby(groups).transform("sum")
    target = base_total + cfg.alpha * uplift_total
    pre_normalized = base + uplift
    pre_total = pre_normalized.groupby(groups).transform("sum")
    normalized = pre_normalized / pre_total.replace(0.0, np.nan) * target
    result["direct_alpha"] = _protect_core(
        normalized.fillna(base), base, target, result["is_core_sku"], result
    ).clip(lower=0.0)

    eligible_floor = (
        result["floor_history_n"].ge(cfg.floor_min_history)
        & result["historical_stockout_rate"].ge(cfg.floor_min_stockout_rate)
        & result["historical_lost_mean"].ge(cfg.floor_min_lost_mean)
    )
    floor = result["floor_demand_p67"] * cfg.floor_scale
    floor_cap = np.minimum(
        result["direct_alpha"] + cfg.floor_unit_cap,
        result["direct_alpha"] * (1.0 + cfg.floor_relative_cap),
    )
    result["direct_alpha_floor"] = np.where(
        eligible_floor,
        np.maximum(result["direct_alpha"], np.minimum(floor, floor_cap)),
        result["direct_alpha"],
    )

    floor_total = result["direct_alpha_floor"].groupby(groups).transform("sum")
    floor_share = result["direct_alpha_floor"] / floor_total.replace(0.0, np.nan)
    tail_bound = result["floor_demand_p67"] * cfg.tail_p67_scale
    result["tail_cap_applied"] = (
        result["floor_history_n"].ge(cfg.tail_min_history)
        & floor_share.gt(cfg.tail_share_threshold)
        & result["direct_alpha_floor"].gt(tail_bound)
        & tail_bound.gt(0.0)
    )
    result["selected_sku_forecast"] = result["direct_alpha_floor"].where(
        ~result["tail_cap_applied"],
        np.minimum(result["direct_alpha_floor"], tail_bound),
    )
    result["selected_sku_forecast"] = result["selected_sku_forecast"].clip(lower=0.0)
    return result
