"""Conservative corrections for persistent bakery/SKU forecast bias.

The correction is deliberately separate from the production allocator. It
builds a time-limited registry from information available strictly before the
forecast date, applies adaptively smoothed multipliers, and restores the
original bakery/category total after changing the SKU mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "category_name"
FORECAST_COL = "forecast_qty"
DEMAND_COL = "demand_qty"


@dataclass(frozen=True)
class CorrectionConfig:
    history_days: int = 49
    recent_days: int = 7
    min_observed_days: int = 28
    min_forecast_days: int = 14
    min_age_days: int = 28
    min_demand_qty: float = 150.0
    min_abs_bias: float = 0.15
    min_directionality: float = 0.40
    min_recent_abs_bias: float = 0.10
    min_smoothing: float = 0.10
    max_smoothing: float = 0.30
    recent_bias_full_strength: float = 0.50
    volume_full_strength_multiple: float = 4.0
    ttl_days: int = 14


REQUIRED_HISTORY_COLUMNS = {
    DATE_COL,
    BAKERY_ID_COL,
    PRODUCT_ID_COL,
    CATEGORY_COL,
    FORECAST_COL,
    DEMAND_COL,
}


def _validate_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _summarize_errors(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["error_qty"] = work[FORECAST_COL] - work[DEMAND_COL]
    work["abs_error_qty"] = work["error_qty"].abs()
    work["positive_forecast_day"] = work[FORECAST_COL].gt(0).astype(int)
    has_lost_demand = "lost_demand_qty" in work.columns
    if has_lost_demand:
        work["lost_demand_qty"] = pd.to_numeric(
            work["lost_demand_qty"],
            errors="coerce",
        ).fillna(0.0)
        work["lost_demand_day"] = work["lost_demand_qty"].gt(0).astype(int)
    group_cols = [BAKERY_ID_COL, PRODUCT_ID_COL, CATEGORY_COL]
    if PRODUCT_NAME_COL in work.columns:
        group_cols.append(PRODUCT_NAME_COL)
    aggregations = {
        "observed_days": (DATE_COL, "nunique"),
        "forecast_days": ("positive_forecast_day", "sum"),
        "first_observed_date": (DATE_COL, "min"),
        "forecast_qty": (FORECAST_COL, "sum"),
        "demand_qty": (DEMAND_COL, "sum"),
        "net_error_qty": ("error_qty", "sum"),
        "absolute_error_qty": ("abs_error_qty", "sum"),
    }
    if has_lost_demand:
        aggregations.update(
            {
                "lost_demand_qty": ("lost_demand_qty", "sum"),
                "lost_demand_days": ("lost_demand_day", "sum"),
            }
        )
    summary = work.groupby(
        group_cols,
        as_index=False,
        dropna=False,
    ).agg(**aggregations)
    summary["bias"] = (
        summary["forecast_qty"] / summary["demand_qty"].replace(0.0, np.nan) - 1.0
    )
    summary["directionality"] = (
        summary["net_error_qty"].abs()
        / summary["absolute_error_qty"].replace(0.0, np.nan)
    )
    if has_lost_demand:
        summary["lost_demand_share"] = (
            summary["lost_demand_qty"]
            / summary["demand_qty"].replace(0.0, np.nan)
        )
    return summary


def _adaptive_smoothing(
    registry: pd.DataFrame,
    *,
    config: CorrectionConfig,
) -> pd.Series:
    direction_score = (
        (registry["directionality"] - config.min_directionality)
        / (1.0 - config.min_directionality)
    ).clip(0.0, 1.0)
    recent_score = (
        (registry["recent_bias"].abs() - config.min_recent_abs_bias)
        / (
            config.recent_bias_full_strength
            - config.min_recent_abs_bias
        )
    ).clip(0.0, 1.0)
    history_score = (
        registry["observed_days"] / float(config.history_days)
    ).clip(0.0, 1.0)
    volume_score = (
        registry["demand_qty"]
        / (
            config.min_demand_qty
            * config.volume_full_strength_multiple
        )
    ).clip(0.0, 1.0)

    if {"lost_demand_days", "lost_demand_share"}.issubset(registry.columns):
        repeated_lost_score = (
            registry["lost_demand_days"]
            / registry["observed_days"].clip(lower=1)
            * 4.0
        ).clip(0.0, 1.0)
        lost_quality = (
            repeated_lost_score
            * (1.0 - 0.5 * registry["lost_demand_share"].clip(0.0, 1.0))
        )
        # An overforecast that remains after adding missed demand is already
        # robust to censoring, so lost-demand reliability need not reduce it.
        lost_quality = np.where(
            registry["net_error_qty"] > 0,
            1.0,
            lost_quality,
        )
        lost_quality = pd.Series(lost_quality, index=registry.index)
    else:
        lost_quality = pd.Series(0.5, index=registry.index)

    smoothing = (
        0.30 * direction_score
        + 0.20 * recent_score
        + 0.15 * history_score
        + 0.15 * volume_score
        + 0.20 * lost_quality
    )
    return smoothing.clip(config.min_smoothing, config.max_smoothing)


def build_correction_registry(
    history: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    config: CorrectionConfig = CorrectionConfig(),
) -> pd.DataFrame:
    """Build corrections using rows strictly earlier than ``as_of_date``."""
    _validate_columns(history, REQUIRED_HISTORY_COLUMNS)
    as_of = pd.Timestamp(as_of_date).normalize()
    work = history.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce").dt.normalize()
    work[FORECAST_COL] = pd.to_numeric(work[FORECAST_COL], errors="coerce").fillna(0.0)
    work[DEMAND_COL] = pd.to_numeric(work[DEMAND_COL], errors="coerce").fillna(0.0)
    work = work[
        work[DATE_COL].between(
            as_of - pd.Timedelta(days=config.history_days),
            as_of - pd.Timedelta(days=1),
        )
    ].copy()
    work = work[work[DEMAND_COL] > 0].copy()
    if work.empty:
        return pd.DataFrame()

    summary = _summarize_errors(work)
    recent = _summarize_errors(
        work[work[DATE_COL] >= as_of - pd.Timedelta(days=config.recent_days)]
    )
    recent = recent[
        [BAKERY_ID_COL, PRODUCT_ID_COL, "bias", "net_error_qty"]
    ].rename(
        columns={
            "bias": "recent_bias",
            "net_error_qty": "recent_net_error_qty",
        }
    )
    summary = summary.merge(
        recent,
        on=[BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
        validate="one_to_one",
    )
    summary["age_days"] = (as_of - summary["first_observed_date"]).dt.days
    same_direction = (
        np.sign(summary["net_error_qty"])
        == np.sign(summary["recent_net_error_qty"])
    )
    eligible = (
        (summary["observed_days"] >= config.min_observed_days)
        & (summary["forecast_days"] >= config.min_forecast_days)
        & (summary["age_days"] >= config.min_age_days)
        & (summary["demand_qty"] >= config.min_demand_qty)
        & (summary["forecast_qty"] > 0)
        & (summary["bias"].abs() >= config.min_abs_bias)
        & (summary["directionality"] >= config.min_directionality)
        & (summary["recent_bias"].abs() >= config.min_recent_abs_bias)
        & same_direction
    )
    registry = summary[eligible].copy()
    if registry.empty:
        return registry

    registry["full_multiplier"] = (
        registry["demand_qty"] / registry["forecast_qty"]
    )
    registry["smoothing"] = _adaptive_smoothing(
        registry,
        config=config,
    )
    registry["multiplier"] = np.exp(
        registry["smoothing"]
        * np.log(registry["full_multiplier"])
    )
    registry["direction"] = np.where(
        registry["net_error_qty"] < 0,
        "underforecast",
        "overforecast",
    )
    registry["valid_from"] = as_of
    registry["valid_to"] = as_of + pd.Timedelta(days=config.ttl_days - 1)
    return registry.sort_values(
        ["directionality", "demand_qty"],
        ascending=[False, False],
    ).reset_index(drop=True)


def apply_category_neutral_corrections(
    forecast: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    forecast_col: str = FORECAST_COL,
    output_col: str = "corrected_forecast_qty",
) -> pd.DataFrame:
    """Apply registry multipliers while preserving bakery/category totals."""
    required = {
        DATE_COL,
        BAKERY_ID_COL,
        PRODUCT_ID_COL,
        CATEGORY_COL,
        forecast_col,
    }
    _validate_columns(forecast, required)
    work = forecast.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce").dt.normalize()
    work[forecast_col] = pd.to_numeric(work[forecast_col], errors="coerce").fillna(0.0)
    if registry.empty:
        work["correction_multiplier"] = 1.0
        work[output_col] = work[forecast_col]
        return work

    lookup = registry[
        [
            BAKERY_ID_COL,
            PRODUCT_ID_COL,
            "multiplier",
            "valid_from",
            "valid_to",
        ]
    ].copy()
    work = work.merge(
        lookup,
        on=[BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
        validate="many_to_one",
    )
    active = work[DATE_COL].between(work["valid_from"], work["valid_to"])
    work["correction_multiplier"] = np.where(
        active,
        work["multiplier"].fillna(1.0),
        1.0,
    )
    work["_candidate_forecast"] = (
        work[forecast_col] * work["correction_multiplier"]
    )
    group_cols = [DATE_COL, BAKERY_ID_COL, CATEGORY_COL]
    original_total = work.groupby(group_cols)[forecast_col].transform("sum")
    candidate_total = work.groupby(group_cols)["_candidate_forecast"].transform("sum")
    scale = original_total / candidate_total.replace(0.0, np.nan)
    work[output_col] = (work["_candidate_forecast"] * scale).fillna(
        work[forecast_col]
    )
    work.drop(
        columns=[
            "_candidate_forecast",
            "multiplier",
            "valid_from",
            "valid_to",
        ],
        inplace=True,
    )
    return work


def forecast_metrics(
    frame: pd.DataFrame,
    *,
    forecast_col: str,
    demand_col: str = DEMAND_COL,
) -> dict[str, float]:
    forecast = pd.to_numeric(frame[forecast_col], errors="coerce").fillna(0.0)
    demand = pd.to_numeric(frame[demand_col], errors="coerce").fillna(0.0)
    error = forecast - demand
    demand_total = float(demand.sum())
    return {
        "rows": int(len(frame)),
        "forecast_qty": float(forecast.sum()),
        "demand_qty": demand_total,
        "bias_pct": float(error.sum() / demand_total * 100.0)
        if demand_total
        else 0.0,
        "wape_pct": float(error.abs().sum() / demand_total * 100.0)
        if demand_total
        else 0.0,
        "underforecast_qty": float((-error).clip(lower=0.0).sum()),
        "overforecast_qty": float(error.clip(lower=0.0).sum()),
    }
