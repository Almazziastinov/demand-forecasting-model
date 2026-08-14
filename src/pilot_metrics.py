"""Canonical, side-effect-free metrics for the bakery forecast pilot.

The module deliberately does not read files or databases.  Callers must bind
facts to one forecast run and validate the pilot scope before constructing the
inputs below.

Lost-demand policy ``sales_rate_v1`` uses the conservative 07:00--19:00
operating window already used by the SKU correction pipeline.  Extrapolation
is allowed after at least two elapsed hours and is capped at the larger of
150% of observed sales or 15 units.  Raw lost demand is diagnostic;
``recognized_qty`` contributes to demand only for a fully realised release.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from math import isfinite
from typing import Iterable


class ForecastStatus(StrEnum):
    NORMAL = "normal"
    UNDERFORECAST = "underforecast"
    OVERFORECAST = "overforecast"
    NO_DATA = "no_data"


class ExecutionStatus(StrEnum):
    NORMAL = "normal"
    UNDERPRODUCTION = "underproduction"
    OVERPRODUCTION = "overproduction"
    NO_DATA = "no_data"


class AvailabilityStatus(StrEnum):
    NORMAL = "normal"
    STOCKOUT = "stockout"
    OVERSTOCK = "overstock"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class LostDemandPolicy:
    """Versioned policy for sales-rate lost-demand extrapolation."""

    version: str = "sales_rate_v1"
    opening_time: time = time(7)
    closing_time: time = time(19)
    minimum_elapsed_hours: float = 2.0
    sold_to_produced_tolerance: float = 0.01
    cap_sales_multiplier: float = 1.5
    cap_minimum_units: float = 15.0


@dataclass(frozen=True)
class LostDemandResult:
    raw_qty: float
    recognized_qty: float
    policy_version: str
    eligible: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class AggregateMetrics:
    row_count: int
    valid_row_count: int
    forecast_qty: float
    demand_qty: float
    error_qty: float
    absolute_error_qty: float
    underforecast_qty: float
    overforecast_qty: float
    wape: float | None
    bias: float | None


def _valid_number(value: float | None, *, nonnegative: bool = True) -> bool:
    return (
        value is not None and isfinite(value) and (value >= 0 if nonnegative else True)
    )


def calculate_lost_demand(
    *,
    sold_qty: float | None,
    produced_qty: float | None,
    closing_stock_qty: float | None,
    last_sale_at: datetime | None,
    policy: LostDemandPolicy = LostDemandPolicy(),
) -> LostDemandResult:
    """Return raw and recognized lost demand without changing reported facts."""

    if not all(_valid_number(value) for value in (sold_qty, produced_qty)):
        return LostDemandResult(
            0.0, 0.0, policy.version, False, "missing_or_invalid_quantity"
        )
    if not _valid_number(closing_stock_qty, nonnegative=False):
        return LostDemandResult(
            0.0, 0.0, policy.version, False, "missing_or_invalid_stock"
        )
    if sold_qty == 0:
        return LostDemandResult(0.0, 0.0, policy.version, True)
    if (
        closing_stock_qty > 0
        and sold_qty < produced_qty - policy.sold_to_produced_tolerance
    ):
        return LostDemandResult(0.0, 0.0, policy.version, True)
    if last_sale_at is None:
        return LostDemandResult(0.0, 0.0, policy.version, False, "missing_last_sale")

    opening = datetime.combine(
        last_sale_at.date(), policy.opening_time, last_sale_at.tzinfo
    )
    closing = datetime.combine(
        last_sale_at.date(), policy.closing_time, last_sale_at.tzinfo
    )
    elapsed_hours = (last_sale_at - opening).total_seconds() / 3600.0
    remaining_hours = (closing - last_sale_at).total_seconds() / 3600.0
    if elapsed_hours < policy.minimum_elapsed_hours:
        return LostDemandResult(
            0.0, 0.0, policy.version, False, "insufficient_elapsed_time"
        )
    if remaining_hours <= 0:
        return LostDemandResult(0.0, 0.0, policy.version, True)

    raw_qty = sold_qty / elapsed_hours * remaining_hours
    cap = max(sold_qty * policy.cap_sales_multiplier, policy.cap_minimum_units)
    raw_qty = min(raw_qty, cap)
    fully_realized = (
        produced_qty > 0
        and sold_qty >= produced_qty - policy.sold_to_produced_tolerance
        and closing_stock_qty <= 0
    )
    if not fully_realized:
        return LostDemandResult(
            raw_qty, 0.0, policy.version, False, "release_not_fully_realized"
        )
    return LostDemandResult(raw_qty, raw_qty, policy.version, True)


def estimated_demand(
    sold_qty: float | None, recognized_lost_qty: float | None
) -> float | None:
    """Demand contract: observed sales plus recognized (not raw) lost demand."""

    if not all(_valid_number(value) for value in (sold_qty, recognized_lost_qty)):
        return None
    return sold_qty + recognized_lost_qty


def classify_forecast(
    forecast_qty: float | None,
    demand_qty: float | None,
    *,
    tolerance_ratio: float = 0.20,
) -> ForecastStatus:
    if not all(_valid_number(value) for value in (forecast_qty, demand_qty)):
        return ForecastStatus.NO_DATA
    if not _valid_number(tolerance_ratio):
        raise ValueError("tolerance_ratio must be finite and nonnegative")
    tolerance = abs(demand_qty) * tolerance_ratio
    error = forecast_qty - demand_qty
    if abs(error) <= tolerance:
        return ForecastStatus.NORMAL
    return ForecastStatus.OVERFORECAST if error > 0 else ForecastStatus.UNDERFORECAST


def classify_execution(
    *,
    plan_qty: float | None,
    produced_qty: float | None,
    available_to_sell_qty: float | None = None,
    tolerance_ratio: float = 0.20,
    tolerance_units: float = 0.0,
) -> ExecutionStatus:
    """Compare the operational plan with production or available-to-sell units."""

    actual = (
        available_to_sell_qty if available_to_sell_qty is not None else produced_qty
    )
    if not all(_valid_number(value) for value in (plan_qty, actual)):
        return ExecutionStatus.NO_DATA
    if not _valid_number(tolerance_ratio) or not _valid_number(tolerance_units):
        raise ValueError("execution tolerances must be finite and nonnegative")
    tolerance = max(plan_qty * tolerance_ratio, tolerance_units)
    error = actual - plan_qty
    if abs(error) <= tolerance:
        return ExecutionStatus.NORMAL
    return (
        ExecutionStatus.OVERPRODUCTION if error > 0 else ExecutionStatus.UNDERPRODUCTION
    )


def classify_availability(
    *,
    sold_qty: float | None,
    closing_stock_qty: float | None,
    recognized_lost_qty: float | None,
    overstock_to_sales_ratio: float = 0.20,
) -> AvailabilityStatus:
    """Classify product availability independently from plan execution."""

    if not all(
        _valid_number(value, nonnegative=(name != "stock"))
        for name, value in (
            ("sold", sold_qty),
            ("stock", closing_stock_qty),
            ("lost", recognized_lost_qty),
        )
    ):
        return AvailabilityStatus.NO_DATA
    if not _valid_number(overstock_to_sales_ratio):
        raise ValueError("overstock_to_sales_ratio must be finite and nonnegative")
    if recognized_lost_qty > 0:
        return AvailabilityStatus.STOCKOUT
    if closing_stock_qty > sold_qty * overstock_to_sales_ratio:
        return AvailabilityStatus.OVERSTOCK
    return AvailabilityStatus.NORMAL


def aggregate_forecast_metrics(
    rows: Iterable[tuple[float | None, float | None]],
) -> AggregateMetrics:
    """Aggregate ``(forecast, demand)`` rows; invalid rows are excluded explicitly."""

    materialized = list(rows)
    valid = [
        (forecast, demand)
        for forecast, demand in materialized
        if _valid_number(forecast) and _valid_number(demand)
    ]
    forecast_total = sum(forecast for forecast, _ in valid)
    demand_total = sum(demand for _, demand in valid)
    errors = [forecast - demand for forecast, demand in valid]
    absolute_error = sum(abs(error) for error in errors)
    return AggregateMetrics(
        row_count=len(materialized),
        valid_row_count=len(valid),
        forecast_qty=forecast_total,
        demand_qty=demand_total,
        error_qty=sum(errors),
        absolute_error_qty=absolute_error,
        underforecast_qty=sum(max(-error, 0.0) for error in errors),
        overforecast_qty=sum(max(error, 0.0) for error in errors),
        wape=absolute_error / demand_total if demand_total > 0 else None,
        bias=sum(errors) / demand_total if demand_total > 0 else None,
    )
