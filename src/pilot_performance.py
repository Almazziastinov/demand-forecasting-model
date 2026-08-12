"""Side-effect-free construction of canonical pilot performance rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from typing import Iterable, Literal

from src.pilot_metrics import (
    AvailabilityStatus,
    ExecutionStatus,
    ForecastStatus,
    LostDemandPolicy,
    calculate_lost_demand,
    classify_availability,
    classify_execution,
    classify_forecast,
)


class DataQualityFlag(StrEnum):
    RUN_MISMATCH = "run_mismatch"
    OUT_OF_SCOPE = "out_of_scope"
    INVALID_FORECAST = "invalid_forecast"
    INVALID_PLAN = "invalid_plan"
    INVALID_PRODUCTION = "invalid_production"
    INVALID_SALES = "invalid_sales"
    INVALID_STOCK = "invalid_stock"
    NEGATIVE_STOCK = "negative_stock"
    TRANSFERS_INCOMPLETE = "transfers_incomplete"
    INVALID_TRANSFER = "invalid_transfer"
    AVAILABLE_BASIS_MISSING = "available_basis_missing"
    AVAILABLE_BALANCE_MISMATCH = "available_balance_mismatch"
    SALES_EXCEED_AVAILABLE = "sales_exceed_available"
    DEMAND_INCOMPLETE = "demand_incomplete"


ExecutionBasis = Literal["produced", "available_to_sell"]


@dataclass(frozen=True)
class PerformanceContract:
    metric_version: str
    scope_version: str
    forecast_run_id: str
    lost_demand_policy: LostDemandPolicy = LostDemandPolicy()
    inventory_equation_version: str = "inventory_v1"
    execution_basis: ExecutionBasis = "produced"
    forecast_tolerance_ratio: float = 0.20
    execution_tolerance_ratio: float = 0.20
    execution_tolerance_units: float = 0.0
    balance_tolerance_units: float = 0.01


@dataclass(frozen=True)
class PerformanceInput:
    business_date: date
    bakery_id: int
    product_id: int
    forecast_run_id: str
    forecast_qty: float | None
    plan_qty: float | None
    produced_qty: float | None
    sold_qty: float | None
    closing_stock_qty: float | None
    last_sale_at: datetime | None
    opening_stock_qty: float | None = None
    received_qty: float | None = None
    sent_qty: float | None = None
    transfers_complete: bool = False
    available_to_sell_qty: float | None = None
    available_to_sell_basis: str | None = None

    @property
    def key(self) -> tuple[date, int, int]:
        return self.business_date, self.bakery_id, self.product_id


@dataclass(frozen=True)
class PerformanceRow:
    business_date: date
    bakery_id: int
    product_id: int
    forecast_run_id: str
    scope_version: str
    metric_version: str
    lost_policy_version: str
    inventory_equation_version: str
    forecast_qty: float | None
    plan_qty: float | None
    produced_qty: float | None
    sold_qty: float | None
    closing_stock_qty: float | None
    received_qty: float | None
    sent_qty: float | None
    available_to_sell_qty: float | None
    available_to_sell_basis: str | None
    lost_demand_raw_qty: float
    lost_demand_recognized_qty: float
    observed_demand_floor_qty: float | None
    demand_qty: float | None
    demand_is_complete: bool
    forecast_status: ForecastStatus
    execution_status: ExecutionStatus
    availability_status: AvailabilityStatus
    eligible_forecast: bool
    eligible_execution: bool
    eligible_lost_demand: bool
    eligible_availability: bool
    blocking_flags: tuple[DataQualityFlag, ...]
    forecast_flags: tuple[DataQualityFlag, ...]
    execution_flags: tuple[DataQualityFlag, ...]
    lost_demand_flags: tuple[DataQualityFlag, ...]
    availability_flags: tuple[DataQualityFlag, ...]
    dq_flags: tuple[DataQualityFlag, ...]
    lost_demand_rejection_reason: str | None


def _valid_nonnegative(value: float | None) -> bool:
    return value is not None and isfinite(value) and value >= 0


def _unique(*groups: Iterable[DataQualityFlag]) -> tuple[DataQualityFlag, ...]:
    return tuple(dict.fromkeys(flag for group in groups for flag in group))


def _validate_contract(contract: PerformanceContract) -> None:
    for name in (
        "metric_version",
        "scope_version",
        "forecast_run_id",
        "inventory_equation_version",
    ):
        if not getattr(contract, name).strip():
            raise ValueError(f"{name} must not be empty")
    if contract.execution_basis not in ("produced", "available_to_sell"):
        raise ValueError("unsupported execution_basis")


def _inventory_flags(
    record: PerformanceInput, contract: PerformanceContract
) -> tuple[DataQualityFlag, ...]:
    flags: list[DataQualityFlag] = []
    if not record.transfers_complete:
        flags.append(DataQualityFlag.TRANSFERS_INCOMPLETE)
        return tuple(flags)
    if not all(
        _valid_nonnegative(value)
        for value in (record.opening_stock_qty, record.received_qty, record.sent_qty)
    ):
        flags.append(DataQualityFlag.INVALID_TRANSFER)
        return tuple(flags)
    if record.available_to_sell_qty is not None:
        if not record.available_to_sell_basis:
            flags.append(DataQualityFlag.AVAILABLE_BASIS_MISSING)
        elif not _valid_nonnegative(record.available_to_sell_qty):
            flags.append(DataQualityFlag.INVALID_PRODUCTION)
        elif _valid_nonnegative(record.produced_qty):
            expected = (
                record.opening_stock_qty
                + record.produced_qty
                + record.received_qty
                - record.sent_qty
            )
            if (
                abs(record.available_to_sell_qty - expected)
                > contract.balance_tolerance_units
            ):
                flags.append(DataQualityFlag.AVAILABLE_BALANCE_MISMATCH)
    return tuple(flags)


def build_performance_rows(
    records: Iterable[PerformanceInput],
    *,
    contract: PerformanceContract,
    scope_bakery_ids: Iterable[int] | None = None,
) -> list[PerformanceRow]:
    """Build rows with independent eligibility for each metric family."""

    _validate_contract(contract)
    materialized = list(records)
    keys = [record.key for record in materialized]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate date/bakery/product key")
    if any(
        record.bakery_id <= 0
        or record.product_id <= 0
        or not isinstance(record.business_date, date)
        for record in materialized
    ):
        raise ValueError("business_date and positive bakery/product keys are required")

    scope = frozenset(scope_bakery_ids) if scope_bakery_ids is not None else None
    rows: list[PerformanceRow] = []
    for record in materialized:
        blocking: list[DataQualityFlag] = []
        if record.forecast_run_id != contract.forecast_run_id:
            blocking.append(DataQualityFlag.RUN_MISMATCH)
        if scope is not None and record.bakery_id not in scope:
            blocking.append(DataQualityFlag.OUT_OF_SCOPE)

        forecast_flags = (
            []
            if _valid_nonnegative(record.forecast_qty)
            else [DataQualityFlag.INVALID_FORECAST]
        )
        execution_flags = (
            []
            if _valid_nonnegative(record.plan_qty)
            else [DataQualityFlag.INVALID_PLAN]
        )
        if not _valid_nonnegative(record.produced_qty):
            execution_flags.append(DataQualityFlag.INVALID_PRODUCTION)
        lost_flags: list[DataQualityFlag] = []
        availability_flags: list[DataQualityFlag] = []
        if not _valid_nonnegative(record.sold_qty):
            lost_flags.append(DataQualityFlag.INVALID_SALES)
            availability_flags.append(DataQualityFlag.INVALID_SALES)
        if record.closing_stock_qty is None or not isfinite(record.closing_stock_qty):
            lost_flags.append(DataQualityFlag.INVALID_STOCK)
            availability_flags.append(DataQualityFlag.INVALID_STOCK)
        elif record.closing_stock_qty < 0:
            lost_flags.append(DataQualityFlag.NEGATIVE_STOCK)
            availability_flags.append(DataQualityFlag.NEGATIVE_STOCK)

        inventory_flags = _inventory_flags(record, contract)
        lost_flags.extend(inventory_flags)
        availability_flags.extend(inventory_flags)
        if contract.execution_basis == "available_to_sell":
            execution_flags.extend(inventory_flags)
            if (
                record.available_to_sell_qty is None
                or not record.available_to_sell_basis
            ):
                execution_flags.append(DataQualityFlag.AVAILABLE_BASIS_MISSING)

        if (
            _valid_nonnegative(record.sold_qty)
            and _valid_nonnegative(record.available_to_sell_qty)
            and record.sold_qty
            > record.available_to_sell_qty + contract.balance_tolerance_units
        ):
            availability_flags.append(DataQualityFlag.SALES_EXCEED_AVAILABLE)

        lost = calculate_lost_demand(
            sold_qty=record.sold_qty,
            produced_qty=record.produced_qty,
            closing_stock_qty=record.closing_stock_qty,
            last_sale_at=record.last_sale_at,
            policy=contract.lost_demand_policy,
        )
        if lost.rejection_reason is not None:
            lost_flags.append(DataQualityFlag.DEMAND_INCOMPLETE)

        blocking_tuple = _unique(blocking)
        lost_tuple = _unique(lost_flags)
        demand_complete = not blocking_tuple and not lost_tuple
        observed_floor = (
            record.sold_qty if _valid_nonnegative(record.sold_qty) else None
        )
        demand = (
            observed_floor + lost.recognized_qty if demand_complete else observed_floor
        )

        forecast_tuple = _unique(forecast_flags)
        if not demand_complete:
            forecast_tuple = _unique(
                forecast_tuple, (DataQualityFlag.DEMAND_INCOMPLETE,)
            )
        execution_tuple = _unique(execution_flags)
        availability_tuple = _unique(availability_flags)
        if not demand_complete:
            availability_tuple = _unique(
                availability_tuple, (DataQualityFlag.DEMAND_INCOMPLETE,)
            )

        eligible_forecast = not blocking_tuple and not forecast_tuple
        eligible_execution = not blocking_tuple and not execution_tuple
        eligible_lost = not blocking_tuple and not lost_tuple
        eligible_availability = not blocking_tuple and not availability_tuple
        actual_available = (
            record.available_to_sell_qty
            if contract.execution_basis == "available_to_sell"
            else None
        )
        rows.append(
            PerformanceRow(
                business_date=record.business_date,
                bakery_id=record.bakery_id,
                product_id=record.product_id,
                forecast_run_id=record.forecast_run_id,
                scope_version=contract.scope_version,
                metric_version=contract.metric_version,
                lost_policy_version=lost.policy_version,
                inventory_equation_version=contract.inventory_equation_version,
                forecast_qty=record.forecast_qty,
                plan_qty=record.plan_qty,
                produced_qty=record.produced_qty,
                sold_qty=record.sold_qty,
                closing_stock_qty=record.closing_stock_qty,
                received_qty=record.received_qty,
                sent_qty=record.sent_qty,
                available_to_sell_qty=record.available_to_sell_qty,
                available_to_sell_basis=record.available_to_sell_basis,
                lost_demand_raw_qty=lost.raw_qty,
                lost_demand_recognized_qty=lost.recognized_qty,
                observed_demand_floor_qty=observed_floor,
                demand_qty=demand,
                demand_is_complete=demand_complete,
                forecast_status=(
                    classify_forecast(
                        record.forecast_qty,
                        demand,
                        tolerance_ratio=contract.forecast_tolerance_ratio,
                    )
                    if eligible_forecast
                    else ForecastStatus.NO_DATA
                ),
                execution_status=(
                    classify_execution(
                        plan_qty=record.plan_qty,
                        produced_qty=record.produced_qty,
                        available_to_sell_qty=actual_available,
                        tolerance_ratio=contract.execution_tolerance_ratio,
                        tolerance_units=contract.execution_tolerance_units,
                    )
                    if eligible_execution
                    else ExecutionStatus.NO_DATA
                ),
                availability_status=(
                    classify_availability(
                        sold_qty=record.sold_qty,
                        closing_stock_qty=record.closing_stock_qty,
                        recognized_lost_qty=lost.recognized_qty,
                    )
                    if eligible_availability
                    else AvailabilityStatus.NO_DATA
                ),
                eligible_forecast=eligible_forecast,
                eligible_execution=eligible_execution,
                eligible_lost_demand=eligible_lost,
                eligible_availability=eligible_availability,
                blocking_flags=blocking_tuple,
                forecast_flags=forecast_tuple,
                execution_flags=execution_tuple,
                lost_demand_flags=lost_tuple,
                availability_flags=availability_tuple,
                dq_flags=_unique(
                    blocking_tuple,
                    forecast_tuple,
                    execution_tuple,
                    lost_tuple,
                    availability_tuple,
                ),
                lost_demand_rejection_reason=lost.rejection_reason,
            )
        )
    return rows
