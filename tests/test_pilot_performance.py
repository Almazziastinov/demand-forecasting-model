from datetime import date, datetime

import pytest

from src.pilot_metrics import AvailabilityStatus, ExecutionStatus, ForecastStatus
from src.pilot_performance import (
    DataQualityFlag,
    PerformanceContract,
    PerformanceInput,
    build_performance_rows,
)


CONTRACT = PerformanceContract("pilot_metrics_v1", "pilot_10_v1", "run-1")


def record(**overrides: object) -> PerformanceInput:
    values: dict[str, object] = {
        "business_date": date(2026, 8, 11),
        "bakery_id": 20,
        "product_id": 108,
        "forecast_run_id": "run-1",
        "forecast_qty": 18.0,
        "plan_qty": 18.0,
        "produced_qty": 6.0,
        "sold_qty": 6.0,
        "closing_stock_qty": 0.0,
        "last_sale_at": datetime(2026, 8, 11, 11),
        "opening_stock_qty": 0.0,
        "received_qty": 0.0,
        "sent_qty": 0.0,
        "transfers_complete": True,
        "available_to_sell_qty": 6.0,
        "available_to_sell_basis": "inventory_v1",
    }
    values.update(overrides)
    return PerformanceInput(**values)  # type: ignore[arg-type]


def test_complete_row_has_independent_statuses_and_lineage() -> None:
    row = build_performance_rows([record()], contract=CONTRACT, scope_bakery_ids={20})[
        0
    ]
    assert row.demand_is_complete and row.demand_qty == pytest.approx(18)
    assert row.eligible_forecast and row.forecast_status is ForecastStatus.NORMAL
    assert row.eligible_execution
    assert row.execution_status is ExecutionStatus.UNDERPRODUCTION
    assert row.eligible_availability
    assert row.availability_status is AvailabilityStatus.STOCKOUT
    assert row.eligible_lost_demand
    assert row.inventory_equation_version == "inventory_v1"
    assert row.metric_version == "pilot_metrics_v1"


def test_missing_last_sale_preserves_sales_but_blocks_demand_score() -> None:
    row = build_performance_rows([record(last_sale_at=None)], contract=CONTRACT)[0]
    assert row.observed_demand_floor_qty == 6
    assert row.demand_qty == 6
    assert not row.demand_is_complete
    assert not row.eligible_forecast
    assert row.forecast_status is ForecastStatus.NO_DATA
    assert not row.eligible_lost_demand
    assert row.eligible_execution
    assert row.execution_status is ExecutionStatus.UNDERPRODUCTION


def test_positive_reconciled_stock_makes_observed_demand_complete() -> None:
    row = build_performance_rows(
        [
            record(
                produced_qty=8, sold_qty=6, closing_stock_qty=2, available_to_sell_qty=8
            )
        ],
        contract=CONTRACT,
    )[0]
    assert row.demand_qty == 6
    assert row.demand_is_complete
    assert row.lost_demand_recognized_qty == 0
    assert row.eligible_forecast
    assert row.eligible_execution
    assert row.eligible_availability


def test_invalid_forecast_does_not_disable_execution() -> None:
    row = build_performance_rows([record(forecast_qty=None)], contract=CONTRACT)[0]
    assert not row.eligible_forecast
    assert DataQualityFlag.INVALID_FORECAST in row.forecast_flags
    assert row.eligible_execution
    assert row.eligible_lost_demand


def test_lineage_and_scope_flags_block_all_metric_families() -> None:
    row = build_performance_rows(
        [record(forecast_run_id="other", bakery_id=99)],
        contract=CONTRACT,
        scope_bakery_ids={20},
    )[0]
    assert row.blocking_flags == (
        DataQualityFlag.RUN_MISMATCH,
        DataQualityFlag.OUT_OF_SCOPE,
    )
    assert not any(
        (
            row.eligible_forecast,
            row.eligible_execution,
            row.eligible_lost_demand,
            row.eligible_availability,
        )
    )


def test_available_execution_requires_explicit_inventory_basis() -> None:
    contract = PerformanceContract(
        "v1", "scope", "run-1", execution_basis="available_to_sell"
    )
    row = build_performance_rows(
        [record(available_to_sell_basis=None)], contract=contract
    )[0]
    assert not row.eligible_execution
    assert DataQualityFlag.AVAILABLE_BASIS_MISSING in row.execution_flags


def test_incomplete_transfers_only_block_inventory_dependent_metrics() -> None:
    row = build_performance_rows(
        [record(transfers_complete=False, received_qty=None, sent_qty=None)],
        contract=CONTRACT,
    )[0]
    assert row.eligible_execution  # produced-basis execution needs no transfers
    assert not row.eligible_lost_demand
    assert not row.eligible_forecast
    assert not row.eligible_availability
    assert DataQualityFlag.TRANSFERS_INCOMPLETE in row.lost_demand_flags


def test_inventory_equation_mismatch_is_explicit() -> None:
    row = build_performance_rows([record(available_to_sell_qty=9)], contract=CONTRACT)[
        0
    ]
    assert DataQualityFlag.AVAILABLE_BALANCE_MISMATCH in row.dq_flags
    assert row.eligible_execution  # produced-basis execution remains valid
    assert not row.eligible_lost_demand


def test_duplicate_and_invalid_business_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_performance_rows([record(), record()], contract=CONTRACT)
    with pytest.raises(ValueError, match="positive"):
        build_performance_rows([record(product_id=0)], contract=CONTRACT)


def test_invalid_contract_lineage_is_rejected() -> None:
    with pytest.raises(ValueError, match="metric_version"):
        build_performance_rows([], contract=PerformanceContract("", "s", "r"))
