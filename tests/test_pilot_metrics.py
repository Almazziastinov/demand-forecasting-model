from datetime import datetime

import pytest

from src.pilot_metrics import (
    AvailabilityStatus,
    ExecutionStatus,
    ForecastStatus,
    aggregate_forecast_metrics,
    calculate_lost_demand,
    classify_availability,
    classify_execution,
    classify_forecast,
    estimated_demand,
)


def test_lost_demand_separates_raw_and_recognized() -> None:
    result = calculate_lost_demand(
        sold_qty=6,
        produced_qty=6,
        closing_stock_qty=0,
        last_sale_at=datetime(2026, 8, 11, 11),
    )
    assert result.policy_version == "sales_rate_v1"
    assert result.raw_qty == pytest.approx(12.0)
    assert result.recognized_qty == pytest.approx(12.0)
    assert result.eligible

    with_stock = calculate_lost_demand(
        sold_qty=5,
        produced_qty=6,
        closing_stock_qty=1,
        last_sale_at=datetime(2026, 8, 11, 11),
    )
    assert with_stock.raw_qty == 0
    assert with_stock.recognized_qty == 0
    assert with_stock.eligible


def test_zero_sales_and_close_of_day_are_complete_zero_lost() -> None:
    zero = calculate_lost_demand(
        sold_qty=0, produced_qty=5, closing_stock_qty=5, last_sale_at=None
    )
    assert zero.eligible and zero.rejection_reason is None
    closed = calculate_lost_demand(
        sold_qty=5,
        produced_qty=5,
        closing_stock_qty=0,
        last_sale_at=datetime(2026, 8, 11, 19),
    )
    assert closed.eligible and closed.recognized_qty == 0


def test_lost_demand_policy_has_elapsed_gate_and_cap() -> None:
    too_early = calculate_lost_demand(
        sold_qty=10,
        produced_qty=10,
        closing_stock_qty=0,
        last_sale_at=datetime(2026, 8, 11, 8, 30),
    )
    assert too_early.raw_qty == 0
    assert too_early.rejection_reason == "insufficient_elapsed_time"

    capped = calculate_lost_demand(
        sold_qty=1,
        produced_qty=1,
        closing_stock_qty=0,
        last_sale_at=datetime(2026, 8, 11, 9),
    )
    assert capped.raw_qty == 5


def test_demand_and_forecast_status_boundaries() -> None:
    assert estimated_demand(10, 2.5) == 12.5
    assert estimated_demand(10, None) is None
    assert classify_forecast(12, 10) is ForecastStatus.NORMAL
    assert classify_forecast(8, 10) is ForecastStatus.NORMAL
    assert classify_forecast(12.01, 10) is ForecastStatus.OVERFORECAST
    assert classify_forecast(7.99, 10) is ForecastStatus.UNDERFORECAST
    assert classify_forecast(None, 10) is ForecastStatus.NO_DATA


def test_execution_uses_plan_and_optional_available_quantity() -> None:
    assert classify_execution(plan_qty=10, produced_qty=8) is ExecutionStatus.NORMAL
    assert (
        classify_execution(plan_qty=10, produced_qty=7.9)
        is ExecutionStatus.UNDERPRODUCTION
    )
    assert (
        classify_execution(plan_qty=10, produced_qty=12.1)
        is ExecutionStatus.OVERPRODUCTION
    )
    assert (
        classify_execution(plan_qty=10, produced_qty=4, available_to_sell_qty=10)
        is ExecutionStatus.NORMAL
    )
    assert classify_execution(plan_qty=None, produced_qty=10) is ExecutionStatus.NO_DATA


def test_availability_is_independent_from_execution() -> None:
    assert (
        classify_availability(sold_qty=10, closing_stock_qty=0, recognized_lost_qty=2)
        is AvailabilityStatus.STOCKOUT
    )
    assert (
        classify_availability(
            sold_qty=10, closing_stock_qty=2.01, recognized_lost_qty=0
        )
        is AvailabilityStatus.OVERSTOCK
    )
    assert (
        classify_availability(sold_qty=10, closing_stock_qty=2, recognized_lost_qty=0)
        is AvailabilityStatus.NORMAL
    )
    assert (
        classify_availability(sold_qty=None, closing_stock_qty=0, recognized_lost_qty=0)
        is AvailabilityStatus.NO_DATA
    )


def test_aggregate_metrics_excludes_no_data_rows() -> None:
    result = aggregate_forecast_metrics([(12, 10), (8, 10), (None, 5)])
    assert result.row_count == 3
    assert result.valid_row_count == 2
    assert result.forecast_qty == 20
    assert result.demand_qty == 20
    assert result.error_qty == 0
    assert result.absolute_error_qty == 4
    assert result.underforecast_qty == 2
    assert result.overforecast_qty == 2
    assert result.wape == pytest.approx(0.2)
    assert result.bias == 0


def test_aggregate_zero_demand_has_undefined_rates() -> None:
    result = aggregate_forecast_metrics([(0, 0)])
    assert result.wape is None
    assert result.bias is None
