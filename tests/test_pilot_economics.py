import pandas as pd
import pytest

from src.pilot_economics import (
    rank_economic_actions,
    simulate_partner_economics,
    summarize_partner_economics,
)


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "business_date": "2026-09-01",
                "forecast_run_id": "prod_direct_alpha_025_20260901_h14",
                "bakery_id": 1,
                "product_id": 10,
                "product_name": "SKU",
                "fact_category_name": "Выпечка",
                "demand_qty": 10,
                "produced_qty": 6,
                "forecast_qty": 10,
                "issued_yesterday_stock": 0,
                "received_qty": 0,
                "sent_qty": 0,
                "unit_price": 100,
                "unit_cost": 40,
                "eligible_lost_demand": True,
            },
            {
                "business_date": "2026-09-02",
                "forecast_run_id": "prod_direct_alpha_025_20260902_h14",
                "bakery_id": 1,
                "product_id": 10,
                "product_name": "SKU",
                "fact_category_name": "Выпечка",
                "demand_qty": 8,
                "produced_qty": 12,
                "forecast_qty": 8,
                "issued_yesterday_stock": 0,
                "received_qty": 0,
                "sent_qty": 0,
                "unit_price": 100,
                "unit_cost": 40,
                "eligible_lost_demand": True,
            },
        ]
    )


def test_two_day_economics_compares_scenarios_on_same_demand() -> None:
    simulated, coverage = simulate_partner_economics(rows())
    summary = summarize_partner_economics(simulated)
    assert coverage["coverage_ratio"] == pytest.approx(1)
    assert summary["actual_gross_profit"] == pytest.approx(680)
    assert summary["ai_gross_profit"] == pytest.approx(1080)
    assert summary["profit_delta"] == pytest.approx(400)
    assert summary["actual_underproduction_qty"] == pytest.approx(4)


def test_economics_excludes_rows_without_cost() -> None:
    detail = rows()
    detail.loc[0, "unit_cost"] = None
    _, coverage = simulate_partner_economics(detail)
    assert coverage["eligible_rows"] == 1
    assert coverage["coverage_ratio"] == pytest.approx(8 / 18)


def test_action_ranking_explains_direction() -> None:
    simulated, _ = simulate_partner_economics(rows())
    actions = rank_economic_actions(simulated)
    assert actions[0]["product_name"] == "SKU"
    assert actions[0]["profit_delta"] == pytest.approx(400)


def test_old_model_rows_are_not_presented_as_direct_economics() -> None:
    detail = rows()
    detail["forecast_run_id"] = "prod_base_bakery_norm_recent_20260817_h14"
    simulated, coverage = simulate_partner_economics(detail)
    assert simulated.empty
    assert coverage["coverage_ratio"] == 0
