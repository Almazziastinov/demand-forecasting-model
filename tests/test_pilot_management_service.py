import pandas as pd

from src.pilot_management_service import PilotManagementService


def test_kpis_use_raw_forecast_not_rounded_production_plan(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            {
                "forecast_qty": 100.0,
                "issued_plan_qty": 110.0,
                "issued_total_for_sale": 140.0,
                "issued_yesterday_stock": 30.0,
                "produced_qty": 110.0,
                "closing_stock_qty": 20.0,
                "eligible_forecast_summary": True,
                "eligible_lost_demand": False,
                "lost_demand_recognized_qty": 0.0,
                "sold_qty": 90.0,
            }
        ]
    )

    result = PilotManagementService(tmp_path)._kpi_block(detail)

    assert result["plan_qty"] == 100.0
    assert result["execution_rate"] == 1.1
    assert result["sold_qty"] == 90.0
    assert result["sellthrough_rate"] == 0.9


def test_apply_filters_uses_aligned_date_mask_after_start_filter(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            {"business_date": "2026-08-31", "sold_qty": 1.0},
            {"business_date": "2026-09-01", "sold_qty": 2.0},
            {"business_date": "2026-09-08", "sold_qty": 3.0},
            {"business_date": "2026-09-09", "sold_qty": 4.0},
        ]
    )

    result = PilotManagementService(tmp_path)._apply_filters(
        detail,
        date_from="2026-09-01",
        date_to="2026-09-08",
    )

    assert result["sold_qty"].tolist() == [2.0, 3.0]


def test_recognized_lost_is_capped_by_raw_forecast_gap(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            {
                "forecast_qty": 100.0,
                "issued_plan_qty": 120.0,
                "issued_total_for_sale": 140.0,
                "produced_qty": 90.0,
                "sold_qty": 80.0,
                "eligible_forecast_summary": True,
                "eligible_lost_demand": True,
                "lost_demand_recognized_qty": 50.0,
            }
        ]
    )

    result = PilotManagementService(tmp_path)._kpi_block(detail)

    assert result["recognized_lost_qty"] == 20.0
