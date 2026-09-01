from datetime import datetime

import pandas as pd
import pytest

from scripts.build_pilot_management_summary import (
    build_historical_scope,
    build_coverage_diagnostics,
    build_detail,
    build_execution_triage,
    build_kpis,
    build_rankings,
    build_priority_queues,
    select_coherent_lead1,
)
from src.pilot_config_service import PilotConfigService


def forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2026-08-10", 20, 1, "old", "2026-08-09 06:00", 5],
            ["2026-08-10", 20, 1, "new", "2026-08-09 07:00", 10],
            ["2026-08-10", 20, 2, "new", "2026-08-09 07:00", 10],
        ],
        columns=[
            "date",
            "bakery_id",
            "product_id",
            "source_run_id",
            "generated_at",
            "forecast_qty",
        ],
    )


def facts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2026-08-10", 20, 1, 10, 10, 0, 0, 0, datetime(2026, 8, 10, 18)],
            ["2026-08-10", 20, 2, 10, 10, 0, 0, 0, datetime(2026, 8, 10, 18)],
        ],
        columns=[
            "date",
            "bakery_id",
            "product_id",
            "qty_sold",
            "qty_produced",
            "qty_received",
            "qty_sent",
            "stock_balance",
            "last_sale_time",
        ],
    )


def test_selects_one_latest_coherent_run() -> None:
    selected, excluded = select_coherent_lead1(forecasts(), scope_bakery_ids=(20,))
    assert excluded.empty
    assert set(selected["source_run_id"]) == {"new"}
    assert len(selected) == 2


def test_selects_older_horizon_when_lead1_run_is_missing() -> None:
    frame = forecasts().copy()
    frame["lead_days"] = [3, 2, 2]

    selected, excluded = select_coherent_lead1(frame, scope_bakery_ids=(20,))

    assert excluded.empty
    assert set(selected["source_run_id"]) == {"new"}
    assert set(selected["lead_days"]) == {2}


def test_dynamic_scope_returns_seed_plus_added_bakery(monkeypatch) -> None:
    service = object.__new__(PilotConfigService)
    monkeypatch.setattr(
        service,
        "get_all_bakeries",
        lambda scope_name: [
            {"bakery_id": bakery_id, "is_active": True}
            for bakery_id in range(1, 39)
        ]
        + [{"bakery_id": 273, "is_active": True}],
    )

    bakery_ids = service.get_active_bakery_ids("expanded_pilot_38")

    assert len(bakery_ids) == 39
    assert bakery_ids[-1] == 273


def test_historical_scope_applies_add_from_event_date() -> None:
    scopes = build_historical_scope(
        pd.Timestamp("2026-07-23").date(),
        pd.Timestamp("2026-08-18").date(),
        [
            {
                "bakery_id": 273,
                "action": "add",
                "changed_at": "2026-08-17T14:44:15",
            }
        ],
    )

    assert len(scopes[pd.Timestamp("2026-07-23").date()]) == 38
    assert 273 not in scopes[pd.Timestamp("2026-08-16").date()]
    assert 273 in scopes[pd.Timestamp("2026-08-17").date()]


def test_build_detail_uses_selected_scope_version() -> None:
    detail, _ = build_detail(
        forecasts(),
        facts(),
        scope_bakery_ids=(20,),
        scope_version="expanded_pilot_38_events_v1",
    )

    assert set(detail["scope_version"]) == {"expanded_pilot_38_events_v1"}


def test_tied_latest_runs_are_excluded() -> None:
    frame = forecasts().iloc[[1, 2]].copy()
    frame.loc[frame.index[1], "source_run_id"] = "other"
    selected, excluded = select_coherent_lead1(frame, scope_bakery_ids=(20,))
    assert selected.empty
    assert excluded.iloc[0]["reason"] == "ambiguous_full_run"


def test_partial_scope_run_keeps_available_bakery_data() -> None:
    frame = forecasts().copy()

    selected, excluded = select_coherent_lead1(
        frame, scope_bakery_ids=(20, 270)
    )

    assert set(selected["source_run_id"]) == {"new"}
    assert excluded.iloc[0]["reason"] == "partial_scope_run"


def test_summary_has_forecast_kpi_but_no_execution_kpi() -> None:
    detail, exclusions = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    assert exclusions.empty
    company, bakery, week = build_kpis(detail)
    assert len(company) == len(bakery) == len(week) == 1
    assert company.iloc[0]["forecast_coverage"] == 1
    assert company.iloc[0]["execution_kpi_included"] == False  # noqa: E712
    assert pd.isna(company.iloc[0]["execution_wape"])


def test_forecast_outside_fact_universe_is_an_explicit_extra() -> None:
    detail, excluded = build_detail(
        forecasts(), facts().iloc[[0]], scope_bakery_ids=(20,)
    )
    assert len(detail) == 1
    assert "forecast_outside_fact_universe" in set(excluded["reason"])


def test_fact_without_forecast_remains_in_universe_and_reduces_coverage() -> None:
    fact_frame = pd.concat(
        [
            facts(),
            pd.DataFrame(
                [["2026-08-10", 20, 3, 4, 4, 0, 0, 0, datetime(2026, 8, 10, 18)]],
                columns=facts().columns,
            ),
        ],
        ignore_index=True,
    )
    detail, _ = build_detail(forecasts(), fact_frame, scope_bakery_ids=(20,))
    company, _, _ = build_kpis(detail)
    assert len(detail) == 3
    assert detail["forecast_qty"].isna().sum() == 1
    assert company.iloc[0]["missing_forecast_rows"] == 1


def test_close_of_day_sales_are_scorable_without_lost_demand() -> None:
    detail, _ = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    assert detail["eligible_forecast_summary"].all()
    diagnostics = build_coverage_diagnostics(detail)
    assert diagnostics[diagnostics["diagnostic"].eq("forecast_ineligible")].empty


def test_rankings_expose_contribution_and_rule_based_priority() -> None:
    detail, _ = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    bakery, sku, bakery_sku, daily = build_rankings(detail)
    assert "contribution_to_total_abs_error" in bakery
    assert {"priority_tier", "priority_flags", "abs_error_qty"}.issubset(sku.columns)
    assert not bakery_sku.empty
    assert len(daily) == 1


def test_priority_queues_enforce_gates_and_contributions() -> None:
    detail, _ = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    model, execution = build_priority_queues(detail)
    assert model["contribution_to_total_abs_error"].sum() == pytest.approx(1.0)
    assert model[model["priority_tier"].eq("M1")]["gates_passed"].all()
    if not execution.empty:
        assert execution["contribution_to_total_abs_error"].sum() == pytest.approx(1.0)
        assert execution[execution["priority_tier"].eq("E1")]["gates_passed"].all()


def test_execution_queue_has_transparent_responsibility_metrics() -> None:
    detail, _ = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    detail["plan_format_version"] = "stock_aware_v2"
    detail["plan_qty"] = 10.0
    detail["eligible_execution"] = True
    _, execution = build_priority_queues(detail)
    required = {
        "plan_fulfillment_ratio",
        "execution_bias",
        "execution_wape",
        "execution_mae_qty",
        "normal_cases",
        "under_cases",
        "over_cases",
        "forecast_complete_demand_wape",
        "forecast_complete_demand_bias",
        "forecast_mae_qty",
        "forecast_normal_case_share",
        "forecast_reliability",
        "deprecated_responsibility_diagnostic",
        "closer_to_demand_than_plan_count",
        "closer_to_demand_than_plan_share",
    }
    assert required.issubset(execution.columns)
    row = execution.iloc[0]
    assert (
        row["normal_cases"] + row["under_cases"] + row["over_cases"] == row["plan_days"]
    )
    assert row["forecast_reliability"] == "insufficient_data"
    assert row["deprecated_responsibility_diagnostic"] == "insufficient_data"


def test_execution_twenty_percent_boundary_is_normal() -> None:
    detail, _ = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    detail["plan_format_version"] = "stock_aware_v2"
    detail["plan_qty"] = 10.0
    detail["produced_qty"] = 8.0
    detail["eligible_execution"] = True
    _, execution = build_priority_queues(detail)
    assert execution["normal_cases"].sum() == len(detail)


def test_triage_requires_comparable_stock_basis_for_mitigation() -> None:
    detail, _ = build_detail(forecasts(), facts(), scope_bakery_ids=(20,))
    detail["plan_format_version"] = "stock_aware_v2"
    detail["plan_qty"] = 10.0
    detail["eligible_execution"] = True
    triage = build_execution_triage(detail)
    assert set(triage["mitigation_signal"]) == {"unavailable"}
    assert not triage["triage"].eq("likely_execution").any()
