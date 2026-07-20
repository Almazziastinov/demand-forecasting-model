from __future__ import annotations

import pandas as pd

from scripts.analyze_stockout_allocation_failures import (
    assign_pipeline_regime,
    build_sku_summary,
    enrich_cases,
)


def test_enrichment_separates_total_volume_from_sku_allocation() -> None:
    cases = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "product_id": [10],
            "product_name": ["SKU"],
            "category_name": ["Category"],
            "daily_sold": [20.0],
            "forecast_qty": [10.0],
            "confirmed_model_shortfall_qty": [10.0],
        }
    )
    bakery = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "bakery_actual_qty": [100.0],
            "bakery_forecast_qty": [110.0],
        }
    )
    totals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "sku_forecast_total_qty": [100.0],
        }
    )

    result = enrich_cases(cases, bakery, totals)

    assert result.loc[0, "bakery_volume_sufficient"]
    assert result.loc[0, "allocation_share_ratio"] == 0.5
    assert result.loc[0, "diagnosis"] == "allocation_failure_likely"


def test_sku_summary_marks_repeated_cross_bakery_pattern_as_systematic() -> None:
    cases = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "bakery_id": [1, 1, 2],
            "product_id": [10, 10, 10],
            "product_name": ["SKU"] * 3,
            "category_name": ["Category"] * 3,
            "confirmed_model_shortfall_qty": [1.0, 2.0, 3.0],
            "bakery_volume_sufficient": [True, True, False],
        }
    )
    all_rows = cases.copy()

    summary = build_sku_summary(cases, all_rows)

    assert summary.loc[0, "underforecast_stockouts"] == 3
    assert summary.loc[0, "bakeries"] == 2
    assert summary.loc[0, "systematic"]


def test_pipeline_regime_separates_late_processing_versions() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-06-30", "2026-07-10", "2026-07-16"],
            "source_run_id": ["no_sku_uplift", "raw_uplift", "raw_uplift"],
        }
    )

    assert assign_pipeline_regime(frame).tolist() == [
        "base_no_sku_uplift",
        "raw_uplift_pre_cap_haircut",
        "current_cap_haircut_stockout",
    ]
