from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_stockout_historical_shadow import (
    add_sales_ranks,
    build_case_replay,
    build_entity_stability,
    summarize_periods,
)


def _cases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-08", "2026-06-09"],
            "bakery_id": [1, 1, 1],
            "bakery_name": ["A", "A", "A"],
            "product_id": [10, 10, 20],
            "product_name": ["P", "P", "Q"],
            "daily_sold": [10.0, 10.0, 8.0],
            "forecast_qty": [7.0, 8.0, 5.0],
            "confirmed_model_shortfall_qty": [3.0, 2.0, 3.0],
            "robust_case_type": ["demand_loss", "demand_loss", "allocation"],
            "bakery_ratio": [0.7, 0.75, 1.0],
            "reference_days_actual_bakery": [3, 4, 4],
            "reference_days_sold": [3, 4, 4],
        }
    )


def _adjustments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-08"],
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "imputed_demand": [2.0, 2.0],
            "reference_days": [3, 4],
        }
    )


def test_case_replay_respects_cutoff_and_only_improves_demand_loss() -> None:
    result = build_case_replay(
        _cases(),
        _adjustments(),
        start=pd.Timestamp("2026-06-08"),
        end=pd.Timestamp("2026-06-09"),
    )
    assert result["date"].min() == pd.Timestamp("2026-06-08")
    assert result["imputed_demand"].tolist() == [2.0, 0.0]
    assert result["shortfall_reduction"].tolist() == [2.0, 0.0]
    assert not result["case_worsened"].any()


def test_case_replay_rejects_adjustment_for_allocation_case() -> None:
    invalid = _adjustments()
    invalid.loc[len(invalid)] = ["2026-06-09", 1, 20, 1.0, 4]
    with pytest.raises(ValueError, match="robust demand-loss"):
        build_case_replay(_cases(), invalid)


def test_period_summary_includes_days_without_cases() -> None:
    replay = build_case_replay(_cases(), _adjustments())
    result = summarize_periods(
        replay,
        start=pd.Timestamp("2026-06-01"),
        end=pd.Timestamp("2026-06-09"),
        frequency="D",
    )
    assert len(result) == 9
    assert (
        result.loc[result["date"].eq(pd.Timestamp("2026-06-02")), "cases"].item() == 0
    )


def test_entity_stability_and_sales_rank_include_top5_and_other_problem_sku() -> None:
    replay = build_case_replay(_cases(), _adjustments())
    stability = build_entity_stability(
        replay,
        keys=["bakery_id", "bakery_name", "product_id", "product_name"],
    )
    daily_sku = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-01"],
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "observed_sales": [100.0, 10.0],
        }
    )
    result = add_sales_ranks(
        stability,
        daily_sku,
        start=pd.Timestamp("2026-06-01"),
        end=pd.Timestamp("2026-06-09"),
    )
    recurrent = result.loc[result["product_id"].eq(10)].iloc[0]
    assert recurrent["recurrent_demand_loss"]
    assert recurrent["is_bakery_top5_by_sales"]
    assert recurrent["is_potentially_problematic"]
