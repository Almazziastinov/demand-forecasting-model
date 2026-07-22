from __future__ import annotations

import pandas as pd

from scripts.build_stockout_adjusted_demand_dataset import build_demand_dataset
from scripts.build_stockout_adjusted_demand_dataset import build_cap_sensitivity
from scripts.build_stockout_adjusted_demand_dataset import summarize_dataset


def test_build_demand_dataset_keeps_observed_and_censored_targets_separate() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "product_name": ["A", "B"],
            "hour": [12, 12],
            "sold": [4.0, 6.0],
        }
    )
    audit = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "product_id": [10],
            "last_sale_hour": [12.0],
            "reference_days": [5],
            "raw_imputed_demand": [3.0],
            "case_cap": [4.0],
            "imputed_demand": [3.0],
        }
    )
    stockouts = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "product_id": [10],
            "qty_produced": [4.0],
        }
    )

    result = build_demand_dataset(hourly, audit, stockouts).set_index("product_id")

    assert result.loc[10, "demand_lower_bound"] == 4.0
    assert result.loc[10, "demand_point_estimate"] == 7.0
    assert result.loc[10, "reconstruction_confidence"] == "high"
    assert result.loc[10, "suggested_training_weight"] == 0.8
    assert not result.loc[10, "is_case_cap_binding"]
    assert result.loc[20, "demand_point_estimate"] == 6.0
    assert result.loc[20, "target_source"] == "observed_sales"


def test_summary_has_no_allocation_assumption_and_checks_key_quality() -> None:
    hourly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "product_id": [10],
            "product_name": ["A"],
            "hour": [12],
            "sold": [4.0],
        }
    )
    audit = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "product_id": [10],
            "last_sale_hour": [12.0],
            "reference_days": [2],
            "raw_imputed_demand": [0.0],
            "case_cap": [3.0],
            "imputed_demand": [0.0],
        }
    )
    stockouts = audit[["date", "bakery_id", "product_id"]]
    dataset = build_demand_dataset(hourly, audit, stockouts)

    summary = summarize_dataset(dataset)

    assert summary["contains_allocation_assumption"] is False
    assert summary["unadjusted_censored_rows"] == 1
    assert summary["quality"]["duplicate_key_rows"] == 0


def test_cap_sensitivity_is_monotonic_in_ratio() -> None:
    audit = pd.DataFrame(
        {
            "daily_sold_observed": [4.0],
            "raw_imputed_demand": [10.0],
            "reference_days": [5],
        }
    )

    result = build_cap_sensitivity(audit)
    cap_20 = result[result["max_case_uplift_units"].eq(20.0)].set_index(
        "max_case_uplift_ratio"
    )

    assert cap_20.loc[0.5, "imputed_demand_units"] == 2.0
    assert cap_20.loc[1.0, "imputed_demand_units"] == 4.0
