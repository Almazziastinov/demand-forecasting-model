from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiment_stockout_adjusted_sku_profiles import (
    align_variant_support,
    attach_targets_and_scopes,
    build_conservative_hourly_adjustments,
    normalize_to_bakery_prediction,
    summarize_variant,
)


def test_conservative_hourly_adjustment_preserves_shape_and_scales_total() -> None:
    demand = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "product_id": [10],
            "is_clear_stockout": [True],
            "demand_lower_bound": [4.0],
            "raw_imputed_demand": [10.0],
            "reference_days": [5],
        }
    )
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "hour": [18, 19],
            "imputed_demand": [1.0, 2.0],
        }
    )

    result = build_conservative_hourly_adjustments(demand, hourly)

    assert result["imputed_demand"].sum() == pytest.approx(2.0)
    assert result.set_index("hour").loc[19, "imputed_demand"] == pytest.approx(
        4.0 / 3.0
    )


def test_normalize_to_bakery_prediction_preserves_predicted_total() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "actual_qty": [4.0, 6.0],
            "predicted_qty": [3.0, 7.0],
        }
    )
    bakery = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "prediction": [12.0],
        }
    )

    result = normalize_to_bakery_prediction(scored, bakery)

    assert result["predicted_demand"].sum() == pytest.approx(12.0)
    assert result.set_index("product_id").loc[
        10, "predicted_demand"
    ] == pytest.approx(3.6)


def test_summary_uses_reconstructed_target_across_all_sku_days() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "observed_sales": [4.0, 6.0],
            "predicted_demand": [6.0, 6.0],
        }
    )
    targets = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "product_id": [10],
            "is_clear_stockout": [True],
            "imputed_demand": [3.0],
            "conservative_imputed": [2.0],
        }
    )
    attached = attach_targets_and_scopes(scored, targets, adjusted_pairs={(1, 10)})

    metrics = summarize_variant(attached, variant="v").set_index("scope")

    assert metrics.loc["all_sku_days_observed_sales", "bias_qty"] == 2.0
    assert metrics.loc["all_sku_days_conservative_demand", "bias_qty"] == 0.0
    assert metrics.loc["all_sku_days_full_reconstructed_demand", "bias_qty"] == -1.0


def test_align_variant_support_uses_union_and_zero_for_missing_prediction() -> None:
    frame_a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "observed_sales": [4.0, 0.0],
            "predicted_demand": [5.0, 2.0],
        }
    )
    frame_b = frame_a[frame_a["product_id"].eq(10)].copy()

    result = align_variant_support({"a": frame_a, "b": frame_b})

    assert len(result["a"]) == len(result["b"]) == 2
    assert result["b"].set_index("product_id").loc[20, "predicted_demand"] == 0.0
