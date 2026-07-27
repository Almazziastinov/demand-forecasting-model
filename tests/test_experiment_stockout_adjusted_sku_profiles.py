from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiment_stockout_adjusted_sku_profiles import (
    build_conservative_hourly_adjustments,
    normalize_to_bakery_prediction,
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
