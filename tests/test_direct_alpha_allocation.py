from __future__ import annotations

import pandas as pd
import pytest

from src.experiments_v2.direct_alpha_allocation import (
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01"] * 3),
            "bakery_id": [23, 23, 23],
            "product_id": [1071, 108, 999],
            "direct_p50": [200.0, 20.0, 80.0],
            "predictive_uplift": [0.0, 10.0, 20.0],
            "loss_scale": [1.0, 1.0, 1.0],
            "broad_56_mean": [200.0, 20.0, 80.0],
            "floor_history_n": [10, 10, 10],
            "floor_demand_p67": [210.0, 30.0, 100.0],
            "historical_stockout_rate": [0.1, 0.8, 0.8],
            "historical_lost_mean": [0.0, 5.0, 5.0],
        }
    )


def test_selected_plan_preserves_soft_alpha_target_before_floor() -> None:
    result = build_selected_direct_plan(sample_rows())
    expected = 300.0 + 0.25 * 30.0
    assert result["direct_alpha"].sum() == pytest.approx(expected)
    core = result.loc[result["product_id"].eq(1071)].iloc[0]
    assert core["is_core_sku"]
    assert core["direct_alpha"] >= core["direct_p50"]


def test_floor_is_bounded_and_tail_cap_protects_dominant_sku() -> None:
    rows = sample_rows()
    rows.loc[rows["product_id"].eq(1071), "direct_p50"] = 400.0
    rows.loc[rows["product_id"].eq(1071), "floor_demand_p67"] = 220.0
    result = build_selected_direct_plan(rows)
    dominant = result.loc[result["product_id"].eq(1071)].iloc[0]
    assert dominant["tail_cap_applied"]
    assert dominant["selected_sku_forecast"] == pytest.approx(220.0)
    floored = result.loc[result["product_id"].eq(108)].iloc[0]
    assert floored["direct_alpha_floor"] <= floored["direct_alpha"] + 5.0
    assert floored["direct_alpha_floor"] <= floored["direct_alpha"] * 1.10


def test_invalid_alpha_is_rejected() -> None:
    with pytest.raises(ValueError, match="alpha"):
        build_selected_direct_plan(
            sample_rows(), DirectAlphaAllocationConfig(alpha=1.1)
        )
