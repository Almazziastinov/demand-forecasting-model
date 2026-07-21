from __future__ import annotations

import pandas as pd

from scripts.run_stockout_direction_combined_replay import build_replay


def test_build_replay_adds_demand_only_to_confirmed_case() -> None:
    cases = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "bakery_id": [1],
            "product_id": [10],
            "bakery_forecast_qty": [100.0],
            "forecast_qty": [8.0],
        }
    )
    adjustments = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "bakery_id": [1],
            "product_id": [10],
            "imputed_demand": [3.0],
        }
    )
    dynamic = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "bakery_id": [1],
            "product_id": [10],
            "adjusted_forecast_qty": [9.0],
        }
    )
    regime = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "bakery_id": [1],
            "product_id": [10],
            "adjusted_forecast_qty": [10.0],
        }
    )
    shares = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": [10],
            "replay_dow": [1],
            "current_share": [0.1],
        }
    )
    result = build_replay(cases, adjustments, dynamic, regime, shares)
    assert result.iloc[0]["demand_only_forecast"] == 11.0
    assert result.iloc[0]["current_profile_plus_demand"] == 13.0
    assert result.iloc[0]["dynamic_plus_demand"] == 12.0
    assert result.iloc[0]["regime_plus_demand"] == 13.0
