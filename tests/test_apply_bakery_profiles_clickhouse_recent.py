from __future__ import annotations

import pandas as pd

from src.experiments_v2.apply_bakery_profiles_clickhouse import (
    _build_recent_correction_targets,
)


def test_recent_correction_targets_filter_dead_sku_and_preserve_day_total() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02", "2026-05-02"]),
            "dow": [5, 5],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [60.0, 40.0],
            "source": ["exact", "exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": [10],
            "recent_qty": [100.0],
            "recent_days_sold": [10],
            "recent_share": [1.0],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="dead_0d",
    )

    live = targets[targets["product_id"] == 10].iloc[0]
    dead = targets[targets["product_id"] == 20].iloc[0]
    assert live["corrected_daily_forecast"] == 100.0
    assert dead["corrected_daily_forecast"] == 0.0
    assert targets["corrected_daily_forecast"].sum() == 100.0


def test_blend_recent_50_can_add_recent_sku_absent_from_profile() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02"]),
            "dow": [5],
            "bakery_id": [1],
            "hour": [9],
            "product_id": [10],
            "sku_hour_forecast": [100.0],
            "source": ["exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 30],
            "recent_qty": [50.0, 50.0],
            "recent_days_sold": [10, 10],
            "recent_share": [0.5, 0.5],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="blend_recent_50",
    )

    existing = targets[targets["product_id"] == 10].iloc[0]
    new = targets[targets["product_id"] == 30].iloc[0]
    assert existing["corrected_daily_forecast"] == 75.0
    assert new["corrected_daily_forecast"] == 25.0
    assert targets["corrected_daily_forecast"].sum() == 100.0
