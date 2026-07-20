from __future__ import annotations

import pandas as pd

from scripts.experiment_non_stockout_share_allocation import (
    allocate_from_non_stockout_shares,
)


def test_allocator_preserves_bakery_day_total_and_uses_only_prior_normal_days() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-06-01", "2026-06-01", "2026-06-08", "2026-06-08"]
            ),
            "bakery_id": [1] * 4,
            "product_id": [10, 20, 10, 20],
            "dow": [0] * 4,
            "forecast_qty": [50.0, 50.0, 50.0, 50.0],
            "daily_sold": [80.0, 20.0, 80.0, 20.0],
            "bakery_actual_qty": [100.0] * 4,
            "stockout_group": ["confirmed_non_stockout"] * 4,
        }
    )

    result = allocate_from_non_stockout_shares(
        frame,
        lookback_days=28,
        min_history_days=1,
        prior_days=1.0,
        use_weekday=True,
    )

    totals = result.groupby("date")["adjusted_forecast_qty"].sum()
    assert totals.tolist() == [100.0, 100.0]
    assert result.loc[0, "profile_days"] == 0
    assert result.loc[2, "profile_days"] == 1
    assert result.loc[2, "adjusted_forecast_qty"] > 50.0
