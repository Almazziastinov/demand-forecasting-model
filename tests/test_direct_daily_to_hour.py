from __future__ import annotations

import pandas as pd
import pytest

from pipelines.forecast_publish.direct_daily_to_hour import (
    expand_direct_sku_day_to_hour,
)


def test_hour_expansion_preserves_each_sku_day() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01", "2026-09-01"]),
            "bakery_id": [23, 23],
            "product_id": [108, 1071],
            "sku_day_forecast": [27.5, 235.0],
        }
    )
    profile = pd.DataFrame(
        {
            "bakery_id": [23, 23],
            "dow": [1, 1],
            "hour": [8, 9],
            "mean_hour_share_norm": [2.0, 3.0],
        }
    )
    hourly = expand_direct_sku_day_to_hour(daily, profile)
    totals = hourly.groupby(["date", "bakery_id", "product_id"])[
        "sku_hour_forecast"
    ].sum()
    assert totals.loc[(pd.Timestamp("2026-09-01"), 23, 108)] == pytest.approx(27.5)
    assert totals.loc[(pd.Timestamp("2026-09-01"), 23, 1071)] == pytest.approx(235.0)


def test_hour_expansion_rejects_missing_bakery_profile() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01"]),
            "bakery_id": [23],
            "product_id": [108],
            "sku_day_forecast": [27.5],
        }
    )
    profile = pd.DataFrame(
        {
            "bakery_id": [29],
            "dow": [1],
            "hour": [8],
            "mean_hour_share_norm": [1.0],
        }
    )
    with pytest.raises(ValueError, match="23"):
        expand_direct_sku_day_to_hour(daily, profile)
