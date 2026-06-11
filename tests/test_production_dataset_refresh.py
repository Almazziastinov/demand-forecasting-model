from __future__ import annotations

import pandas as pd

from pipelines.forecast_publish.production_dataset_refresh import (
    build_uplifted_daily_from_clickhouse_multipliers,
)
from pipelines.forecast_publish.production_dataset_refresh import (
    resolve_default_refresh_dates,
)


def test_resolve_default_refresh_dates_uses_moscow_business_day() -> None:
    dates = resolve_default_refresh_dates(
        pd.Timestamp("2026-06-10 22:30:00", tz="UTC"),
        timezone="Europe/Moscow",
    )

    assert dates.forecast_start_date == "2026-06-11"
    assert dates.history_end_date == "2026-06-10"


def test_build_uplifted_daily_from_clickhouse_multipliers() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-10", "2026-06-10"]),
            "bakery_id": [1, 2],
            "bakery_name": ["B1", "B2"],
            "city": ["Kazan", "Kazan"],
            "bakery_sales": [100.0, 80.0],
            "line_amount_sum": [1000.0, 800.0],
            "priced_quantity": [100.0, 80.0],
            "price_x_qty_sum": [1000.0, 800.0],
            "avg_price": [10.0, 10.0],
            "dow": [2, 2],
            "day": [10, 10],
            "month": [6, 6],
            "iso_week": [24, 24],
            "is_weekend": [0, 0],
            "is_month_start": [0, 0],
            "is_month_end": [0, 0],
            "is_payday_week": [0, 0],
        }
    )
    profile = pd.DataFrame(
        {
            "bakery_id": [1, 1, 2, 2],
            "dow": [2, 2, 1, 1],
            "hour": [8, 9, 8, 9],
            "mean_hour_share_norm": [0.25, 0.75, 0.50, 0.50],
        }
    )
    exact_multipliers = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "dow": [2, 2],
            "hour": [8, 9],
            "sku_uplift_multiplier": [2.0, 1.0],
        }
    )
    fallback_multipliers = pd.DataFrame(
        {
            "bakery_id": [2, 2],
            "hour": [8, 9],
            "sku_uplift_multiplier": [1.5, 1.0],
        }
    )

    uplifted, summary = build_uplifted_daily_from_clickhouse_multipliers(
        daily,
        profile,
        exact_multipliers,
        fallback_multipliers,
    )

    b1 = uplifted[uplifted["bakery_id"] == 1].iloc[0]
    b2 = uplifted[uplifted["bakery_id"] == 2].iloc[0]
    assert round(float(b1["bakery_sales_uplifted"]), 4) == 125.0
    assert round(float(b2["bakery_sales_uplifted"]), 4) == 100.0
    assert summary["uplift_source"] == "clickhouse_uplift_multipliers"
