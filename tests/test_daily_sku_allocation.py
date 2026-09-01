import pandas as pd
import pytest

from src.experiments_v2.daily_sku_allocation import (
    allocate_category_totals,
    build_daily_sku_shares,
)


def _history() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-07-01", periods=14):
        rows.extend(
            [
                {
                    "date": day,
                    "bakery_id": 1,
                    "city": "Казань",
                    "category": "Выпечка",
                    "product_id": 10,
                    "demand_mid": 30.0,
                },
                {
                    "date": day,
                    "bakery_id": 1,
                    "city": "Казань",
                    "category": "Выпечка",
                    "product_id": 20,
                    "demand_mid": 70.0,
                },
                {
                    "date": day,
                    "bakery_id": 2,
                    "city": "Казань",
                    "category": "Выпечка",
                    "product_id": 30,
                    "demand_mid": 20.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_daily_shares_cover_full_universe_and_sum_to_one() -> None:
    universe = pd.DataFrame(
        [
            {"bakery_id": 1, "city": "Казань", "category": "Выпечка", "product_id": 10},
            {"bakery_id": 1, "city": "Казань", "category": "Выпечка", "product_id": 20},
            {"bakery_id": 1, "city": "Казань", "category": "Выпечка", "product_id": 30},
        ]
    )
    shares = build_daily_sku_shares(_history(), universe, "2026-07-15")

    assert set(shares["product_id"]) == {10, 20, 30}
    assert shares["sku_share"].sum() == pytest.approx(1.0)
    assert shares.loc[shares["product_id"].eq(30), "sku_share"].iat[0] > 0.0


def test_future_history_is_rejected() -> None:
    universe = _history().tail(1)[["bakery_id", "city", "category", "product_id"]]
    with pytest.raises(ValueError, match="on or after forecast_date"):
        build_daily_sku_shares(_history(), universe, "2026-07-14")


def test_category_total_is_preserved() -> None:
    shares = pd.DataFrame(
        [
            {"bakery_id": 1, "city": "Казань", "category": "Выпечка", "product_id": 10, "sku_share": 0.25},
            {"bakery_id": 1, "city": "Казань", "category": "Выпечка", "product_id": 20, "sku_share": 0.75},
        ]
    )
    totals = pd.DataFrame([{"bakery_id": 1, "category": "Выпечка", "category_forecast": 200.0}])
    allocated = allocate_category_totals(totals, shares)

    assert allocated["sku_day_forecast"].sum() == pytest.approx(200.0)
