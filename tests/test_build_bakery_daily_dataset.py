"""Tests for bakery-level daily dataset builder."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.build_bakery_daily_dataset import (  # noqa: E402
    add_calendar_features,
    add_lag_features,
    aggregate_chunk,
    merge_partial_results,
)


def _raw_chunk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_date": "2026-01-05",
                "cash_event_type": "Продажа",
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
            },
            {
                "check_date": "2026-01-05",
                "cash_event_type": "Продажа",
                "quantity": 3.0,
                "price": 100.0,
                "line_amount": 300.0,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
            },
            {
                "check_date": "2026-01-06",
                "cash_event_type": "Возврат",
                "quantity": 5.0,
                "price": 90.0,
                "line_amount": 450.0,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
            },
        ]
    )


def test_aggregate_chunk_keeps_only_sales():
    part = aggregate_chunk(_raw_chunk())
    assert len(part) == 1
    row = part.iloc[0]
    assert row["bakery_sales"] == 5.0
    assert row["line_amount_sum"] == 500.0


def test_add_lag_features_builds_bakery_lags():
    daily = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-01"), "bakery_id": "B1", "bakery_name": "Bakery 1", "city": "Kazan", "bakery_sales": 1.0, "line_amount_sum": 100.0, "priced_quantity": 1.0, "price_x_qty_sum": 100.0},
            {"date": pd.Timestamp("2026-01-02"), "bakery_id": "B1", "bakery_name": "Bakery 1", "city": "Kazan", "bakery_sales": 2.0, "line_amount_sum": 200.0, "priced_quantity": 2.0, "price_x_qty_sum": 200.0},
            {"date": pd.Timestamp("2026-01-03"), "bakery_id": "B1", "bakery_name": "Bakery 1", "city": "Kazan", "bakery_sales": 3.0, "line_amount_sum": 300.0, "priced_quantity": 3.0, "price_x_qty_sum": 300.0},
        ]
    )
    daily = add_calendar_features(daily)
    daily = add_lag_features(daily)
    assert daily.loc[daily["date"] == pd.Timestamp("2026-01-02"), "bakery_sales_lag1"].iloc[0] == 1.0


def test_aggregate_chunk_supports_legacy_russian_snapshot_columns():
    raw = pd.DataFrame(
        [
            {
                "Дата продажи": "01.01.2026",
                "Вид события по кассе": "Продажа",
                "Касса.Торговая точка": "Bakery Legacy",
                "Цена": 120.0,
                "Кол-во": 2.0,
            }
        ]
    )
    part = aggregate_chunk(raw)
    assert len(part) == 1
    row = part.iloc[0]
    assert row["bakery_id"] == "Bakery Legacy"
    assert row["city"] == "unknown"
    assert row["line_amount_sum"] == 240.0
