"""Tests for SKU hour-share profile builder."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.build_sku_hour_share_profile import aggregate_sku_hourly_chunk  # noqa: E402
from src.experiments_v2.build_sku_hour_share_profile import build_sku_hour_share_profile  # noqa: E402


def _hourly() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "dow": 0, "hour": 8, "bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P1", "product_name": "Product 1", "category_name": "Cat", "sku_hour_sales": 2.0},
            {"date": pd.Timestamp("2026-01-05"), "dow": 0, "hour": 8, "bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P2", "product_name": "Product 2", "category_name": "Cat", "sku_hour_sales": 6.0},
            {"date": pd.Timestamp("2026-01-12"), "dow": 0, "hour": 8, "bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P1", "product_name": "Product 1", "category_name": "Cat", "sku_hour_sales": 1.0},
            {"date": pd.Timestamp("2026-01-12"), "dow": 0, "hour": 8, "bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P2", "product_name": "Product 2", "category_name": "Cat", "sku_hour_sales": 3.0},
        ]
    )


def test_build_sku_hour_share_profile_normalizes_per_bakery_hour():
    profile, applied = build_sku_hour_share_profile(_hourly())
    sums = profile.groupby(["bakery_id", "dow", "hour"])["mean_sku_share_in_hour_norm"].sum()
    assert float(sums.iloc[0]) == 1.0
    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    p2 = profile[profile["product_id"] == "P2"].iloc[0]
    assert round(float(p1["mean_sku_share_in_hour_norm"]), 4) == 0.25
    assert round(float(p2["mean_sku_share_in_hour_norm"]), 4) == 0.75


def test_aggregate_sku_hourly_chunk_supports_legacy_russian_snapshot_columns():
    raw = pd.DataFrame(
        [
            {
                "Дата продажи": "01.01.2026",
                "Дата время чека": "01.01.2026 14:21:01",
                "Вид события по кассе": "Продажа",
                "Касса.Торговая точка": "Bakery Legacy",
                "Номенклатура": "Product Legacy",
                "Категория": "Cat Legacy",
                "Кол-во": 2.0,
            }
        ]
    )
    hourly = aggregate_sku_hourly_chunk(raw)
    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["bakery_id"] == "Bakery Legacy"
    assert row["product_id"] == "Product Legacy"
    assert row["hour"] == 14
    assert row["sku_hour_sales"] == 2.0
