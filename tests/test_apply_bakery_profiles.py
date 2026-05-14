"""Tests for bakery-driven allocation layer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.apply_bakery_profiles import allocate_bakery_to_hour  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import allocate_hour_to_sku  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import apply_profiles  # noqa: E402


def test_allocate_bakery_profiles_preserves_bakery_total():
    bakery_forecast = pd.DataFrame(
        [
            {"date": "2026-01-05", "dow": 0, "bakery_id": "B1", "bakery_name": "Bakery 1", "bakery_day_forecast": 100.0}
        ]
    )
    bakery_profile = pd.DataFrame(
        [
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "dow": 0, "hour": 8, "mean_hour_share_norm": 0.25},
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "dow": 0, "hour": 9, "mean_hour_share_norm": 0.75},
        ]
    )
    sku_profile = pd.DataFrame(
        [
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P1", "product_name": "P1", "category_name": "Cat", "dow": 0, "hour": 8, "mean_sku_share_in_hour_norm": 0.4},
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P2", "product_name": "P2", "category_name": "Cat", "dow": 0, "hour": 8, "mean_sku_share_in_hour_norm": 0.6},
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P1", "product_name": "P1", "category_name": "Cat", "dow": 0, "hour": 9, "mean_sku_share_in_hour_norm": 0.5},
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P2", "product_name": "P2", "category_name": "Cat", "dow": 0, "hour": 9, "mean_sku_share_in_hour_norm": 0.5},
        ]
    )

    hourly = allocate_bakery_to_hour(bakery_forecast, bakery_profile)
    sku_hourly = allocate_hour_to_sku(hourly, sku_profile)

    assert round(float(hourly["bakery_hour_forecast"].sum()), 4) == 100.0
    assert round(float(sku_hourly["sku_hour_forecast"].sum()), 4) == 100.0

    p1_day = sku_hourly[sku_hourly["product_id"] == "P1"]["sku_hour_forecast"].sum()
    p2_day = sku_hourly[sku_hourly["product_id"] == "P2"]["sku_hour_forecast"].sum()
    assert round(float(p1_day), 4) == 47.5
    assert round(float(p2_day), 4) == 52.5


def test_apply_profiles_accepts_bakery_sales_as_fallback_forecast_column():
    tmp_path = Path("tests") / "_tmp_apply_profiles" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)

    bakery_forecast_path = tmp_path / "bakery_daily_sales.csv"
    bakery_hour_profile_path = tmp_path / "bakery_hour_profile.csv"
    sku_hour_profile_path = tmp_path / "sku_hour_share_profile_smoothed.csv"

    pd.DataFrame(
        [
            {"date": "2026-01-05", "bakery_id": "B1", "bakery_name": "Bakery 1", "bakery_sales": 80.0}
        ]
    ).to_csv(bakery_forecast_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "dow": 0, "hour": 8, "mean_hour_share_norm": 1.0}
        ]
    ).to_csv(bakery_hour_profile_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P1", "product_name": "P1", "category_name": "Cat", "dow": 0, "hour": 8, "mean_sku_share_in_hour_norm": 0.25},
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "product_id": "P2", "product_name": "P2", "category_name": "Cat", "dow": 0, "hour": 8, "mean_sku_share_in_hour_norm": 0.75},
        ]
    ).to_csv(sku_hour_profile_path, index=False, encoding="utf-8-sig")

    paths = apply_profiles(
        bakery_forecast_path,
        bakery_hour_profile_path,
        sku_hour_profile_path,
        tmp_path,
        output_suffix="smoothed",
    )

    sku_hourly = pd.read_csv(paths["sku_hourly"], encoding="utf-8-sig")
    sku_daily = pd.read_csv(paths["sku_daily"], encoding="utf-8-sig")

    assert round(float(sku_hourly["sku_hour_forecast"].sum()), 4) == 80.0
    assert round(float(sku_daily["sku_day_forecast"].sum()), 4) == 80.0
    assert sorted(sku_daily["product_id"].tolist()) == ["P1", "P2"]


def test_allocate_bakery_to_hour_uses_bakery_level_fallback_when_dow_missing():
    bakery_forecast = pd.DataFrame(
        [
            {"date": "2026-01-07", "dow": 2, "bakery_id": "B1", "bakery_name": "Bakery 1", "bakery_day_forecast": 40.0}
        ]
    )
    bakery_profile = pd.DataFrame(
        [
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "dow": 0, "hour": 8, "mean_hour_share_norm": 0.25},
            {"bakery_id": "B1", "bakery_name": "Bakery 1", "dow": 0, "hour": 9, "mean_hour_share_norm": 0.75},
        ]
    )

    hourly = allocate_bakery_to_hour(bakery_forecast, bakery_profile)
    assert len(hourly) == 2
    assert round(float(hourly["bakery_hour_forecast"].sum()), 4) == 40.0
    assert sorted(hourly["hour"].tolist()) == [8, 9]
