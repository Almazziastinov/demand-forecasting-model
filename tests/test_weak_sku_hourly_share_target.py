"""Tests for weak SKU hourly-share demand fallback."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.weak_sku_hourly_share_target import build_weak_hourly_share_target  # noqa: E402


def _daily_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Дата": pd.Timestamp("2026-01-05"),
                "Пекарня": "B1",
                "Категория": "Cat1",
                "Номенклатура": "P1",
                "Продано": 2.0,
                "Спрос": 2.0,
            },
            {
                "Дата": pd.Timestamp("2026-01-05"),
                "Пекарня": "B1",
                "Категория": "Cat1",
                "Номенклатура": "P2",
                "Продано": 5.0,
                "Спрос": 5.0,
            },
        ]
    )


def _weak_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Пекарня": "B1", "Номенклатура": "P1", "best_r2": -0.2, "is_weak_sku": True},
            {"Пекарня": "B1", "Номенклатура": "P2", "best_r2": 0.3, "is_weak_sku": False},
        ]
    )


def _hourly_daily() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Дата": pd.Timestamp("2026-01-05"),
                "Пекарня": "B1",
                "Категория": "Cat1",
                "Номенклатура": "P1",
                "observed_sales": 2.0,
                "expected_sales_from_hourly_profile": 4.0,
                "total_hourly_gap": 2.0,
                "profiled_hours": 5,
                "hourly_profile_positive_slots": 8,
            },
            {
                "Дата": pd.Timestamp("2026-01-05"),
                "Пекарня": "B1",
                "Категория": "Cat1",
                "Номенклатура": "P2",
                "observed_sales": 5.0,
                "expected_sales_from_hourly_profile": 7.0,
                "total_hourly_gap": 2.0,
                "profiled_hours": 5,
                "hourly_profile_positive_slots": 8,
            },
        ]
    )


def test_weak_hourly_share_target_applies_only_to_weak_sku():
    result = build_weak_hourly_share_target(_daily_df(), _weak_map(), _hourly_daily())
    weak_row = result[result["Номенклатура"] == "P1"].iloc[0]
    strong_row = result[result["Номенклатура"] == "P2"].iloc[0]

    assert bool(weak_row["weak_hourly_share_eligible"])
    assert weak_row["Спрос_weak_hourly_share"] == 4.0
    assert not bool(strong_row["weak_hourly_share_eligible"])
    assert strong_row["Спрос_weak_hourly_share"] == 5.0


def test_weak_hourly_share_target_respects_existing_higher_base_demand():
    daily = _daily_df()
    daily.loc[daily["Номенклатура"] == "P1", "Спрос"] = 4.5
    result = build_weak_hourly_share_target(daily, _weak_map(), _hourly_daily())
    weak_row = result[result["Номенклатура"] == "P1"].iloc[0]

    assert weak_row["Спрос_weak_hourly_share"] == 4.5
