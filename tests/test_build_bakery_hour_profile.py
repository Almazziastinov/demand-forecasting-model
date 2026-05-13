"""Tests for bakery-level hourly profile builder."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.build_bakery_hour_profile import aggregate_hourly_chunk  # noqa: E402
from src.experiments_v2.build_bakery_hour_profile import build_hour_profile  # noqa: E402


def _hourly() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "dow": 0, "bakery_id": "B1", "bakery_name": "Bakery 1", "hour": 8, "bakery_hour_sales": 2.0},
            {"date": pd.Timestamp("2026-01-05"), "dow": 0, "bakery_id": "B1", "bakery_name": "Bakery 1", "hour": 9, "bakery_hour_sales": 6.0},
            {"date": pd.Timestamp("2026-01-12"), "dow": 0, "bakery_id": "B1", "bakery_name": "Bakery 1", "hour": 8, "bakery_hour_sales": 1.0},
            {"date": pd.Timestamp("2026-01-12"), "dow": 0, "bakery_id": "B1", "bakery_name": "Bakery 1", "hour": 9, "bakery_hour_sales": 3.0},
        ]
    )


def test_build_hour_profile_normalizes_mean_share():
    profile, applied = build_hour_profile(_hourly())
    sums = profile.groupby(["bakery_id", "dow"])["mean_hour_share_norm"].sum()
    assert float(sums.iloc[0]) == 1.0
    hour8 = profile[profile["hour"] == 8].iloc[0]
    hour9 = profile[profile["hour"] == 9].iloc[0]
    assert round(float(hour8["mean_hour_share_norm"]), 4) == 0.25
    assert round(float(hour9["mean_hour_share_norm"]), 4) == 0.75


def test_aggregate_hourly_chunk_supports_legacy_russian_snapshot_columns():
    raw = pd.DataFrame(
        [
            {
                "Дата продажи": "01.01.2026",
                "Дата время чека": "01.01.2026 14:21:01",
                "Вид события по кассе": "Продажа",
                "Касса.Торговая точка": "Bakery Legacy",
                "Кол-во": 2.0,
            }
        ]
    )
    hourly = aggregate_hourly_chunk(raw)
    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["bakery_id"] == "Bakery Legacy"
    assert row["hour"] == 14
    assert row["bakery_hour_sales"] == 2.0
