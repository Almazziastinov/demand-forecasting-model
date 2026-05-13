"""Tests for experimental positive-only hourly profiles."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.hourly_positive_profile import (  # noqa: E402
    BAKERY_COL,
    CATEGORY_COL,
    DATE_COL,
    DOW_COL,
    HOUR_COL,
    PRODUCT_COL,
    apply_profiles,
    build_daily_from_applied,
    build_positive_profiles,
)


def _make_hourly() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {DATE_COL: pd.Timestamp("2026-01-05"), DOW_COL: 0, BAKERY_COL: "B1", CATEGORY_COL: "Cat1", PRODUCT_COL: "P1", HOUR_COL: 9, "sku_qty": 2.0, "bakery_qty": 10.0},
            {DATE_COL: pd.Timestamp("2026-01-12"), DOW_COL: 0, BAKERY_COL: "B1", CATEGORY_COL: "Cat1", PRODUCT_COL: "P1", HOUR_COL: 9, "sku_qty": 4.0, "bakery_qty": 20.0},
            {DATE_COL: pd.Timestamp("2026-01-19"), DOW_COL: 0, BAKERY_COL: "B1", CATEGORY_COL: "Cat1", PRODUCT_COL: "P1", HOUR_COL: 9, "sku_qty": 0.0, "bakery_qty": 15.0},
        ]
    )


def test_build_positive_profiles_uses_only_positive_slots():
    profiles = build_positive_profiles(_make_hourly())
    row = profiles.iloc[0]
    assert row["n_positive_slots"] == 2
    assert row["mean_share_positive"] == 0.2


def test_apply_profiles_and_daily_rollup():
    hourly = _make_hourly()
    profiles = build_positive_profiles(hourly)
    applied = apply_profiles(hourly, profiles)
    daily = build_daily_from_applied(applied)
    last_day = daily[daily[DATE_COL] == pd.Timestamp("2026-01-19")].iloc[0]
    assert last_day["expected_sales_from_hourly_profile"] == 3.0
    assert last_day["total_hourly_gap"] == 3.0
